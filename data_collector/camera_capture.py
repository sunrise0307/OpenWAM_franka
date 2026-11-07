"""Capture two color camera streams with host and device timestamps."""

import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time


def json_line(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class StreamWorker:
    def __init__(self, name):
        self.name = name
        self.error = None
        self.thread = None
        self.count = 0

    def launch(self, target):
        def run():
            try:
                target()
            except Exception as exc:
                self.error = exc

        self.thread = threading.Thread(target=run, name=self.name, daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"{self.name} failed: {self.error}") from self.error

    def join(self, timeout=8):
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                raise RuntimeError(f"{self.name} did not stop")
        self.check()


class ColorRecorder(StreamWorker):
    MAX_STREAM_SILENCE_NS = 1_000_000_000
    WARMUP_SECONDS = 3.0

    def __init__(self, slot, serial, root, stop, queue_size=64, clock_verifier=None,
                 source=None):
        super().__init__(slot)
        self.serial = serial
        self.root = root / slot
        self.root.mkdir(parents=True)
        self.stop = stop
        self.queue = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self.device_missing = 0
        self.last_frame_number = None
        self.last_device_timestamp_ms = None
        self.timestamp_domain = None
        self.max_receive_gap_ns = 0
        self.last_receive_ns = None
        self.first = threading.Event()
        self.capture = StreamWorker(slot + "_capture")
        self.writer = StreamWorker(slot + "_writer")
        self.pipeline = None
        self.profile_info = None
        self.clock_verifier = clock_verifier
        self.source = source

    def start(self):
        if self.source is not None:
            self.rs = self.source.rs
            self.profile_info = dict(self.source.profile_info)
            self.profile_info["warmup_seconds"] = self.source.WARMUP_SECONDS
            self.writer.launch(self._write)
            self.source.attach(self)
            return
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if self.clock_verifier is None:
            devices = [device for device in rs.context().query_devices()
                       if device.get_info(rs.camera_info.serial_number) == self.serial]
            if len(devices) != 1:
                raise RuntimeError(f"Camera {self.serial} is not uniquely available")
            sensors = [sensor for sensor in devices[0].query_sensors()
                       if sensor.get_info(rs.camera_info.name) == "RGB Camera"]
            if len(sensors) != 1 or not sensors[0].supports(rs.option.global_time_enabled):
                raise RuntimeError(f"Camera {self.serial} lacks RGB global time")
            sensors[0].set_option(rs.option.global_time_enabled, 1)
        profile = self.pipeline.start(config)
        device = profile.get_device()
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        color_sensor = next(sensor for sensor in device.query_sensors()
                            if sensor.get_info(rs.camera_info.name) == "RGB Camera")
        self.profile_info = {
            "serial": self.serial,
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "sdk_version": rs.__version__ if hasattr(rs, "__version__") else None,
            "format": str(stream.format()),
            "fps": stream.fps(),
            "width": stream.width(),
            "height": stream.height(),
            "recording_codec": "libx264rgb_lossless_qp0",
            "video_time_base": "1/30",
            "warmup_seconds": self.WARMUP_SECONDS,
            "global_time_enabled": (color_sensor.get_option(rs.option.global_time_enabled)
                                    if color_sensor.supports(rs.option.global_time_enabled)
                                    else None),
            "auto_exposure_enabled": (color_sensor.get_option(rs.option.enable_auto_exposure)
                                      if color_sensor.supports(rs.option.enable_auto_exposure)
                                      else None),
            "intrinsics": {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx,
                           "ppy": intr.ppy, "coeffs": list(intr.coeffs),
                           "model": str(intr.model)},
        }
        self.capture.launch(self._capture)
        self.writer.launch(self._write)

    def _capture(self):
        last_frame_ns = time.monotonic_ns()
        warmup_deadline_ns = last_frame_ns + int(self.WARMUP_SECONDS * 1_000_000_000)
        while not self.stop.is_set():
            before = time.monotonic_ns()
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError as exc:
                if not self.stop.is_set() and time.monotonic_ns() - last_frame_ns > 2_000_000_000:
                    raise RuntimeError("Color stream stopped for over 2 s") from exc
                continue
            after = time.monotonic_ns()
            wall_after = time.time_ns()
            anchor_end = time.monotonic_ns()
            frame = frames.get_color_frame()
            if not frame:
                if after - last_frame_ns > 2_000_000_000:
                    raise RuntimeError("No color frames received for over 2 s")
                continue
            last_frame_ns = after
            if after < warmup_deadline_ns:
                continue
            self._record_frame(frame, before, after, wall_after, anchor_end)

    def _record_frame(self, frame, before, after, wall_after, anchor_end):
        rs = self.rs
        keys = ("actual_exposure", "frame_timestamp", "sensor_timestamp",
                "time_of_arrival")
        number = int(frame.get_frame_number())
        if self.last_frame_number is not None and number <= self.last_frame_number:
            raise RuntimeError("Color frame number did not increase")
        if self.last_frame_number is not None:
            self.device_missing += max(0, number - self.last_frame_number - 1)
        self.last_frame_number = number
        device_timestamp_ms = float(frame.get_timestamp())
        if not math.isfinite(device_timestamp_ms):
            raise RuntimeError("Invalid color frame timestamp")
        domain = str(frame.get_frame_timestamp_domain())
        if self.timestamp_domain is not None and domain != self.timestamp_domain:
            raise RuntimeError("Color timestamp domain changed")
        if (self.last_device_timestamp_ms is not None and
                device_timestamp_ms <= self.last_device_timestamp_ms):
            raise RuntimeError("Color device timestamp did not increase")
        if self.last_receive_ns is not None:
            self.max_receive_gap_ns = max(self.max_receive_gap_ns,
                                          after - self.last_receive_ns)
        self.last_receive_ns = after
        self.last_device_timestamp_ms = device_timestamp_ms
        self.timestamp_domain = domain
        reference_ns = (self.clock_verifier.observe(self.name, device_timestamp_ms,
                        domain, after, 100_000_000) if self.clock_verifier else None)
        meta = {
            "frame_number": number,
            "host_timestamp_ns": after,
            "device_timestamp": device_timestamp_ms,
            "device_timestamp_unit": "ms",
            "device_clock_domain": domain,
            "valid": True,
            "device_timestamp_ms": device_timestamp_ms,
            "timestamp_domain": domain,
            "mapped_exposure_midpoint_ns": reference_ns,
            "host_receive_before_monotonic_ns": before,
            "host_receive_after_monotonic_ns": after,
            "host_realtime_minus_monotonic_ns": wall_after - (after + anchor_end) // 2,
            "host_clock_anchor_width_ns": anchor_end - after,
            "frame_metadata_raw": {},
        }
        for key in keys:
            enum = getattr(rs.frame_metadata_value, key, None)
            meta["frame_metadata_raw"][key] = (
                frame.get_frame_metadata(enum)
                if enum is not None and frame.supports_frame_metadata(enum)
                else None
            )
        import numpy as np

        image = np.asanyarray(frame.get_data()).copy()
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise RuntimeError("Unexpected RGB8 color frame shape")
        try:
            self.queue.put_nowait((image, meta))
            self.first.set()
        except queue.Full:
            self.dropped += 1

    def _write(self):
        import cv2
        import imageio_ffmpeg

        video_path = self.root / "color.mp4"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                   "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", "640x480", "-framerate", "30", "-i", "pipe:0",
                   "-an", "-c:v", "libx264rgb", "-preset", "ultrafast", "-qp", "0",
                   "-bf", "0", "-video_track_timescale", "30",
                   str(video_path)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            with open(self.root / "frames.jsonl", "w", encoding="utf-8") as index:
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        image, meta = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    pixels = image.tobytes()
                    process.stdin.write(pixels)
                    meta.update({"frame_index": self.count,
                                 "sequence": self.count, "file": "color.mp4",
                                 "video_frame_index": self.count,
                                 "video_pts": self.count, "video_time_base": "1/30",
                                 "rgb_sha256": hashlib.sha256(pixels).hexdigest(),
                                 "video_timestamp_is_exposure_time": False})
                    json_line(index, meta)
                    self.count += 1
                    self.queue.task_done()
            process.stdin.close()
            error = process.stderr.read().decode("utf-8", errors="replace")
            if process.wait() != 0:
                raise RuntimeError("Lossless RGB MP4 encoding failed: " + error[-1000:])
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError("Encoded RGB MP4 cannot be decoded")
            try:
                with open(self.root / "frames.jsonl", encoding="utf-8") as index:
                    for line in index:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError("Encoded RGB MP4 has fewer frames than index")
                        expected = json.loads(line)["rgb_sha256"]
                        actual = hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
                        if actual != expected:
                            raise RuntimeError("Encoded RGB MP4 differs from source pixels")
                if capture.read()[0]:
                    raise RuntimeError("Encoded RGB MP4 has unindexed frames")
            finally:
                capture.release()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stderr.close()

    def check(self):
        if self.source is not None:
            self.source.check()
        self.capture.check()
        self.writer.check()
        if (not self.stop.is_set() and self.last_receive_ns is not None and
                time.monotonic_ns() - self.last_receive_ns > self.MAX_STREAM_SILENCE_NS):
            raise RuntimeError(f"{self.name}: camera stream stalled")

    def close(self):
        stop_error = None
        if self.source is not None:
            self.source.detach(self)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as exc:
                if self.capture.thread is not None:
                    stop_error = exc
        for worker in (self.capture, self.writer):
            try:
                worker.join(timeout=120 if worker is self.writer else 8)
            except Exception as exc:
                stop_error = stop_error or exc
        if stop_error is not None:
            raise stop_error


class SessionCamera(StreamWorker):
    WARMUP_SECONDS = ColorRecorder.WARMUP_SECONDS

    def __init__(self, slot, serial, preview_path):
        super().__init__(slot + "_session")
        self.slot = slot
        self.serial = serial
        self.preview_path = preview_path
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.recorder = None
        self.pipeline = None
        self.rs = None
        self.profile_info = None
        self.last_receive_ns = None

    def start(self):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        devices = [device for device in rs.context().query_devices()
                   if device.get_info(rs.camera_info.serial_number) == self.serial]
        if len(devices) != 1:
            raise RuntimeError(f"Camera {self.serial} is not uniquely available")
        sensors = [sensor for sensor in devices[0].query_sensors()
                   if sensor.get_info(rs.camera_info.name) == "RGB Camera"]
        if len(sensors) != 1 or not sensors[0].supports(rs.option.global_time_enabled):
            raise RuntimeError(f"Camera {self.serial} lacks RGB global time")
        sensors[0].set_option(rs.option.global_time_enabled, 1)
        profile = self.pipeline.start(config)
        device = profile.get_device()
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        sensor = next(sensor for sensor in device.query_sensors()
                      if sensor.get_info(rs.camera_info.name) == "RGB Camera")
        self.profile_info = {
            "serial": self.serial,
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "sdk_version": rs.__version__ if hasattr(rs, "__version__") else None,
            "format": str(stream.format()), "fps": stream.fps(),
            "width": stream.width(), "height": stream.height(),
            "recording_codec": "libx264rgb_lossless_qp0", "video_time_base": "1/30",
            "warmup_seconds": self.WARMUP_SECONDS,
            "global_time_enabled": sensor.get_option(rs.option.global_time_enabled),
            "auto_exposure_enabled": (sensor.get_option(rs.option.enable_auto_exposure)
                                      if sensor.supports(rs.option.enable_auto_exposure)
                                      else None),
            "intrinsics": {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx,
                           "ppy": intr.ppy, "coeffs": list(intr.coeffs),
                           "model": str(intr.model)},
        }
        self.launch(self._capture)

    def attach(self, recorder):
        self.check()
        if not self.ready.is_set():
            raise RuntimeError(f"{self.slot}: camera has not warmed up")
        with self.lock:
            if self.recorder is not None:
                raise RuntimeError(f"{self.slot}: camera is already recording")
            self.recorder = recorder

    def detach(self, recorder):
        with self.lock:
            if self.recorder is recorder:
                self.recorder = None

    def _capture(self):
        import cv2
        import numpy as np

        last_frame_ns = time.monotonic_ns()
        deadline_ns = last_frame_ns + int(self.WARMUP_SECONDS * 1_000_000_000)
        next_preview_ns = 0
        while not self.stop.is_set():
            before = time.monotonic_ns()
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError as exc:
                if not self.stop.is_set() and time.monotonic_ns() - last_frame_ns > 2_000_000_000:
                    raise RuntimeError("Color stream stopped for over 2 s") from exc
                continue
            after = time.monotonic_ns()
            wall_after = time.time_ns()
            anchor_end = time.monotonic_ns()
            frame = frames.get_color_frame()
            if not frame:
                if after - last_frame_ns > 2_000_000_000:
                    raise RuntimeError("No color frames received for over 2 s")
                continue
            last_frame_ns = after
            self.last_receive_ns = after
            if after < deadline_ns:
                continue
            with self.lock:
                recorder = self.recorder
                if recorder is not None:
                    recorder._record_frame(frame, before, after, wall_after, anchor_end)
            if recorder is None and after >= next_preview_ns:
                image = np.asanyarray(frame.get_data())
                encoded, data = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if not encoded:
                    raise RuntimeError("Camera preview encoding failed")
                temporary = self.preview_path.with_name(self.preview_path.name + ".tmp")
                temporary.write_bytes(data.tobytes())
                os.replace(temporary, self.preview_path)
                self.ready.set()
                next_preview_ns = after + 1_000_000_000

    def check(self):
        super().check()
        if (self.last_receive_ns is not None and
                time.monotonic_ns() - self.last_receive_ns >
                ColorRecorder.MAX_STREAM_SILENCE_NS):
            raise RuntimeError(f"{self.slot}: camera stream stalled")

    def close(self):
        self.stop.set()
        stop_error = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError as exc:
                if self.thread is not None:
                    stop_error = exc
        self.join()
        if stop_error is not None:
            raise stop_error


