"""Keep the newest two color frames available while inference is blocked."""

import threading
import time

import numpy as np


class ColorCamera:
    def __init__(self, serial, label, warmup_s=3.0):
        self.serial = serial
        self.label = label
        self.warmup_s = warmup_s
        self.pipeline = None
        self.started = False
        self.thread = None
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_time_ns = None
        self.error = None

    def start(self):
        import pyrealsense2 as rs

        matches = [device for device in rs.context().query_devices()
                   if device.get_info(rs.camera_info.serial_number) == self.serial]
        if len(matches) != 1:
            raise RuntimeError("%s camera serial is not uniquely available" % self.label)
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.pipeline = rs.pipeline()
        self.pipeline.start(config)
        self.started = True
        self.thread = threading.Thread(target=self._capture, name=self.label + "_camera", daemon=True)
        self.thread.start()

    def _capture(self):
        ready_at = time.monotonic() + self.warmup_s
        last_frame_at = time.monotonic()
        try:
            while not self.stop_event.is_set():
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=500)
                except RuntimeError:
                    if time.monotonic() - last_frame_at > 1.0:
                        raise RuntimeError("camera stopped delivering frames")
                    continue
                frame = frames.get_color_frame()
                if not frame:
                    continue
                received_ns = time.monotonic_ns()
                last_frame_at = time.monotonic()
                if last_frame_at < ready_at:
                    continue
                image = np.asanyarray(frame.get_data()).copy()
                if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                    raise RuntimeError("camera delivered an unexpected color frame")
                with self.lock:
                    self.latest_frame = image
                    self.latest_time_ns = received_ns
                self.ready.set()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.error = exc
                self.ready.set()

    def snapshot(self):
        if self.error is not None:
            raise RuntimeError("%s camera failed: %s" % (self.label, self.error)) from self.error
        with self.lock:
            if self.latest_frame is None:
                raise RuntimeError("%s camera is not ready" % self.label)
            return self.latest_frame.copy(), self.latest_time_ns

    def close(self):
        self.stop_event.set()
        if self.started:
            self.pipeline.stop()
            self.started = False
        if self.thread is not None:
            self.thread.join(timeout=2)


class DualCameras:
    def __init__(self, head_serial, wrist_serial, maximum_age_s=0.2, maximum_skew_s=0.08):
        if not head_serial or not wrist_serial or head_serial == wrist_serial:
            raise ValueError("two distinct camera serials are required")
        self.head = ColorCamera(head_serial, "head")
        self.wrist = ColorCamera(wrist_serial, "left_wrist")
        self.maximum_age_s = maximum_age_s
        self.maximum_skew_s = maximum_skew_s

    def start(self):
        try:
            self.head.start()
            self.wrist.start()
            for camera in (self.head, self.wrist):
                if not camera.ready.wait(timeout=camera.warmup_s + 5):
                    raise RuntimeError("%s camera did not become ready" % camera.label)
                camera.snapshot()
        except Exception:
            self.close()
            raise

    def snapshot(self):
        head, head_time = self.head.snapshot()
        wrist, wrist_time = self.wrist.snapshot()
        now = time.monotonic_ns()
        if now - head_time > self.maximum_age_s * 1e9 or now - wrist_time > self.maximum_age_s * 1e9:
            raise RuntimeError("camera frame is too old")
        if abs(head_time - wrist_time) > self.maximum_skew_s * 1e9:
            raise RuntimeError("camera frames are too far apart in time")
        return head, wrist

    def close(self):
        errors = []
        for camera in (self.head, self.wrist):
            try:
                camera.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("camera shutdown failed: %s" % errors[0]) from errors[0]
