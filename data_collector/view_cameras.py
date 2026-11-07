#!/usr/bin/env python3
"""Show the serial numbers and live color images of two RealSense cameras."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="查看两台 RealSense 相机的序列号和实时图像")
    parser.add_argument(
        "--serial", action="append", metavar="SERIAL",
        help="指定相机序列号；使用两次以选择两台相机",
    )
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pyrealsense2 as rs

    devices = [(device.get_info(rs.camera_info.serial_number),
                device.get_info(rs.camera_info.name))
               for device in rs.context().query_devices()]
    print("已连接的 RealSense 相机：")
    for serial, name in devices:
        print(f"  {name}: {serial}")
    if not devices:
        parser.error("未检测到 RealSense 相机")

    available = {serial for serial, _ in devices}
    if args.serial:
        if len(args.serial) != 2 or len(set(args.serial)) != 2:
            parser.error("请用两次 --serial 指定两个不同的序列号")
        if any(serial not in available for serial in args.serial):
            parser.error("指定的序列号不在已连接设备中")
        serials = args.serial
    else:
        if len(devices) != 2:
            parser.error("自动选择需要恰好两台相机；请用两次 --serial 指定相机")
        serials = [serial for serial, _ in devices]

    pipelines = []
    windows_created = False
    try:
        for serial in serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipeline.start(config)
            pipelines.append(pipeline)

        for index, serial in enumerate(serials, start=1):
            cv2.namedWindow(f"Camera {index} - {serial}", cv2.WINDOW_AUTOSIZE)
            windows_created = True

        print("按 q 或 Esc 退出预览。")
        while True:
            for index, (serial, pipeline) in enumerate(zip(serials, pipelines), start=1):
                frame = pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
                if frame:
                    image = cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_RGB2BGR)
                    cv2.putText(image, serial, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, serial, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(f"Camera {index} - {serial}", image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        for pipeline in reversed(pipelines):
            pipeline.stop()
        if windows_created:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
