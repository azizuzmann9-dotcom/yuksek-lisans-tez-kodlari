#!/usr/bin/env python3

import math
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray


class CameraViewer(Node):
    # Önceki merkez toleransı korunur.
    CENTER_TOLERANCE_X = 20
    CENTER_TOLERANCE_Y = 15
    CENTER_HOLD_TIME = 0.35

    # İyi sonuç veren beş noktalı sürümün PATH ayarları korunur.
    PATH_TOLERANCE_NORM = 0.04
    PATH_HOLD_TIME = 0.50

    MIN_CONTOUR_AREA = 1000
    BOX_SMOOTHING_ALPHA = 0.25
    REACQUIRE_AFTER_LOST_FRAMES = 15
    SHOW_WINDOWS = True

    MODE_CENTER = 0.0
    MODE_PATH = 1.0
    MODE_DONE = 2.0

    def __init__(self):
        super().__init__('camera_viewer')

        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            '/iris/iris_camera/image_raw',
            self.image_callback,
            10,
        )
        self.create_subscription(
            Bool,
            '/offboard_active',
            self.offboard_callback,
            10,
        )

        self.error_publisher = self.create_publisher(
            Float32MultiArray,
            '/window_error',
            10,
        )

        # Pencere merkezinden başlayan yedi hedefli zikzak rota.
        # Başlangıç merkezi (0.50, 0.50) kontrol algoritmasında örtük olarak
        # kabul edilir; ilk hedef sol-orta noktadır.
        self.path_uv = [
            (0.22, 0.50),  # 1: sol orta
            (0.22, 0.82),  # 2: sol alt
            (0.78, 0.82),  # 3: sağ alt
            (0.78, 0.50),  # 4: sağ orta
            (0.22, 0.50),  # 5: sol orta
            (0.22, 0.18),  # 6: sol üst
            (0.78, 0.18),  # 7: sağ üst
        ]

        self.offboard_active = False
        self.reset_mission(clear_window_lock=True)

        self.get_logger().info(
            'Camera viewer started: V13 7-point zigzag segment control, center 20x15 px'
        )
        self.get_logger().info(
            f'CENTER X={self.CENTER_TOLERANCE_X}px, '
            f'Y={self.CENTER_TOLERANCE_Y}px, '
            f'hold={self.CENTER_HOLD_TIME:.2f}s'
        )
        self.get_logger().info(
            f'PATH tolerance={self.PATH_TOLERANCE_NORM:.3f}, '
            f'hold={self.PATH_HOLD_TIME:.2f}s'
        )

    def reset_mission(self, clear_window_lock=True):
        self.mode = 'CENTER'
        self.path_index = 0
        self.pump_requested = False
        self.lost_frames = 0

        self.center_hold_start = None
        self.center_hold_elapsed = 0.0
        self.center_inside_tolerance = False

        self.path_hold_start = None
        self.path_hold_elapsed = 0.0
        self.path_inside_tolerance = False

        if clear_window_lock:
            self.locked_box = None
            self.smoothed_box = None

    def reset_center_hold(self):
        self.center_hold_start = None
        self.center_hold_elapsed = 0.0
        self.center_inside_tolerance = False

    def reset_path_hold(self):
        self.path_hold_start = None
        self.path_hold_elapsed = 0.0
        self.path_inside_tolerance = False

    def offboard_callback(self, msg):
        new_state = bool(msg.data)

        if self.offboard_active and not new_state:
            self.reset_mission(clear_window_lock=True)
            self.get_logger().info(
                'OFFBOARD kapandi: gorev ve pencere kilidi sifirlandi'
            )

        self.offboard_active = new_state

    @staticmethod
    def box_center(box):
        x, y, width, height = box
        return x + width / 2.0, y + height / 2.0

    def choose_same_window(self, candidates):
        if not candidates:
            return None

        if (
            self.locked_box is None
            or self.lost_frames >= self.REACQUIRE_AFTER_LOST_FRAMES
        ):
            return max(candidates, key=lambda item: item[4])[:4]

        previous_cx, previous_cy = self.box_center(self.locked_box)
        _, _, previous_width, previous_height = self.locked_box
        previous_diag = max(1.0, math.hypot(previous_width, previous_height))

        best_box = None
        best_score = float('inf')

        for x, y, width, height, _area in candidates:
            cx = x + width / 2.0
            cy = y + height / 2.0

            center_distance = math.hypot(cx - previous_cx, cy - previous_cy)
            size_difference = (
                abs(width - previous_width) / max(previous_width, 1.0)
                + abs(height - previous_height) / max(previous_height, 1.0)
            )

            if center_distance > max(140.0, previous_diag * 0.75):
                continue

            score = center_distance / previous_diag + 0.45 * size_difference

            if score < best_score:
                best_score = score
                best_box = (x, y, width, height)

        return best_box

    def smooth_box(self, detected_box):
        if self.smoothed_box is None:
            self.smoothed_box = tuple(float(value) for value in detected_box)
        else:
            alpha = self.BOX_SMOOTHING_ALPHA
            self.smoothed_box = tuple(
                alpha * new_value + (1.0 - alpha) * old_value
                for old_value, new_value in zip(self.smoothed_box, detected_box)
            )

        return tuple(int(round(value)) for value in self.smoothed_box)

    def segment_code_for_index(self, path_index):
        """Aktif hedefe giden ana hareket eksenini kodlar.

        1.0 = yatay segment, 2.0 = düşey segment. İlk hedef için
        başlangıç noktası pencere merkezi kabul edilir.
        """
        if not self.path_uv:
            return 0.0

        index = max(0, min(int(path_index), len(self.path_uv) - 1))
        target_u, target_v = self.path_uv[index]

        if index == 0:
            previous_u, previous_v = 0.50, 0.50
        else:
            previous_u, previous_v = self.path_uv[index - 1]

        delta_u = target_u - previous_u
        delta_v = target_v - previous_v

        if abs(delta_u) >= abs(delta_v):
            return 1.0
        return 2.0

    def publish_state(
        self,
        error_x,
        error_y,
        window_found,
        target_u=float('nan'),
        target_v=float('nan'),
        actual_u=float('nan'),
        actual_v=float('nan'),
    ):
        mode_code = {
            'CENTER': self.MODE_CENTER,
            'PATH': self.MODE_PATH,
            'DONE': self.MODE_DONE,
        }[self.mode]

        msg_out = Float32MultiArray()
        msg_out.data = [
            float(error_x),
            float(error_y),
            float(mode_code),
            float(self.path_index),
            1.0 if window_found else 0.0,
            1.0 if self.pump_requested else 0.0,
            float(target_u),
            float(target_v),
            float(actual_u),
            float(actual_v),
            float(self.lost_frames),
            float(self.path_hold_elapsed),
            1.0 if self.path_inside_tolerance else 0.0,
            (
                self.segment_code_for_index(self.path_index)
                if self.mode == 'PATH'
                else 0.0
            ),
        ]
        self.error_publisher.publish(msg_out)

    def show_images(self, frame, edges):
        if not self.SHOW_WINDOWS:
            return

        cv2.imshow('cam', frame)
        cv2.imshow('edges', edges)
        cv2.waitKey(1)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_height <= 0:
                continue

            ratio = box_width / float(box_height)
            if 0.6 < ratio < 1.8:
                candidates.append((x, y, box_width, box_height, area))

        detected_box = self.choose_same_window(candidates)

        frame_height, frame_width, _ = frame.shape
        image_center_x = frame_width // 2
        image_center_y = frame_height // 2

        cv2.circle(
            frame,
            (image_center_x, image_center_y),
            5,
            (255, 0, 0),
            -1,
        )

        if self.mode == 'CENTER':
            cv2.rectangle(
                frame,
                (
                    image_center_x - self.CENTER_TOLERANCE_X,
                    image_center_y - self.CENTER_TOLERANCE_Y,
                ),
                (
                    image_center_x + self.CENTER_TOLERANCE_X,
                    image_center_y + self.CENTER_TOLERANCE_Y,
                ),
                (255, 255, 0),
                1,
            )

        if detected_box is None:
            self.lost_frames += 1
            self.reset_center_hold()
            self.reset_path_hold()

            if self.lost_frames == self.REACQUIRE_AFTER_LOST_FRAMES:
                self.locked_box = None
                self.smoothed_box = None

            self.publish_state(9999.0, 9999.0, window_found=False)

            cv2.putText(
                frame,
                'PENCERE YOK - HOLD',
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
            self.show_images(frame, edges)
            return

        self.lost_frames = 0
        self.locked_box = detected_box

        x, y, box_width, box_height = self.smooth_box(detected_box)
        window_center_x = x + box_width // 2
        window_center_y = y + box_height // 2

        actual_u = (image_center_x - x) / max(float(box_width), 1.0)
        actual_v = (image_center_y - y) / max(float(box_height), 1.0)

        cv2.rectangle(
            frame,
            (x, y),
            (x + box_width, y + box_height),
            (0, 255, 0),
            2,
        )
        cv2.circle(
            frame,
            (window_center_x, window_center_y),
            5,
            (0, 0, 255),
            -1,
        )

        pixel_path = []
        for u, v in self.path_uv:
            target_x = int(round(x + u * box_width))
            target_y = int(round(y + v * box_height))
            pixel_path.append((target_x, target_y))

        for index, point in enumerate(pixel_path):
            cv2.circle(frame, point, 5, (0, 0, 200), -1)
            if index > 0:
                cv2.line(
                    frame,
                    pixel_path[index - 1],
                    point,
                    (0, 0, 255),
                    2,
                )

        target_u = 0.50
        target_v = 0.50

        if self.mode == 'CENTER':
            self.reset_path_hold()

            error_x = window_center_x - image_center_x
            error_y = window_center_y - image_center_y

            self.center_inside_tolerance = (
                abs(error_x) <= self.CENTER_TOLERANCE_X
                and abs(error_y) <= self.CENTER_TOLERANCE_Y
            )

            now = time.monotonic()

            if self.center_inside_tolerance and self.offboard_active:
                if self.center_hold_start is None:
                    self.center_hold_start = now

                self.center_hold_elapsed = now - self.center_hold_start

                if self.center_hold_elapsed >= self.CENTER_HOLD_TIME:
                    self.mode = 'PATH'
                    self.path_index = 0
                    self.pump_requested = True
                    self.reset_center_hold()
                    self.reset_path_hold()

                    target_u, target_v = self.path_uv[self.path_index]
                    target_x, target_y = pixel_path[self.path_index]
                    error_x = target_x - image_center_x
                    error_y = target_y - image_center_y

                    self.get_logger().info(
                        'Merkez sabitlendi: pompa ve PATH basladi'
                    )
            else:
                self.reset_center_hold()

        elif self.mode == 'PATH':
            self.reset_center_hold()

            target_u, target_v = self.path_uv[self.path_index]
            target_x, target_y = pixel_path[self.path_index]

            error_x = target_x - image_center_x
            error_y = target_y - image_center_y

            normalized_error_u = target_u - actual_u
            normalized_error_v = target_v - actual_v

            self.path_inside_tolerance = (
                abs(normalized_error_u) <= self.PATH_TOLERANCE_NORM
                and abs(normalized_error_v) <= self.PATH_TOLERANCE_NORM
            )

            now = time.monotonic()

            if self.path_inside_tolerance:
                if self.path_hold_start is None:
                    self.path_hold_start = now

                self.path_hold_elapsed = now - self.path_hold_start

                if self.path_hold_elapsed >= self.PATH_HOLD_TIME:
                    if self.path_index < len(self.path_uv) - 1:
                        self.path_index += 1
                        self.reset_path_hold()

                        target_u, target_v = self.path_uv[self.path_index]
                        target_x, target_y = pixel_path[self.path_index]
                        error_x = target_x - image_center_x
                        error_y = target_y - image_center_y

                        self.get_logger().info(
                            f'PATH hedefi degisti: {self.path_index + 1}/'
                            f'{len(self.path_uv)}'
                        )
                    else:
                        self.mode = 'DONE'
                        self.reset_path_hold()
                        error_x = 0.0
                        error_y = 0.0
                        target_u, target_v = self.path_uv[-1]

                        self.get_logger().info(
                            'PATH tamamlandi: hareket HOLD, pompa acik'
                        )
            else:
                self.reset_path_hold()

        else:
            self.reset_center_hold()
            self.reset_path_hold()
            target_u, target_v = self.path_uv[-1]
            error_x = 0.0
            error_y = 0.0

        if self.mode == 'PATH':
            target_x, target_y = pixel_path[self.path_index]
            cv2.circle(frame, (target_x, target_y), 9, (0, 255, 255), -1)

        self.publish_state(
            error_x,
            error_y,
            window_found=True,
            target_u=target_u,
            target_v=target_v,
            actual_u=actual_u,
            actual_v=actual_v,
        )

        cv2.putText(
            frame,
            f'MODE: {self.mode}  PATH: {self.path_index + 1}/'
            f'{len(self.path_uv)}',
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            'PUMP: ' + ('ON' if self.pump_requested else 'OFF'),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            'OFFBOARD: ' + ('ON' if self.offboard_active else 'OFF'),
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        if self.mode == 'CENTER':
            cv2.putText(
                frame,
                f'CENTER HOLD: {self.center_hold_elapsed:.2f}/'
                f'{self.CENTER_HOLD_TIME:.2f}s',
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        if self.mode == 'PATH':
            cv2.putText(
                frame,
                f'HOLD: {self.path_hold_elapsed:.2f}/'
                f'{self.PATH_HOLD_TIME:.2f}s',
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        self.show_images(frame, edges)


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
