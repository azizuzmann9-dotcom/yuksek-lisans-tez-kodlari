#!/usr/bin/env python3

import csv
import math
import time
from pathlib import Path

import rclpy
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32MultiArray


class WindowOffboard(Node):
    MODE_CENTER = 0
    MODE_PATH = 1
    MODE_DONE = 2

    # İstenen yeni temel kazançlar.
    SIDE_KP = 0.0014
    SIDE_KD = 0.00055
    VERTICAL_KP = 0.0016
    FORWARD_HOLD_KP = 0.20

    MAX_SIDE_SPEED = 0.15
    MAX_VERTICAL_SPEED = 0.15
    MAX_FORWARD_SPEED = 0.04

    # CENTER daha sakin hareket eder.
    CENTER_MAX_SIDE_SPEED = 0.10
    CENTER_MAX_VERTICAL_SPEED = 0.10

    # 3. hedefteki uzun yatay geçişe özel fren.
    PATH3_SIDE_KD = 0.00075
    PATH3_MAX_SIDE_SPEED = 0.090
    PATH3_NEAR_DISTANCE_PX = 60.0
    PATH3_FINAL_DISTANCE_PX = 28.0
    PATH3_NEAR_SPEED = 0.055
    PATH3_FINAL_SPEED = 0.030

    # Düşey segmentte önce kısa X hizalama.
    X_ALIGN_ENTER_TOLERANCE = 7.0
    X_ALIGN_EXIT_TOLERANCE = 26.0
    X_ALIGN_HOLD_TIME = 0.18
    VERTICAL_PREALIGN_MAX_SIDE_SPEED = 0.080

    # Düşey hareket sırasında küçük X konum tutma.
    X_HOLD_DEADBAND = 5.0
    X_HOLD_KP = 0.0009
    X_HOLD_KD = 0.00018
    MAX_X_HOLD_SPEED = 0.022
    MAX_X_HOLD_D_TERM = 0.018

    # Yatay segment sırasında küçük Y konum tutma.
    Y_HOLD_DEADBAND = 5.0
    Y_HOLD_KP = 0.0010
    MAX_Y_HOLD_SPEED = 0.025

    DERIVATIVE_ALPHA = 0.20
    MAX_SIDE_D_TERM = 0.08

    DEADBAND_X = 8.0
    DEADBAND_Y = 8.0
    VISION_TIMEOUT = 0.50

    PHASE_CENTER = 'CENTER'
    PHASE_VERTICAL_X = 'VERTICAL_X_ALIGN'
    PHASE_VERTICAL_Y = 'VERTICAL_Y_MOVE'
    PHASE_HORIZONTAL = 'HORIZONTAL_MOVE'
    PHASE_HOLD = 'HOLD'

    def __init__(self):
        super().__init__('window_offboard')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile,
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile,
        )
        self.pump_pub = self.create_publisher(Bool, '/pump_active', 10)
        self.offboard_state_pub = self.create_publisher(
            Bool,
            '/offboard_active',
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            '/window_error',
            self.error_callback,
            10,
        )
        self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile,
        )
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position_v1',
            self.position_callback,
            qos_profile,
        )
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile,
        )

        self.error_x = 9999.0
        self.error_y = 9999.0
        self.window_found = False

        self.mission_mode = self.MODE_CENTER
        self.path_index = 0
        self.path_segment_code = 0
        self.pump_requested = False
        self.lost_frames = 0
        self.path_hold_elapsed = 0.0
        self.path_inside_tolerance = False

        self.target_u = float('nan')
        self.target_v = float('nan')
        self.actual_u = float('nan')
        self.actual_v = float('nan')

        self.current_yaw = 0.0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0

        self.hold_y = None
        self.offboard_active = False
        self.pump_active = False

        self.control_phase = self.PHASE_CENTER
        self.x_align_hold_start = None
        self.x_align_hold_elapsed = 0.0
        self.vertical_realign_count = 0

        self.last_vision_message_monotonic = 0.0
        self.last_control_monotonic = time.monotonic()
        self.last_error_x = None
        self.filtered_derivative_x = 0.0
        self.derivative_reset_requested = True

        self.last_print_time = time.monotonic()
        self.last_log_flush = time.monotonic()
        self.log_start_monotonic = time.monotonic()

        log_dir = Path.home() / 'drone_ws' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        log_name = (
            'gazebo_flight_log_zigzag_7point_v13_'
            + time.strftime('%Y%m%d_%H%M%S')
            + '.csv'
        )
        self.log_path = log_dir / log_name
        self.log_file = open(self.log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)

        self.log_header = [
            'time',
            'offboard_active',
            'vision_data_fresh',
            'window_found',
            'mission_mode',
            'path_index',
            'lost_frames',
            'path_inside_tolerance',
            'path_hold_elapsed',
            'error_x',
            'error_y',
            'target_u',
            'target_v',
            'actual_u',
            'actual_v',
            'current_x',
            'current_y',
            'current_z',
            'vx_cmd',
            'vy_cmd',
            'vz_cmd',
            'yaw_cmd',
            'side_kp',
            'side_kd',
            'vertical_kp',
            'derivative_x_raw',
            'derivative_x_filtered',
            'side_p_term',
            'side_d_term',
            'pump_requested',
            'pump_active',

            'elapsed_time_s',
            'current_vx',
            'current_vy',
            'current_vz',
            'segment_type',
            'control_phase',
            'x_align_hold_elapsed',
            'vertical_realign_count',
            'active_side_kp',
            'active_side_kd',
            'active_side_speed_limit',
            'x_hold_p_term',
            'x_hold_d_term',
            'y_hold_p_term',
            'path3_brake_zone',
        ]
        self.csv_writer.writerow(self.log_header)
        self.log_file.flush()

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(
            'Controller started: V13 7-point zigzag segment-based control'
        )
        self.get_logger().info(
            f'SIDE_KP={self.SIDE_KP}, SIDE_KD={self.SIDE_KD}, '
            f'VERTICAL_KP={self.VERTICAL_KP}'
        )
        self.get_logger().info(
            f'X realign threshold={self.X_ALIGN_EXIT_TOLERANCE}px'
        )
        self.get_logger().info(f'CSV kayit basladi: {self.log_path}')

    @staticmethod
    def clamp(value, limit):
        return max(-limit, min(limit, value))

    def request_derivative_reset(self):
        self.derivative_reset_requested = True
        self.last_error_x = None
        self.filtered_derivative_x = 0.0

    def segment_type(self):
        if self.mission_mode == self.MODE_CENTER:
            return 'CENTER'
        if self.mission_mode != self.MODE_PATH:
            return 'HOLD'

        # Kamera düğümü aktif zikzak hedefinin ana hareket eksenini
        # 14. mesaj alanında gönderir. Eski 13 alanlı mesajlarla geriye
        # uyumluluk için güvenli bir indeks yedeği korunur.
        if self.path_segment_code == 1:
            return 'HORIZONTAL'
        if self.path_segment_code == 2:
            return 'VERTICAL'

        return 'VERTICAL' if self.path_index % 2 == 1 else 'HORIZONTAL'

    def reset_segment_control(self):
        segment_type = self.segment_type()

        self.x_align_hold_start = None
        self.x_align_hold_elapsed = 0.0

        if segment_type == 'VERTICAL':
            self.control_phase = self.PHASE_VERTICAL_X
        elif segment_type == 'HORIZONTAL':
            self.control_phase = self.PHASE_HORIZONTAL
        elif segment_type == 'CENTER':
            self.control_phase = self.PHASE_CENTER
        else:
            self.control_phase = self.PHASE_HOLD

        self.request_derivative_reset()

    def error_callback(self, msg):
        if len(msg.data) < 13:
            self.get_logger().warning(
                'window_error mesaji eski formatta; 13 alan bekleniyor'
            )
            return

        previous_mode = self.mission_mode
        previous_path_index = self.path_index
        previous_window_found = self.window_found

        self.error_x = float(msg.data[0])
        self.error_y = float(msg.data[1])
        self.mission_mode = int(round(msg.data[2]))
        self.path_index = int(round(msg.data[3]))
        self.window_found = bool(round(msg.data[4]))
        self.pump_requested = bool(round(msg.data[5]))
        self.target_u = float(msg.data[6])
        self.target_v = float(msg.data[7])
        self.actual_u = float(msg.data[8])
        self.actual_v = float(msg.data[9])
        self.lost_frames = int(round(msg.data[10]))
        self.path_hold_elapsed = float(msg.data[11])
        self.path_inside_tolerance = bool(round(msg.data[12]))
        if len(msg.data) >= 14:
            self.path_segment_code = int(round(msg.data[13]))
        else:
            self.path_segment_code = 0

        self.last_vision_message_monotonic = time.monotonic()

        if (
            self.mission_mode != previous_mode
            or self.path_index != previous_path_index
            or (previous_window_found and not self.window_found)
        ):
            self.reset_segment_control()

    def attitude_callback(self, msg):
        q = msg.q
        w, x, y, z = q[0], q[1], q[2], q[3]

        self.current_yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def position_callback(self, msg):
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_z = msg.z

        self.current_vx = msg.vx if math.isfinite(msg.vx) else 0.0
        self.current_vy = msg.vy if math.isfinite(msg.vy) else 0.0
        self.current_vz = msg.vz if math.isfinite(msg.vz) else 0.0

    def vehicle_status_callback(self, msg):
        new_offboard_active = (
            msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        )

        if not self.offboard_active and new_offboard_active:
            self.hold_y = self.current_y
            self.reset_segment_control()
            self.get_logger().info(
                f'OFFBOARD acildi: Y mesafesi kilitlendi ({self.hold_y:.2f})'
            )

        if self.offboard_active and not new_offboard_active:
            self.pump_active = False
            self.hold_y = None
            self.reset_segment_control()
            self.get_logger().info(
                'OFFBOARD kapandi: pompa ve kontrol fazi sifirlandi'
            )

        self.offboard_active = new_offboard_active

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.offboard_control_mode_pub.publish(msg)

    def publish_velocity_setpoint(self, vx, vy, vz, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yaw = float(yaw)
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        self.trajectory_setpoint_pub.publish(msg)

    def publish_states(self):
        pump_msg = Bool()
        pump_msg.data = self.pump_active
        self.pump_pub.publish(pump_msg)

        offboard_msg = Bool()
        offboard_msg.data = self.offboard_active
        self.offboard_state_pub.publish(offboard_msg)

    def horizontal_parameters(self):
        active_kp = self.SIDE_KP
        active_kd = self.SIDE_KD
        speed_limit = self.MAX_SIDE_SPEED
        brake_zone = 'NORMAL'

        if (
            self.mission_mode == self.MODE_PATH
            and self.segment_type() == 'HORIZONTAL'
            and self.path_index == 2
        ):
            active_kd = self.PATH3_SIDE_KD
            absolute_error = abs(self.error_x)

            if absolute_error <= self.PATH3_FINAL_DISTANCE_PX:
                speed_limit = self.PATH3_FINAL_SPEED
                brake_zone = 'PATH3_FINAL'
            elif absolute_error <= self.PATH3_NEAR_DISTANCE_PX:
                speed_limit = self.PATH3_NEAR_SPEED
                brake_zone = 'PATH3_NEAR'
            else:
                speed_limit = self.PATH3_MAX_SIDE_SPEED
                brake_zone = 'PATH3_FAR'

        return active_kp, active_kd, speed_limit, brake_zone

    def write_log(
        self,
        vx,
        vy,
        vz,
        vision_data_fresh,
        derivative_x_raw,
        side_p_term,
        side_d_term,
        segment_type,
        active_side_kp,
        active_side_kd,
        active_side_speed_limit,
        x_hold_p_term,
        x_hold_d_term,
        y_hold_p_term,
        path3_brake_zone,
    ):
        row = [
            time.time(),
            self.offboard_active,
            vision_data_fresh,
            self.window_found,
            self.mission_mode,
            self.path_index,
            self.lost_frames,
            self.path_inside_tolerance,
            self.path_hold_elapsed,
            self.error_x,
            self.error_y,
            self.target_u,
            self.target_v,
            self.actual_u,
            self.actual_v,
            self.current_x,
            self.current_y,
            self.current_z,
            vx,
            vy,
            vz,
            self.current_yaw,
            self.SIDE_KP,
            self.SIDE_KD,
            self.VERTICAL_KP,
            derivative_x_raw,
            self.filtered_derivative_x,
            side_p_term,
            side_d_term,
            self.pump_requested,
            self.pump_active,

            time.monotonic() - self.log_start_monotonic,
            self.current_vx,
            self.current_vy,
            self.current_vz,
            segment_type,
            self.control_phase,
            self.x_align_hold_elapsed,
            self.vertical_realign_count,
            active_side_kp,
            active_side_kd,
            active_side_speed_limit,
            x_hold_p_term,
            x_hold_d_term,
            y_hold_p_term,
            path3_brake_zone,
        ]

        if len(row) != len(self.log_header):
            raise RuntimeError(
                f'CSV kolon sayisi uyusmuyor: '
                f'{len(row)} != {len(self.log_header)}'
            )

        self.csv_writer.writerow(row)

        now = time.monotonic()
        if now - self.last_log_flush >= 0.5:
            self.log_file.flush()
            self.last_log_flush = now

    def timer_callback(self):
        self.publish_offboard_control_mode()

        now = time.monotonic()
        dt = max(0.001, min(0.20, now - self.last_control_monotonic))
        self.last_control_monotonic = now

        vx = 0.0
        vy = 0.0
        vz = 0.0

        derivative_x_raw = 0.0
        side_p_term = 0.0
        side_d_term = 0.0
        x_hold_p_term = 0.0
        x_hold_d_term = 0.0
        y_hold_p_term = 0.0

        active_side_kp = self.SIDE_KP
        active_side_kd = self.SIDE_KD
        active_side_speed_limit = self.MAX_SIDE_SPEED
        path3_brake_zone = 'NORMAL'

        segment_type = self.segment_type()
        x_cmd = 'HOLD'
        y_cmd = 'HOLD'
        move_cmd = 'BEKLE'

        vision_data_fresh = (
            now - self.last_vision_message_monotonic <= self.VISION_TIMEOUT
        )

        tracking_allowed = (
            self.offboard_active
            and vision_data_fresh
            and self.window_found
            and self.mission_mode != self.MODE_DONE
        )

        if tracking_allowed:
            if self.derivative_reset_requested or self.last_error_x is None:
                derivative_x_raw = 0.0
                self.filtered_derivative_x = 0.0
                self.last_error_x = self.error_x
                self.derivative_reset_requested = False
            else:
                derivative_x_raw = (
                    self.error_x - self.last_error_x
                ) / dt
                self.filtered_derivative_x = (
                    self.DERIVATIVE_ALPHA * derivative_x_raw
                    + (1.0 - self.DERIVATIVE_ALPHA)
                    * self.filtered_derivative_x
                )
                self.last_error_x = self.error_x

            (
                active_side_kp,
                active_side_kd,
                active_side_speed_limit,
                path3_brake_zone,
            ) = self.horizontal_parameters()

            side_p_term = active_side_kp * self.error_x
            side_d_term = (
                active_side_kd * self.filtered_derivative_x
            )
            side_d_term = self.clamp(
                side_d_term,
                self.MAX_SIDE_D_TERM,
            )

            if segment_type == 'CENTER':
                self.control_phase = self.PHASE_CENTER

                if abs(self.error_x) > self.DEADBAND_X:
                    vx = self.clamp(
                        -(side_p_term + side_d_term),
                        self.CENTER_MAX_SIDE_SPEED,
                    )
                    x_cmd = 'CENTER X PD'
                else:
                    vx = 0.0
                    x_cmd = 'CENTER X TAMAM'

                if self.error_y > self.DEADBAND_Y:
                    vz = min(
                        self.CENTER_MAX_VERTICAL_SPEED,
                        self.VERTICAL_KP * abs(self.error_y),
                    )
                    y_cmd = 'CENTER ASAGI'
                elif self.error_y < -self.DEADBAND_Y:
                    vz = -min(
                        self.CENTER_MAX_VERTICAL_SPEED,
                        self.VERTICAL_KP * abs(self.error_y),
                    )
                    y_cmd = 'CENTER YUKARI'
                else:
                    vz = 0.0
                    y_cmd = 'CENTER Y TAMAM'

                move_cmd = 'CENTER'

            elif segment_type == 'HORIZONTAL':
                self.control_phase = self.PHASE_HORIZONTAL

                # Yatay segmentte doğrudan yatay PD.
                if abs(self.error_x) > self.DEADBAND_X:
                    vx = self.clamp(
                        -(side_p_term + side_d_term),
                        active_side_speed_limit,
                    )
                    x_cmd = 'YATAY ANA PD'
                else:
                    vx = 0.0
                    x_cmd = 'X TAMAM'

                # Düşey eksen yalnız küçük konum tutma yapar.
                if abs(self.error_y) > self.Y_HOLD_DEADBAND:
                    y_hold_p_term = self.Y_HOLD_KP * self.error_y
                    vz = self.clamp(
                        y_hold_p_term,
                        self.MAX_Y_HOLD_SPEED,
                    )
                    y_cmd = 'Y KÜÇÜK HOLD'
                else:
                    vz = 0.0
                    y_cmd = 'Y HOLD'

                move_cmd = 'YATAY SEGMENT'

            elif segment_type == 'VERTICAL':
                if self.control_phase not in (
                    self.PHASE_VERTICAL_X,
                    self.PHASE_VERTICAL_Y,
                ):
                    self.control_phase = self.PHASE_VERTICAL_X

                if self.control_phase == self.PHASE_VERTICAL_X:
                    # Önce yalnız kısa yatay hizalama; düşey hareket yok.
                    vz = 0.0
                    y_cmd = 'Y BEKLE'

                    if abs(self.error_x) <= self.X_ALIGN_ENTER_TOLERANCE:
                        if self.x_align_hold_start is None:
                            self.x_align_hold_start = now

                        self.x_align_hold_elapsed = (
                            now - self.x_align_hold_start
                        )
                        vx = 0.0
                        x_cmd = 'X SABITLE'

                        if (
                            self.x_align_hold_elapsed
                            >= self.X_ALIGN_HOLD_TIME
                        ):
                            self.control_phase = self.PHASE_VERTICAL_Y
                            self.x_align_hold_start = None
                            self.x_align_hold_elapsed = 0.0
                            x_cmd = 'X TAMAM -> Y'
                    else:
                        self.x_align_hold_start = None
                        self.x_align_hold_elapsed = 0.0

                        vx = self.clamp(
                            -(side_p_term + side_d_term),
                            self.VERTICAL_PREALIGN_MAX_SIDE_SPEED,
                        )
                        x_cmd = 'DUSEY ONCESI X PD'

                    move_cmd = 'DUSEY SEGMENT - X HIZALA'

                else:
                    # X ancak ciddi biçimde kaçarsa düşey hareketi durdur.
                    if abs(self.error_x) > self.X_ALIGN_EXIT_TOLERANCE:
                        self.control_phase = self.PHASE_VERTICAL_X
                        self.x_align_hold_start = None
                        self.x_align_hold_elapsed = 0.0
                        self.vertical_realign_count += 1
                        self.request_derivative_reset()

                        vx = 0.0
                        vz = 0.0
                        x_cmd = 'X COK KACTI - YENIDEN HIZALA'
                        y_cmd = 'Y DUR'
                        move_cmd = 'DUSEY SEGMENT - REALIGN'
                    else:
                        # Düşey hareket sürerken yalnız küçük X düzeltmesi.
                        if abs(self.error_x) > self.X_HOLD_DEADBAND:
                            x_hold_p_term = (
                                self.X_HOLD_KP * self.error_x
                            )
                            x_hold_d_term = (
                                self.X_HOLD_KD
                                * self.filtered_derivative_x
                            )
                            x_hold_d_term = self.clamp(
                                x_hold_d_term,
                                self.MAX_X_HOLD_D_TERM,
                            )
                            vx = self.clamp(
                                -(x_hold_p_term + x_hold_d_term),
                                self.MAX_X_HOLD_SPEED,
                            )
                            x_cmd = 'X KÜÇÜK HOLD'
                        else:
                            vx = 0.0
                            x_cmd = 'X HOLD'

                        if self.error_y > self.DEADBAND_Y:
                            vz = min(
                                self.MAX_VERTICAL_SPEED,
                                self.VERTICAL_KP * abs(self.error_y),
                            )
                            y_cmd = 'DUSEY ASAGI'
                        elif self.error_y < -self.DEADBAND_Y:
                            vz = -min(
                                self.MAX_VERTICAL_SPEED,
                                self.VERTICAL_KP * abs(self.error_y),
                            )
                            y_cmd = 'DUSEY YUKARI'
                        else:
                            vz = 0.0
                            y_cmd = 'Y TAMAM'

                        move_cmd = 'DUSEY SEGMENT - Y HAREKET'

            if self.hold_y is not None:
                forward_error = self.hold_y - self.current_y
                vy = self.clamp(
                    self.FORWARD_HOLD_KP * forward_error,
                    self.MAX_FORWARD_SPEED,
                )

        else:
            self.control_phase = self.PHASE_HOLD
            self.request_derivative_reset()
            self.x_align_hold_start = None
            self.x_align_hold_elapsed = 0.0

            if not self.offboard_active:
                move_cmd = 'OFFBOARD DEGIL'
            elif not vision_data_fresh:
                move_cmd = 'KAMERA VERISI YOK - HOLD'
            elif not self.window_found:
                move_cmd = 'PENCERE YOK - HOLD'
            elif self.mission_mode == self.MODE_DONE:
                move_cmd = 'PATH BITTI - HOLD'

        self.pump_active = self.offboard_active and self.pump_requested

        self.publish_states()
        self.publish_velocity_setpoint(vx, vy, vz, self.current_yaw)
        self.write_log(
            vx,
            vy,
            vz,
            vision_data_fresh,
            derivative_x_raw,
            side_p_term,
            side_d_term,
            segment_type,
            active_side_kp,
            active_side_kd,
            active_side_speed_limit,
            x_hold_p_term,
            x_hold_d_term,
            y_hold_p_term,
            path3_brake_zone,
        )

        if now - self.last_print_time > 1.0:
            print('offboard_active:', self.offboard_active)
            print('vision_data_fresh:', vision_data_fresh)
            print('window_found:', self.window_found)
            print('mission_mode:', self.mission_mode)
            print('path_index:', self.path_index)
            print('segment_type:', segment_type)
            print('control_phase:', self.control_phase)
            print(
                'x_align_hold:',
                f'{self.x_align_hold_elapsed:.2f}/'
                f'{self.X_ALIGN_HOLD_TIME:.2f}s',
                'realign_count:',
                self.vertical_realign_count,
            )
            print('error_x:', self.error_x, 'error_y:', self.error_y)
            print(
                'X PD:',
                'Kp=', active_side_kp,
                'Kd=', active_side_kd,
                'P=', round(side_p_term, 5),
                'D=', round(side_d_term, 5),
                'limit=', active_side_speed_limit,
                'zone=', path3_brake_zone,
            )
            print(
                'vehicle_velocity:',
                round(self.current_vx, 4),
                round(self.current_vy, 4),
                round(self.current_vz, 4),
            )
            print('command_velocity:', vx, vy, vz)
            print('pump_active:', self.pump_active)
            print('CMD:', x_cmd, '|', y_cmd, '|', move_cmd)
            print('--------------------------------')
            self.last_print_time = now

    def destroy_node(self):
        try:
            self.pump_active = False
            self.publish_states()
            self.log_file.flush()
            self.log_file.close()
            self.get_logger().info(
                f'CSV kayit kapatildi: {self.log_path}'
            )
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WindowOffboard()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
