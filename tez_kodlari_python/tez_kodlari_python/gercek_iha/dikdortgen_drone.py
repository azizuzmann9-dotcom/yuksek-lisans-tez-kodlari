#!/usr/bin/env python3

import csv
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from gpiozero import PWMOutputDevice
from pymavlink import mavutil


SERIAL_PORT = "/dev/ttyACM0"
CAMERA_DEVICE = "/dev/video0"

PUMP_GPIO = 17
PUMP_POWER = 1.0
PUMP_ENABLED = True

# Merkezleme toleransı 40 x 40 pikseldir.
# Bekleme süresi yoktur; pencere merkezi bu bölgeye girer girmez pompa açılır
# ve beş noktalı PATH görevi başlar.
CENTER_TOLERANCE_X = 40
CENTER_TOLERANCE_Y = 40

# PATH noktasına tek karede girildiğinde doğrudan sonraki noktaya geçilir.
# Hedefte bekleme süresi kullanılmaz.
PATH_TOLERANCE = 75

# Ana kontrol kazançları: yatay eksende PD, düşey eksende P.
SIDE_KP = 0.0014
SIDE_KD = 0.00055
VERTICAL_KP = 0.0016

# Düşey segment sırasında yatay konum tutan yardımcı PD kontrolü.
X_HOLD_KP = 0.0009
X_HOLD_KD = 0.00018
MAX_X_HOLD_SPEED = 0.022

# Yatay segment sırasında düşey konum tutan yardımcı P kontrolü.
Y_HOLD_KP = 0.0010
MAX_Y_HOLD_SPEED = 0.025

# Türev filtresi ve D terimi sınırları.
DERIVATIVE_ALPHA = 0.20
MAX_SIDE_D_TERM = 0.08
MAX_X_HOLD_D_TERM = 0.018

# Ana ve yardımcı kontrol ölü bölgeleri.
MAIN_DEADBAND_X = 8
MAIN_DEADBAND_Y = 8
HELPER_DEADBAND_X = 5
HELPER_DEADBAND_Y = 5

# Drone kodundaki ana hız sınırları korunmuştur.
MAX_VY = 0.18
MAX_VZ = 0.15
VX_FORWARD = 0.0

# Pencere algılanamadığında kör hareket yapılmaz. Drone hemen sıfır hız
# komutuyla bekler; pompa, PATH modu ve path_index korunur.

# Aynı pencereyi takip etme ayarları.
BOX_SMOOTH_ALPHA = 0.25
TRACK_MAX_DISTANCE_RATIO = 0.85
TRACK_MIN_AREA_RATIO = 0.35
TRACK_MAX_AREA_RATIO = 2.80
RELOCK_BEFORE_PUMP_FRAMES = 30

# Uçuş sırasında canlı OpenCV penceresi gösterilmez. Gerekli PATH verileri yalnızca CSV loguna yazılır.
# Masa testinde görüntüyü görmek istersen geçici olarak True yapılabilir.
SHOW_WINDOWS = False
DISPLAY_W = 480
DISPLAY_H = 320

# CSV kaydı kontrol döngüsünden bağımsız olarak 10 Hz ile sınırlandırılır.
# Görev durumu değiştiğinde veya hedefe ulaşıldığında önemli olay satırı ayrıca
# anında kaydedilir; böylece olaylar kaçırılmaz.
LOG_RATE_HZ = 10.0
LOG_INTERVAL_S = 1.0 / LOG_RATE_HZ
LOG_FLUSH_INTERVAL_S = 0.5

LOG_DIR = Path.home() / "drone_ws" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / ("flight_log_" + RUN_STAMP + ".csv")


# PATH yalnızca beş ana hedef noktasından oluşur; ara noktalar kullanılmaz.
# alt orta -> alt sol -> üst sol -> üst sağ -> alt sağ
PATH_NORMALIZED = [
    (0.50, 0.82),
    (0.22, 0.82),
    (0.22, 0.18),
    (0.78, 0.18),
    (0.78, 0.82),
]
PATH_SERIALIZED = ";".join(
    f"{u:.4f}:{v:.4f}" for u, v in PATH_NORMALIZED
)


def wake_camera():
    print("Kamera uyandiriliyor. Kamerada PC Camera sec.")

    # Bu bölüm özellikle iki kez bırakıldı.
    for i in range(2):
        print(str(i + 1) + ". kamera acma denemesi")

        try:
            subprocess.run(
                [
                    "ffplay",
                    "-autoexit",
                    "-t",
                    "2",
                    "-f",
                    "v4l2",
                    "-input_format",
                    "mjpeg",
                    CAMERA_DEVICE,
                ],
                timeout=6,
                check=False,
            )
        except Exception as exc:
            print("ffplay denemesi bitti:", exc)

        # Yalnızca uçuş döngüsü başlamadan önce.
        time.sleep(0.5)


def open_camera():
    print("OpenCV kamera aciliyor:", CAMERA_DEVICE)

    cap_local = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap_local.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start = time.time()
    while time.time() - start < 15:
        ret, frame = cap_local.read()
        if ret and frame is not None:
            print("Kamera hazir")
            return cap_local

        print("Kamera bekleniyor...")
        time.sleep(0.5)

    cap_local.release()
    raise RuntimeError("Kamera acilamadi. PC Camera secili mi?")


def limit(value, max_value):
    return max(-max_value, min(max_value, value))


def send_velocity(vx, vy, vz):
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0,
        0,
        0,
        float(vx),
        float(vy),
        float(vz),
        0,
        0,
        0,
        0,
        0,
    )


def stop_if_guided(flight_mode):
    if flight_mode == "GUIDED":
        send_velocity(0.0, 0.0, 0.0)


def reset_horizontal_derivative():
    """Hedef veya kontrol durumu değiştiğinde türev sıçramasını önler."""
    global last_error_x
    global filtered_derivative_x
    global derivative_reset_requested

    last_error_x = None
    filtered_derivative_x = 0.0
    derivative_reset_requested = True


def update_horizontal_derivative(error_x, dt):
    """Yatay hata türevini hesaplar ve birinci dereceden filtre uygular."""
    global last_error_x
    global filtered_derivative_x
    global derivative_reset_requested

    if derivative_reset_requested or last_error_x is None:
        raw_derivative = 0.0
        filtered_derivative_x = 0.0
        last_error_x = float(error_x)
        derivative_reset_requested = False
        return raw_derivative, filtered_derivative_x

    raw_derivative = (float(error_x) - last_error_x) / max(dt, 0.001)
    filtered_derivative_x = (
        DERIVATIVE_ALPHA * raw_derivative
        + (1.0 - DERIVATIVE_ALPHA) * filtered_derivative_x
    )
    last_error_x = float(error_x)
    return raw_derivative, filtered_derivative_x


def get_segment_type(vision_mode, path_index):
    if vision_mode == "CENTER":
        return "CENTER"
    if vision_mode != "PATH":
        return "HOLD"

    # Merkezden P1'e, P2'den P3'e ve P4'ten P5'e düşey hareket edilir.
    if path_index in (0, 2, 4):
        return "VERTICAL"
    return "HORIZONTAL"


def errors_to_velocity(error_x, error_y, vision_mode, path_index, dt):
    """Segment türüne göre ana ve yardımcı kontrol komutlarını üretir."""
    derivative_raw, derivative_filtered = update_horizontal_derivative(
        error_x,
        dt,
    )

    side_p_term = SIDE_KP * error_x
    side_d_term = limit(
        SIDE_KD * derivative_filtered,
        MAX_SIDE_D_TERM,
    )

    x_hold_p_term = 0.0
    x_hold_d_term = 0.0
    y_hold_p_term = 0.0

    segment_type = get_segment_type(vision_mode, path_index)
    vy = 0.0
    vz = 0.0

    if segment_type == "CENTER":
        # Merkezleme sırasında ana yatay PD ve ana düşey P birlikte çalışır.
        if abs(error_x) > MAIN_DEADBAND_X:
            vy = limit(side_p_term + side_d_term, MAX_VY)

        if abs(error_y) > MAIN_DEADBAND_Y:
            vz = limit(VERTICAL_KP * error_y, MAX_VZ)

    elif segment_type == "HORIZONTAL":
        # Ana hareket yatay PD ile yapılır.
        if abs(error_x) > MAIN_DEADBAND_X:
            vy = limit(side_p_term + side_d_term, MAX_VY)

        # Düşey sapmalar düşük hızlı yardımcı P kontrolüyle sınırlandırılır.
        if abs(error_y) > HELPER_DEADBAND_Y:
            y_hold_p_term = Y_HOLD_KP * error_y
            vz = limit(y_hold_p_term, MAX_Y_HOLD_SPEED)

    elif segment_type == "VERTICAL":
        # Ana hareket düşey P kontrolüyle yapılır.
        if abs(error_y) > MAIN_DEADBAND_Y:
            vz = limit(VERTICAL_KP * error_y, MAX_VZ)

        # Yatay sapmalar düşük hızlı yardımcı PD kontrolüyle sınırlandırılır.
        if abs(error_x) > HELPER_DEADBAND_X:
            x_hold_p_term = X_HOLD_KP * error_x
            x_hold_d_term = limit(
                X_HOLD_KD * derivative_filtered,
                MAX_X_HOLD_D_TERM,
            )
            vy = limit(
                x_hold_p_term + x_hold_d_term,
                MAX_X_HOLD_SPEED,
            )

    return (
        VX_FORWARD,
        vy,
        vz,
        segment_type,
        derivative_raw,
        derivative_filtered,
        side_p_term,
        side_d_term,
        x_hold_p_term,
        x_hold_d_term,
        y_hold_p_term,
    )


def request_message_interval(message_id, frequency_hz):
    """ArduPilot'tan log icin gerekli MAVLink mesajlarini düzenli ister."""
    if frequency_hz <= 0:
        return

    interval_us = int(1_000_000 / frequency_hz)
    try:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )
    except Exception as exc:
        print("MAVLink mesaj araligi ayarlanamadi:", message_id, exc)


def update_flight_mode():
    """Uçuş modunu ve Gazebo logundaki karşılık gelen telemetriyi günceller."""
    global last_flight_mode
    global current_x, current_y, current_z
    global current_vx, current_vy, current_vz
    global current_roll, current_pitch, current_yaw
    global local_position_received, attitude_received
    global last_local_position_monotonic, last_attitude_monotonic

    while True:
        msg = master.recv_match(blocking=False)
        if msg is None:
            break

        msg_type = msg.get_type()

        if msg_type == "HEARTBEAT":
            try:
                last_flight_mode = mavutil.mode_string_v10(msg)
            except Exception:
                pass

        elif msg_type == "LOCAL_POSITION_NED":
            current_x = float(msg.x)
            current_y = float(msg.y)
            current_z = float(msg.z)
            current_vx = float(msg.vx)
            current_vy = float(msg.vy)
            current_vz = float(msg.vz)
            local_position_received = True
            last_local_position_monotonic = time.monotonic()

        elif msg_type == "ATTITUDE":
            current_roll = float(msg.roll)
            current_pitch = float(msg.pitch)
            current_yaw = float(msg.yaw)
            attitude_received = True
            last_attitude_monotonic = time.monotonic()

    return last_flight_mode


def turn_pump_on():
    global pump_on

    if not PUMP_ENABLED or pump_on:
        return

    pump.value = PUMP_POWER
    pump_on = True
    print("Pompa acildi. GUIDED modunda acik kalacak.")


def pump_off():
    global pump_on
    pump.value = 0.0
    pump_on = False


def box_center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def smooth_box(previous_box, new_box, alpha=BOX_SMOOTH_ALPHA):
    if previous_box is None:
        return tuple(float(value) for value in new_box)

    return tuple(
        previous_box[i] * (1.0 - alpha) + float(new_box[i]) * alpha
        for i in range(4)
    )


def select_same_window(candidates, reference_box, lost_frames):
    """Önceki pencereye konum ve boyut olarak en yakın adayı seçer."""
    if not candidates:
        return None

    if reference_box is None:
        return max(candidates, key=lambda item: item[4])[:4]

    ref_x, ref_y, ref_w, ref_h = reference_box
    ref_cx, ref_cy = box_center(reference_box)
    ref_area = max(ref_w * ref_h, 1.0)
    ref_diagonal = max(math.hypot(ref_w, ref_h), 1.0)

    # Birkaç kare kayıpta yeniden yakalama alanı kontrollü biçimde genişler.
    distance_ratio_limit = min(
        TRACK_MAX_DISTANCE_RATIO + lost_frames * 0.035,
        1.60,
    )

    best_box = None
    best_score = float("inf")

    for x, y, w, h, area in candidates:
        candidate_cx = x + w / 2.0
        candidate_cy = y + h / 2.0
        center_distance = math.hypot(candidate_cx - ref_cx, candidate_cy - ref_cy)
        distance_ratio = center_distance / ref_diagonal

        candidate_area = max(float(w * h), 1.0)
        area_ratio = candidate_area / ref_area

        if distance_ratio > distance_ratio_limit:
            continue
        if not (TRACK_MIN_AREA_RATIO <= area_ratio <= TRACK_MAX_AREA_RATIO):
            continue

        # Merkez yakınlığı ana ölçüt, boyut değişimi yardımcı ölçüttür.
        score = distance_ratio + 0.35 * abs(math.log(area_ratio))
        if score < best_score:
            best_score = score
            best_box = (x, y, w, h)

    return best_box


def normalized_to_pixel(point_norm, box):
    u, v = point_norm
    x, y, w, h = box
    return int(round(x + u * w)), int(round(y + v * h))


def camera_aim_on_window(drone_x, drone_y, box):
    """Kamera merkezinin pencere üzerindeki normalize karşılığını hesaplar."""
    x, y, w, h = box
    if w <= 1 or h <= 1:
        return None

    u = (drone_x - x) / w
    v = (drone_y - y) / h
    return u, v



def overlay(frame, status, control_status, err_dist, avg_err, flight_mode,
            lost_frames, window_locked):
    cv2.putText(frame, status, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
    cv2.putText(
        frame,
        control_status,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255) if flight_mode == "GUIDED" else (0, 0, 255),
        2,
    )
    cv2.putText(frame, "PUMP: " + ("ON" if pump_on else "OFF"), (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (0, 255, 255) if pump_on else (0, 0, 255), 2)
    cv2.putText(
        frame,
        "ERR: " + str(round(err_dist, 1)) + " AVG: " + str(round(avg_err, 1)),
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 0),
        2,
    )
    cv2.putText(frame, "FMODE: " + str(flight_mode), (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
    cv2.putText(
        frame,
        "LOCK: " + ("YES" if window_locked else "NO") +
        " LOST: " + str(lost_frames),
        (20, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 0) if window_locked else (0, 0, 255),
        2,
    )


# ================= BASLANGIC =================

print("MAVLink baglantisi bekleniyor:", SERIAL_PORT)
master = mavutil.mavlink_connection(SERIAL_PORT)
master.wait_heartbeat()
last_flight_mode = master.flightmode

print("MAVLink baglandi")
print("System:", master.target_system, "Component:", master.target_component)
print("Ilk mod:", last_flight_mode)

master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_ALL,
    10,
    1,
)

# Gazebo logundaki konum, hız ve yaw alanlarını gerçek uçuşta da doldurmak
# için ilgili MAVLink mesajları ayrıca 10 Hz istenir.
request_message_interval(
    mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
    10,
)
request_message_interval(
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
    10,
)

pump = PWMOutputDevice(PUMP_GPIO, frequency=100)
pump.value = 0.0
pump_on = False

# Kamera iki kez uyandırılır; bu akış değiştirilmedi.
wake_camera()
cap = open_camera()

vision_mode = "CENTER"
path_index = 0
error_history = []

tracking_box = None
smoothed_box = None
window_locked = False
lost_frames = 0
last_window_seen_monotonic = None

# Yalnızca GUIDED moda yeniden giriş anını algılamak için kullanılır.
# Karmaşık PATH devam ettirme mantığı yoktur.
previous_guided_active = last_flight_mode == "GUIDED"

# Yatay PD ve yardımcı yatay PD için türev durumu.
last_error_x = None
filtered_derivative_x = 0.0
derivative_reset_requested = True
last_control_monotonic = time.monotonic()

# Gazebo kontrol logundaki araç durum alanlarının gerçek sistem karşılıkları.
# LOCAL_POSITION_NED: metre ve m/s, yerel NED koordinat sistemi.
# ATTITUDE: roll, pitch ve yaw radyan cinsindedir.
current_x = None
current_y = None
current_z = None
current_vx = None
current_vy = None
current_vz = None
current_roll = None
current_pitch = None
current_yaw = None
local_position_received = False
attitude_received = False
last_local_position_monotonic = None
last_attitude_monotonic = None

log_start_monotonic = time.monotonic()
last_print = 0.0
last_log_flush = 0.0
last_log_write_monotonic = 0.0
last_logged_event_state = None

log_file = open(LOG_FILE, "w", newline="")
log_writer = csv.writer(log_file)
log_header = [
    # Gazebo V11 kontrol loguyla aynı çekirdek kolonlar.
    "time",
    "offboard_active",
    "vision_data_fresh",
    "window_found",
    "mission_mode",
    "path_index",
    "lost_frames",
    "path_inside_tolerance",
    "path_hold_elapsed",
    "error_x",
    "error_y",
    "target_u",
    "target_v",
    "actual_u",
    "actual_v",
    "current_x",
    "current_y",
    "current_z",
    "vx_cmd",
    "vy_cmd",
    "vz_cmd",
    "yaw_cmd",
    "side_kp",
    "side_kd",
    "vertical_kp",
    "derivative_x_raw",
    "derivative_x_filtered",
    "side_p_term",
    "side_d_term",
    "pump_requested",
    "pump_active",
    "elapsed_time_s",
    "current_vx",
    "current_vy",
    "current_vz",
    "segment_type",
    "control_phase",
    "x_align_hold_elapsed",
    "vertical_realign_count",
    "active_side_kp",
    "active_side_kd",
    "active_side_speed_limit",
    "x_hold_p_term",
    "x_hold_d_term",
    "y_hold_p_term",
    "path2_brake_zone",

    # Gerçek drone/kamera koduna özgü ek kolonlar.
    "flight_mode",
    "guided_active",
    "vision_mode",
    "window_locked",
    "lost_duration",
    "tracking_source",
    "window_x",
    "window_y",
    "window_w",
    "window_h",
    "window_cx",
    "window_cy",
    "target_x",
    "target_y",
    "actual_x",
    "actual_y",
    "error_distance",
    "avg_error",
    "path_total_points",
    "planned_path_uv",
    "pump_on",
    "status",
    "control_status",
    "current_roll",
    "current_pitch",
    "current_yaw",
    "local_position_received",
    "attitude_received",
    "local_position_age_s",
    "attitude_age_s",
    "command_frame",
    "telemetry_frame",
    "frame_width",
    "frame_height",
    "target_reached_this_frame",
    "reached_path_index",
]
log_writer.writerow(log_header)
log_file.flush()

print("Log dosyasi:", LOG_FILE)
print("Gazebo V11 uyumlu kontrol, telemetri ve 5 noktalı PATH verileri CSV loguna kaydedilecek.")
print("GUIDED olana kadar komut yok. GUIDED moda her giriste merkezleme baştan başlar.")
print(
    "Kontrol: yatay PD Kp=", SIDE_KP,
    "Kd=", SIDE_KD,
    "düşey P Kp=", VERTICAL_KP,
)
print("Merkez toleransi: 40x40 px; merkezde ve hedefte bekleme yok.")
print("CSV log hizi:", LOG_RATE_HZ, "Hz; önemli durum değişimleri ayrıca anında kaydedilir.")

try:
    while True:
        flight_mode = update_flight_mode()
        guided_active = flight_mode == "GUIDED"
        loop_monotonic = time.monotonic()
        control_dt = max(
            0.001,
            min(0.20, loop_monotonic - last_control_monotonic),
        )
        last_control_monotonic = loop_monotonic

        # GUIDED modundan çıkıldığı anda pompa hemen kapatılır.
        # Görev durumu yeniden GUIDED girişinde baştan sıfırlanacaktır.
        if previous_guided_active and not guided_active:
            reset_horizontal_derivative()
            if pump_on:
                pump_off()
                print("GUIDED kapandi: pompa kapatildi.")

        # Manuel moddan GUIDED moda yeniden geçildiğinde görev basit ve
        # öngörülebilir şekilde CENTER aşamasından yeniden başlar.
        # Eski PATH noktasını devam ettirme veya özel resume mantığı yoktur.
        if guided_active and not previous_guided_active:
            vision_mode = "CENTER"
            path_index = 0
            tracking_box = None
            smoothed_box = None
            window_locked = False
            lost_frames = 0
            last_window_seen_monotonic = None
            error_history.clear()
            reset_horizontal_derivative()
            print("GUIDED aktif: merkezleme ve PATH baştan başlayacak.")

        previous_guided_active = guided_active

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame yok. Kontrol dongusu sonlandiriliyor.")
            stop_if_guided(flight_mode)
            break

        frame_h, frame_w, _ = frame.shape
        drone_x = frame_w // 2
        drone_y = frame_h // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 160])
        upper = np.array([180, 80, 255])
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:
                continue

            x, y, window_w, window_h = cv2.boundingRect(contour)
            ratio = window_w / float(window_h)
            if 0.6 < ratio < 1.8:
                candidates.append((x, y, window_w, window_h, area))

        selected_box = select_same_window(candidates, tracking_box, lost_frames)

        # Pompa açılmadan önce uzun süre hedef kaybolursa ilk yanlış kilidi bırakıp
        # tekrar aramaya izin verilir. Pompa açıldıktan sonra farklı hedefe geçilmez.
        if (
            selected_box is None
            and not pump_on
            and lost_frames >= RELOCK_BEFORE_PUMP_FRAMES
        ):
            tracking_box = None
            smoothed_box = None
            window_locked = False
            selected_box = select_same_window(candidates, None, 0)

        if selected_box is not None:
            tracking_box = selected_box
            smoothed_box = smooth_box(smoothed_box, selected_box)
            window_locked = True
            lost_frames = 0
            last_window_seen_monotonic = loop_monotonic
        else:
            lost_frames += 1

        window_found = selected_box is not None and smoothed_box is not None
        if last_window_seen_monotonic is None:
            lost_duration = 0.0
        else:
            lost_duration = max(0.0, loop_monotonic - last_window_seen_monotonic)
        tracking_source = "DETECTED" if window_found else "LOST"

        vx = 0.0
        vy = 0.0
        vz = 0.0
        error_x = 0
        error_y = 0
        err_dist = 0.0
        avg_err = 0.0
        status = "BEKLE"
        control_status = "MANUEL/BEKLE - KOMUT YOK"

        # Gazebo uyumlu görev/log alanları. Bu sürümde merkezde veya hedefte
        # bekleme bulunmadığından hold süreleri daima 0.0'dır.
        vision_data_fresh = True
        path_inside_tolerance = False
        path_hold_elapsed = 0.0
        target_reached_this_frame = False
        reached_path_index = None
        pump_requested = guided_active and vision_mode in ("PATH", "DONE")

        segment_type = "HOLD"
        derivative_x_raw = 0.0
        derivative_x_filtered = filtered_derivative_x
        side_p_term = 0.0
        side_d_term = 0.0
        x_hold_p_term = 0.0
        x_hold_d_term = 0.0
        y_hold_p_term = 0.0

        target_x = None
        target_y = None
        target_u = None
        target_v = None
        actual_x = None
        actual_y = None
        actual_u = None
        actual_v = None
        window_x = None
        window_y = None
        window_w = None
        window_h = None
        window_cx = None
        window_cy = None

        cv2.circle(frame, (drone_x, drone_y), 5, (255, 0, 255), -1)
        cv2.rectangle(
            frame,
            (drone_x - CENTER_TOLERANCE_X, drone_y - CENTER_TOLERANCE_Y),
            (drone_x + CENTER_TOLERANCE_X, drone_y + CENTER_TOLERANCE_Y),
            (255, 255, 0),
            1,
        )

        if window_found:
            box = smoothed_box
            window_x, window_y, window_w, window_h = box
            window_cx = window_x + window_w / 2.0
            window_cy = window_y + window_h / 2.0

            draw_box = tuple(int(round(value)) for value in box)
            bx, by, bw, bh = draw_box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.circle(
                frame,
                (int(round(window_cx)), int(round(window_cy))),
                5,
                (0, 0, 255),
                -1,
            )

            aim_norm = camera_aim_on_window(drone_x, drone_y, box)
            if aim_norm is not None:
                actual_u, actual_v = aim_norm
                actual_x, actual_y = normalized_to_pixel(aim_norm, box)

            if vision_mode == "CENTER":
                error_x = int(round(window_cx - drone_x))
                error_y = int(round(window_cy - drone_y))

                centered = (
                    abs(error_x) <= CENTER_TOLERANCE_X
                    and abs(error_y) <= CENTER_TOLERANCE_Y
                )

                if centered:
                    if guided_active:
                        turn_pump_on()
                        vision_mode = "PATH"
                        path_index = 0
                        reset_horizontal_derivative()

                        target_u, target_v = PATH_NORMALIZED[path_index]
                        target_x, target_y = normalized_to_pixel(
                            PATH_NORMALIZED[path_index], box
                        )
                        error_x = target_x - drone_x
                        error_y = target_y - drone_y
                        status = "POMPA ACILDI - PATH BASLADI"
                    else:
                        status = "MERKEZDE - GUIDED BEKLENIYOR"
                else:
                    status = "ORTALANIYOR"

            elif vision_mode == "PATH":
                path_index = min(path_index, len(PATH_NORMALIZED) - 1)
                target_u, target_v = PATH_NORMALIZED[path_index]
                target_x, target_y = normalized_to_pixel(
                    PATH_NORMALIZED[path_index], box
                )

                error_x = target_x - drone_x
                error_y = target_y - drone_y
                status = "PATH NOKTA " + str(path_index + 1) + "/" + str(
                    len(PATH_NORMALIZED)
                )

                path_inside_tolerance = (
                    abs(error_x) <= PATH_TOLERANCE
                    and abs(error_y) <= PATH_TOLERANCE
                )

                if path_inside_tolerance:
                    target_reached_this_frame = True
                    reached_path_index = path_index
                    if path_index < len(PATH_NORMALIZED) - 1:
                        path_index += 1
                        reset_horizontal_derivative()

                        # Bekleme yapılmadan yeni hedef aynı karede etkinleştirilir.
                        target_u, target_v = PATH_NORMALIZED[path_index]
                        target_x, target_y = normalized_to_pixel(
                            PATH_NORMALIZED[path_index], box
                        )
                        error_x = target_x - drone_x
                        error_y = target_y - drone_y
                        status = "PATH NOKTA " + str(path_index + 1) + "/" + str(
                            len(PATH_NORMALIZED)
                        )
                    else:
                        vision_mode = "DONE"
                        reset_horizontal_derivative()
                        status = "PATH BITTI - POMPA " + ("ACIK" if pump_on else "KAPALI")

            else:
                status = "GOREV TAMAM - POMPA " + ("ACIK" if pump_on else "KAPALI")
                if guided_active:
                    send_velocity(0.0, 0.0, 0.0)

            err_dist = math.sqrt(error_x**2 + error_y**2)
            error_history.append(err_dist)
            if len(error_history) > 500:
                error_history.pop(0)
            avg_err = sum(error_history) / len(error_history)

            if guided_active and vision_mode != "DONE":
                (
                    vx,
                    vy,
                    vz,
                    segment_type,
                    derivative_x_raw,
                    derivative_x_filtered,
                    side_p_term,
                    side_d_term,
                    x_hold_p_term,
                    x_hold_d_term,
                    y_hold_p_term,
                ) = errors_to_velocity(
                    error_x,
                    error_y,
                    vision_mode,
                    path_index,
                    control_dt,
                )
                send_velocity(vx, vy, vz)
                control_status = (
                    "GUIDED AKTIF - " + segment_type + " KONTROL"
                )
            elif guided_active and vision_mode == "DONE":
                send_velocity(0.0, 0.0, 0.0)
                control_status = "GOREV TAMAM - HOLD"
            else:
                control_status = "MANUEL/BEKLE - KOMUT YOK"

        else:

            # Pencere algılanamadığında son hareket komutu sürdürülmez.
            # Drone hemen durur; pompa, vision_mode ve path_index korunur.
            # Aynı pencere tekrar algılanınca aynı PATH noktasından devam edilir.
            vx = 0.0
            vy = 0.0
            vz = 0.0
            reset_horizontal_derivative()
            derivative_x_filtered = filtered_derivative_x
            if guided_active:
                send_velocity(0.0, 0.0, 0.0)

            tracking_source = "HOLD"
            status = "PENCERE KAYIP - HOLD"
            if pump_on:
                control_status = "PENCERE YOK - HOLD - POMPA ACIK"
            else:
                control_status = "PENCERE YOK - HOLD"

            cv2.putText(
                frame,
                "AYNI PENCERE BEKLENIYOR",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 255),
                2,
            )

        if SHOW_WINDOWS:
            overlay(
                frame,
                status,
                control_status,
                err_dist,
                avg_err,
                flight_mode,
                lost_frames,
                window_locked,
            )

        # Gazebo V11 log kolonlarının bu gerçek-drone sürümündeki karşılıkları.
        mission_mode = {"CENTER": 0, "PATH": 1, "DONE": 2}.get(
            vision_mode,
            -1,
        )
        pump_requested = guided_active and vision_mode in ("PATH", "DONE")

        if not guided_active or not window_found or vision_mode == "DONE":
            control_phase = "HOLD"
        elif segment_type == "CENTER":
            control_phase = "CENTER"
        elif segment_type == "HORIZONTAL":
            control_phase = "HORIZONTAL_MOVE"
        elif segment_type == "VERTICAL":
            control_phase = "VERTICAL_MOVE"
        else:
            control_phase = "HOLD"

        # Bu sürümde dikey segment öncesi hizalama beklemesi ve yeniden
        # hizalama fazı yoktur; ilgili Gazebo kolonları sıfır olarak tutulur.
        x_align_hold_elapsed = 0.0
        vertical_realign_count = 0

        if segment_type in ("CENTER", "HORIZONTAL"):
            active_side_kp = SIDE_KP
            active_side_kd = SIDE_KD
            active_side_speed_limit = MAX_VY
        elif segment_type == "VERTICAL":
            active_side_kp = X_HOLD_KP
            active_side_kd = X_HOLD_KD
            active_side_speed_limit = MAX_X_HOLD_SPEED
        else:
            active_side_kp = 0.0
            active_side_kd = 0.0
            active_side_speed_limit = 0.0

        path2_brake_zone = "NOT_USED"
        elapsed_time_s = loop_monotonic - log_start_monotonic

        local_position_age_s = (
            None
            if last_local_position_monotonic is None
            else max(0.0, loop_monotonic - last_local_position_monotonic)
        )
        attitude_age_s = (
            None
            if last_attitude_monotonic is None
            else max(0.0, loop_monotonic - last_attitude_monotonic)
        )

        now = time.time()
        log_row = [
            # Gazebo V11 ile aynı çekirdek veri sırası.
            now,
            guided_active,
            vision_data_fresh,
            window_found,
            mission_mode,
            path_index,
            lost_frames,
            path_inside_tolerance,
            path_hold_elapsed,
            error_x,
            error_y,
            target_u,
            target_v,
            actual_u,
            actual_v,
            current_x,
            current_y,
            current_z,
            vx,
            vy,
            vz,
            None,  # ArduPilot BODY_NED komutunda bu kod ayrı yaw setpoint göndermiyor.
            SIDE_KP,
            SIDE_KD,
            VERTICAL_KP,
            derivative_x_raw,
            derivative_x_filtered,
            side_p_term,
            side_d_term,
            pump_requested,
            pump_on,
            elapsed_time_s,
            current_vx,
            current_vy,
            current_vz,
            segment_type,
            control_phase,
            x_align_hold_elapsed,
            vertical_realign_count,
            active_side_kp,
            active_side_kd,
            active_side_speed_limit,
            x_hold_p_term,
            x_hold_d_term,
            y_hold_p_term,
            path2_brake_zone,

            # Gerçek drone/kamera ek verileri.
            flight_mode,
            guided_active,
            vision_mode,
            window_locked,
            round(lost_duration, 4),
            tracking_source,
            window_x,
            window_y,
            window_w,
            window_h,
            window_cx,
            window_cy,
            target_x,
            target_y,
            actual_x,
            actual_y,
            err_dist,
            avg_err,
            len(PATH_NORMALIZED),
            PATH_SERIALIZED,
            pump_on,
            status,
            control_status,
            current_roll,
            current_pitch,
            current_yaw,
            local_position_received,
            attitude_received,
            local_position_age_s,
            attitude_age_s,
            "BODY_NED",
            "LOCAL_NED",
            frame_w,
            frame_h,
            target_reached_this_frame,
            reached_path_index,
        ]

        if len(log_row) != len(log_header):
            raise RuntimeError(
                "CSV kolon sayisi uyusmuyor: "
                + str(len(log_row))
                + " != "
                + str(len(log_header))
            )

        # Kamera ve kontrol döngüsü tam hızında çalışır; CSV satırı normalde
        # en fazla 10 Hz ile yazılır. Mod, görev aşaması, pencere durumu, pompa
        # durumu veya hedef indeksi değişirse önemli olay kaybolmasın diye satır
        # zaman aralığı beklenmeden ayrıca kaydedilir.
        event_state = (
            guided_active,
            vision_mode,
            window_found,
            pump_on,
            path_index,
            target_reached_this_frame,
        )
        important_event = (
            target_reached_this_frame
            or event_state != last_logged_event_state
        )
        log_interval_elapsed = (
            loop_monotonic - last_log_write_monotonic >= LOG_INTERVAL_S
        )

        if log_interval_elapsed or important_event:
            log_writer.writerow(log_row)
            last_log_write_monotonic = loop_monotonic
            last_logged_event_state = event_state

            if now - last_log_flush >= LOG_FLUSH_INTERVAL_S:
                log_file.flush()
                last_log_flush = now

        if now - last_print > 0.5:
            print(
                "flight_mode:", flight_mode,
                "vision_mode:", vision_mode,
                "found:", window_found,
                "locked:", window_locked,
                "lost:", lost_frames,
                "lost_s:", round(lost_duration, 3),
                "source:", tracking_source,
                "target:", (target_x, target_y),
                "actual_uv:",
                None if actual_u is None else (round(actual_u, 3), round(actual_v, 3)),
                "error_x:", error_x,
                "error_y:", error_y,
                "vx:", round(vx, 3),
                "vy:", round(vy, 3),
                "vz:", round(vz, 3),
                "path:", path_index,
                "segment:", segment_type,
                "d_raw:", round(derivative_x_raw, 3),
                "d_filt:", round(derivative_x_filtered, 3),
                "vehicle_xyz:", (current_x, current_y, current_z),
                "vehicle_v:", (current_vx, current_vy, current_vz),
                "yaw:", current_yaw,
                "pump:", pump_on,
                status,
            )
            last_print = now

        if SHOW_WINDOWS:
            small = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            mask_small = cv2.resize(mask, (DISPLAY_W, DISPLAY_H))

            cv2.imshow("camera - debug", small)
            cv2.imshow("mask", mask_small)

            if cv2.waitKey(1) == ord("q"):
                break

finally:
    print("Cikis yapiliyor...")

    # Program sonlanırken GPIO çıkışı ayrıca güvenli duruma alınır.
    pump_off()
    stop_if_guided(update_flight_mode())

    log_file.flush()
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print("Sistem kapandi")
    print("Log dosyasi:", LOG_FILE)
