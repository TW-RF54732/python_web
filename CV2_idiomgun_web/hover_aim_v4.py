import cv2
import mediapipe as mp
import time
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ========================
# 初始化 MediaPipe
# ========================
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=r"C:\python_web\CV2_idiomgun_web\hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# ========================
# 大拇指判斷（連續幀穩定版）
# ========================
CONFIRM_FRAMES     = 3
thumb_up_counter   = 0
thumb_down_counter = 0
thumb_state        = False

def raw_is_thumb_up(lm):
    return lm[4].y < lm[3].y and lm[4].y < lm[5].y

def update_thumb_state(detected: bool) -> bool:
    global thumb_up_counter, thumb_down_counter, thumb_state
    if detected:
        thumb_up_counter   += 1
        thumb_down_counter  = 0
        if thumb_up_counter >= CONFIRM_FRAMES:
            thumb_state = True
    else:
        thumb_down_counter += 1
        thumb_up_counter    = 0
        if thumb_down_counter >= CONFIRM_FRAMES:
            thumb_state = False
    return thumb_state

# ========================
# 攝影機
# ========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ========================
# 參數
# ========================
cross_x, cross_y = W // 2, H // 2
SMOOTH       = 0.65
HOVER_TIME   = 1.0
LOST_TIMEOUT = 0.8

# ========================
# 方塊設定（安全區內）
# ========================
SAFE_TOP    = 80
SAFE_BOTTOM = int(H * 0.62)
BOX_SIZE    = 100
SIDE_MARGIN = 50

boxes = [
    (SIDE_MARGIN,                SAFE_TOP,                SIDE_MARGIN + BOX_SIZE,       SAFE_TOP + BOX_SIZE),
    (W - SIDE_MARGIN - BOX_SIZE, SAFE_TOP,                W - SIDE_MARGIN,              SAFE_TOP + BOX_SIZE),
    (SIDE_MARGIN,                SAFE_BOTTOM - BOX_SIZE,  SIDE_MARGIN + BOX_SIZE,       SAFE_BOTTOM),
    (W - SIDE_MARGIN - BOX_SIZE, SAFE_BOTTOM - BOX_SIZE,  W - SIDE_MARGIN,              SAFE_BOTTOM),
]
box_labels = ["A", "B", "C", "D"]

# ========================
# 狀態
# ========================
hover_start_time = None
current_target   = None
last_seen_time   = time.time()
last_valid_x     = cross_x
last_valid_y     = cross_y

# 橫幅狀態
BANNER_HEIGHT    = 90         # 橫幅高度（px）
BANNER_DURATION  = 2.2        # 顯示秒數
SLIDE_DURATION   = 0.18       # 滑入/滑出動畫秒數
banner_text      = ""
banner_start     = None       # 觸發時間

# ========================
# 繪製：簡約邊框方塊
# ========================
def draw_box(frame, x1, y1, x2, y2, label, progress=0.0, active=False):
    bw = x2 - x1

    border_color = (255, 255, 255) if active else (160, 160, 160)
    text_color   = (255, 255, 255) if active else (160, 160, 160)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    # 停留進度：底部細線
    if progress > 0:
        bar_x2 = x1 + int(bw * progress)
        cv2.line(frame, (x1, y2), (bar_x2, y2), (0, 220, 100), 2, cv2.LINE_AA)

    # 置中字母
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 1.1
    fw   = 2
    (tw, th), _ = cv2.getTextSize(label, font, fs, fw)
    tx = x1 + (bw - tw) // 2
    ty = y1 + ((y2 - y1) + th) // 2
    cv2.putText(frame, label, (tx, ty), font, fs, text_color, fw, cv2.LINE_AA)

# ========================
# 繪製：準心
# ========================
def draw_crosshair(frame, cx, cy, locked=False):
    color  = (0, 210, 80) if not locked else (0, 150, 255)
    gap    = 5
    arm    = 14
    cv2.circle(frame, (cx, cy), 15, color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - arm - gap, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + gap, cy),       (cx + arm + gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm - gap), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + gap),       (cx, cy + arm + gap), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

# ========================
# 繪製：滑入橫幅
# ========================
def draw_banner(frame, text, banner_start):
    h, w = frame.shape[:2]
    now     = time.time()
    elapsed = now - banner_start
    remain  = BANNER_DURATION - elapsed

    # 計算 Y 偏移（滑入 / 靜止 / 滑出）
    if elapsed < SLIDE_DURATION:
        # 滑入
        t      = elapsed / SLIDE_DURATION
        t      = t * t * (3 - 2 * t)          # smoothstep
        offset = int((1 - t) * BANNER_HEIGHT)
    elif remain < SLIDE_DURATION:
        # 滑出
        t      = (SLIDE_DURATION - remain) / SLIDE_DURATION
        t      = max(0.0, min(1.0, t))
        t      = t * t * (3 - 2 * t)
        offset = int(t * BANNER_HEIGHT)
    else:
        offset = 0

    # 橫幅頂端 Y
    banner_y = h - BANNER_HEIGHT + offset

    # 半透明底色
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, banner_y), (w, h + offset), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # 頂部細分隔線
    cv2.line(frame, (0, banner_y), (w, banner_y), (80, 80, 80), 1)

    # 文字置中
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs   = 1.6
    fw   = 3
    (tw, th), _ = cv2.getTextSize(text, font, fs, fw)
    tx = (w - tw) // 2
    ty = banner_y + (BANNER_HEIGHT + th) // 2 - offset // 2

    # 文字陰影
    cv2.putText(frame, text, (tx + 2, ty + 2), font, fs, (0, 0, 0),   fw + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty),          font, fs, (240, 240, 240), fw, cv2.LINE_AA)

# ========================
# 主迴圈
# ========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ──────────────────────────────
    # 手勢偵測
    # ──────────────────────────────
    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    result    = landmarker.detect_for_video(mp_image, timestamp)

    hand_detected = bool(result.hand_landmarks)
    cursor_locked = False

    if hand_detected:
        lm = result.hand_landmarks[0]
        last_seen_time = time.time()
        stable_thumb   = update_thumb_state(raw_is_thumb_up(lm))

        if stable_thumb:
            new_x = max(0, min(w, int(lm[4].x * w)))
            new_y = max(0, min(h, int(lm[4].y * h)))
            cross_x = int(SMOOTH * cross_x + (1 - SMOOTH) * new_x)
            cross_y = int(SMOOTH * cross_y + (1 - SMOOTH) * new_y)
            last_valid_x, last_valid_y = cross_x, cross_y
    else:
        stable_thumb = update_thumb_state(False)

    # 手消失補償
    if not hand_detected and time.time() - last_seen_time < LOST_TIMEOUT:
        stable_thumb  = thumb_state
        cross_x, cross_y = last_valid_x, last_valid_y
        cursor_locked = True

    thumb_mode = stable_thumb

    # ──────────────────────────────
    # 橫幅顯示中：暫停互動邏輯，但繼續偵測手勢
    # ──────────────────────────────
    banner_active = (
        banner_text and
        banner_start is not None and
        time.time() - banner_start < BANNER_DURATION
    )

    if not banner_active:
        # 停留判定
        hit_box = None
        if thumb_mode:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if x1 < cross_x < x2 and y1 < cross_y < y2:
                    hit_box = i
                    break

            if hit_box is not None:
                if current_target != hit_box:
                    current_target   = hit_box
                    hover_start_time = time.time()
                elif time.time() - hover_start_time > HOVER_TIME:
                    banner_text  = f"選擇：{box_labels[hit_box]}"
                    banner_start = time.time()
                    current_target   = None
                    hover_start_time = None
            else:
                current_target   = None
                hover_start_time = None
        else:
            current_target   = None
            hover_start_time = None

    # ──────────────────────────────
    # 繪製方塊
    # ──────────────────────────────
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        progress = 0.0
        active   = (current_target == i and not banner_active)
        if active and hover_start_time is not None:
            progress = min((time.time() - hover_start_time) / HOVER_TIME, 1.0)
        draw_box(frame, x1, y1, x2, y2, box_labels[i], progress, active)

    # ──────────────────────────────
    # 繪製準心
    # ──────────────────────────────
    if thumb_mode:
        draw_crosshair(frame, cross_x, cross_y, locked=cursor_locked)

    # ──────────────────────────────
    # 繪製橫幅（最後繪製，確保在最上層）
    # ──────────────────────────────
    if banner_active:
        draw_banner(frame, banner_text, banner_start)

    cv2.imshow("Hover Aim System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
