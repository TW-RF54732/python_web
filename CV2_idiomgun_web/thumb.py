import cv2
import mediapipe as mp
import time

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
# 判斷大拇指朝上
# 改良版：同時允許「只看到部分手」時仍嘗試追蹤拇指
# ========================
def is_thumb_up(lm):
    """
    判斷拇指朝上：
    - 拇指尖(4) 比 拇指第一關節(3) 高（y 值更小）
    - 額外確認拇指尖比食指根(5)高，避免誤判
    """
    thumb_up = lm[4].y < lm[3].y
    above_index_base = lm[4].y < lm[5].y  # 拇指尖比食指根高
    return thumb_up and above_index_base

def get_thumb_tip(lm, w, h):
    """回傳拇指尖的像素座標"""
    return int(lm[4].x * w), int(lm[4].y * h)

# ========================
# 攝影機
# ========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ========================
# 參數設定
# ========================
cross_x, cross_y = W // 2, H // 2

SMOOTH = 0.7          # 平滑係數（越大越慣性）
HOVER_TIME = 1.0      # 停留觸發秒數
LOST_TIMEOUT = 0.8    # 手消失補償秒數（加長）

# ========================
# 📦 方框設定（解決下方太低問題）
# ========================
# 核心策略：方框統一控制在「安全區域」內
# 上方留 margin_top，下方不超過 H * 0.65（確保手腕不會出框）
SAFE_TOP    = 80              # 上方安全邊距
SAFE_BOTTOM = int(H * 0.65)  # 下方安全上限（原本到 H-250 約 0.65H）
BOX_SIZE    = 100             # 方框大小
SIDE_MARGIN = 50              # 左右邊距

boxes = [
    # 左上
    (SIDE_MARGIN,       SAFE_TOP,                SIDE_MARGIN + BOX_SIZE,  SAFE_TOP + BOX_SIZE),
    # 右上
    (W - SIDE_MARGIN - BOX_SIZE, SAFE_TOP,       W - SIDE_MARGIN,         SAFE_TOP + BOX_SIZE),
    # 左下（往上移，不超過 SAFE_BOTTOM）
    (SIDE_MARGIN,       SAFE_BOTTOM - BOX_SIZE,  SIDE_MARGIN + BOX_SIZE,  SAFE_BOTTOM),
    # 右下（往上移，不超過 SAFE_BOTTOM）
    (W - SIDE_MARGIN - BOX_SIZE, SAFE_BOTTOM - BOX_SIZE, W - SIDE_MARGIN, SAFE_BOTTOM),
]

# ========================
# 狀態變數
# ========================
hover_start_time = None
current_target   = None
text             = ""
text_timer       = 0
last_seen_time   = time.time()
last_valid_x     = cross_x   # 最後有效拇指 x（邊界鎖定用）
last_valid_y     = cross_y   # 最後有效拇指 y

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
    # 畫安全區示意線（可選，debug 用）
    # ──────────────────────────────
    cv2.line(frame, (0, SAFE_BOTTOM), (w, SAFE_BOTTOM), (50, 50, 200), 1)
    cv2.putText(frame, "Safe Zone", (5, SAFE_BOTTOM - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 200), 1)

    # ──────────────────────────────
    # 畫方框（含停留進度條）
    # ──────────────────────────────
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (255, 0, 0)

        # 若正在停留此方框，顯示進度條
        if current_target == i and hover_start_time is not None:
            elapsed  = time.time() - hover_start_time
            progress = min(elapsed / HOVER_TIME, 1.0)
            bar_w    = int((x2 - x1) * progress)
            cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 10), (0, 255, 0), -1)
            color = (0, 200, 255)  # 高亮框

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ──────────────────────────────
    # 手勢偵測
    # ──────────────────────────────
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    result   = landmarker.detect_for_video(mp_image, timestamp)

    thumb_mode = False
    hand_detected = False

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        last_seen_time = time.time()
        hand_detected  = True

        if is_thumb_up(lm):
            thumb_mode = True
            new_x, new_y = get_thumb_tip(lm, w, h)

            # ✅ 邊界鎖定：拇指超出畫面時保留上次位置
            # （MediaPipe 偶爾回傳 <0 或 >1 的值）
            new_x = max(0, min(w, new_x))
            new_y = max(0, min(h, new_y))

            # 平滑移動
            cross_x = int(SMOOTH * cross_x + (1 - SMOOTH) * new_x)
            cross_y = int(SMOOTH * cross_y + (1 - SMOOTH) * new_y)

            # 更新最後有效位置
            last_valid_x = cross_x
            last_valid_y = cross_y

    # ✅ 手消失補償：短暫失偵維持游標在最後位置
    if not hand_detected:
        lost_duration = time.time() - last_seen_time
        if lost_duration < LOST_TIMEOUT:
            thumb_mode = True
            # 游標保持在 last_valid 位置，不移動
            cross_x = last_valid_x
            cross_y = last_valid_y
        else:
            thumb_mode = False

    # ──────────────────────────────
    # 停留判定
    # ──────────────────────────────
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
            else:
                if time.time() - hover_start_time > HOVER_TIME:
                    text       = f"Box {hit_box + 1} ✓"
                    text_timer = time.time()
                    hover_start_time = time.time()  # 重設，避免連續觸發
        else:
            current_target   = None
            hover_start_time = None
    else:
        current_target   = None
        hover_start_time = None

    # ──────────────────────────────
    # 畫準心游標
    # ──────────────────────────────
    if thumb_mode:
        cx, cy = cross_x, cross_y
        # 外圓
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 2)
        # 中心點
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
        # 十字線
        cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)

    # ──────────────────────────────
    # 顯示觸發文字
    # ──────────────────────────────
    if text:
        cv2.putText(frame, text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 255), 3)
        if time.time() - text_timer > 1.5:
            text = ""

    # ──────────────────────────────
    # Debug 資訊（左下角）
    # ──────────────────────────────
    debug_color = (0, 255, 0) if thumb_mode else (0, 0, 255)
    status_text = "THUMB UP" if thumb_mode else "NO HAND"
    cv2.putText(frame, status_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)

    cv2.imshow("Hover Aim System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()