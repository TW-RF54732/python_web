import cv2
import mediapipe as mp
import time
import random
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ────────────────────────────────────────────
# 成語資料（內嵌，不需外部檔案）
# ────────────────────────────────────────────
idioms = {
    "怒髮衝冠": {0: {"easy": ["努"], "medium": ["弩"], "hard": ["恕"]}, 1: {"medium": ["髮"]}, 2: {"easy": ["沖"], "medium": ["充"]}},
    "四面楚歌": {0: {"easy": ["西"], "medium": ["死"], "hard": ["匹"]}, 2: {"easy": ["處"], "hard": ["濋"]}, 3: {"easy": ["哥"]}},
    "畫蛇添足": {0: {"easy": ["劃"], "medium": ["晝"], "hard": ["書"]}, 1: {"easy": ["它"]}, 2: {"easy": ["填"], "medium": ["婖"]}},
    "守株待兔": {0: {"easy": ["首"], "medium": ["宋"], "hard": ["宇"]}, 1: {"easy": ["珠"], "medium": ["殊"], "hard": ["朱"]}, 2: {"hard": ["侍"]}, 3: {"medium": ["免"]}},
    "自相矛盾": {0: {"easy": ["白"], "medium": ["目"], "hard": ["由"]}, 1: {"medium": ["湘"], "hard": ["箱"]}, 2: {"easy": ["予"], "medium": ["茅"]}, 3: {"easy": ["頓"], "medium": ["鈍"], "hard": ["遁"]}},
    "一鼓作氣": {1: {"easy": ["股"]}, 2: {"easy": ["做"]}, 3: {"easy": ["汽"], "medium": ["棄"]}},
    "亡羊補牢": {0: {"easy": ["忘"], "medium": ["芒"]}, 2: {"easy": ["捕"]}},
    "破釜沉舟": {0: {"easy": ["坡"], "medium": ["波"], "hard": ["披"]}, 1: {"easy": ["斧"]}, 2: {"medium": ["沈"]}},
    "對症下藥": {0: {"easy": ["隊"], "hard": ["對"]}, 1: {"easy": ["証"], "medium": ["政"]}},
    "刻苦耐勞": {0: {"easy": ["克"], "medium": ["剋"]}, 1: {"easy": ["奈"]}, 2: {"easy": ["牢"]}},
    "一目了然": {1: {"easy": ["自"], "medium": ["且"], "hard": ["日"]}, 2: {"easy": ["瞭"]}, 3: {"easy": ["燃"], "hard": ["染"]}},
    "一箭雙鵰": {1: {"easy": ["剪"], "medium": ["劍"]}, 2: {"hard": ["爽"]}, 3: {"easy": ["雕"], "medium": ["凋"]}},
    "三思而行": {1: {"easy": ["恩"]}},
    "五花八門": {1: {"medium": ["化"]}, 2: {"easy": ["人"], "medium": ["入"]}, 3: {"hard": ["們"]}},
    "虎視眈眈": {1: {"hard": ["示"]}, 2: {"easy": ["耽"]}, 3: {"easy": ["耽"]}},
    "魚目混珠": {0: {"easy": ["漁"]}, 1: {"easy": ["自"], "medium": ["且"], "hard": ["日"]}, 2: {"hard": ["渾"]}, 3: {"easy": ["株"], "hard": ["誅"]}},
    "青出於藍": {0: {"easy": ["清"]}, 2: {"hard": ["于"]}, 3: {"medium": ["籃"]}},
    "百發百中": {0: {"easy": ["白"]}, 1: {"medium": ["廢"]}, 2: {"easy": ["白"]}, 3: {"easy": ["忠"], "medium": ["仲"]}},
    "當機立斷": {0: {"easy": ["檔"], "medium": ["擋"]}, 1: {"easy": ["基"], "hard": ["奇"]}, 3: {"easy": ["段"], "hard": ["鍛"]}},
    "半途而廢": {0: {"easy": ["牛"], "hard": ["丰"]}, 1: {"easy": ["圖"], "medium": ["徒"], "hard": ["徙"]}, 3: {"easy": ["費"]}},
    "水落石出": {1: {"easy": ["洛"], "medium": ["絡"], "hard": ["駱"]}, 2: {"easy": ["右"]}},
    "手忙腳亂": {1: {"easy": ["茫"], "medium": ["盲"], "hard": ["芒"]}, 3: {"medium": ["辭"]}},
    "心驚膽跳": {0: {"easy": ["必"]}, 2: {"easy": ["擔"], "hard": ["憚"]}, 3: {"medium": ["挑"], "hard": ["眺"]}},
    "名列前茅": {0: {"easy": ["各"], "hard": ["洛"]}, 1: {"medium": ["烈"], "hard": ["裂"]}, 2: {"medium": ["煎"], "hard": ["箭"]}, 3: {"medium": ["矛"]}},
    "全神貫注": {0: {"easy": ["金"]}, 1: {"easy": ["伸"], "hard": ["紳"]}, 2: {"easy": ["串"], "medium": ["慣"], "hard": ["摜"]}, 3: {"easy": ["住"], "medium": ["註"], "hard": ["柱"]}},
    "按部就班": {0: {"easy": ["案"], "medium": ["暗"], "hard": ["黯"]}, 1: {"easy": ["陪"], "medium": ["步"], "hard": ["倍"]}, 3: {"easy": ["般"], "medium": ["斑"], "hard": ["搬"]}},
    "前功盡棄": {0: {"easy": ["剪"], "hard": ["煎"]}, 1: {"easy": ["工"], "medium": ["攻"], "hard": ["公"]}, 2: {"easy": ["儘"], "medium": ["進"], "hard": ["禁"]}, 3: {"easy": ["氣"]}},
    "持之以恆": {0: {"hard": ["待"]}, 1: {"easy": ["支"]}, 2: {"easy": ["已"], "medium": ["己"], "hard": ["乙"]}, 3: {"easy": ["衡"]}},
    "如魚得水": {0: {"hard": ["奴"]}, 1: {"easy": ["漁"]}, 2: {"easy": ["德"]}},
}
options_pool = list("家國山水風雲花草人口手足心肝腦頭耳目鼻天地日月星空海河湖江田土木火金石")

# ────────────────────────────────────────────
# MediaPipe 初始化
# ────────────────────────────────────────────
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

# ────────────────────────────────────────────
# 手勢偵測函式
# ────────────────────────────────────────────
def is_thumb_up(lm):
    """大拇指朝上（其他手指蜷縮）"""
    thumb_up = lm[4].y < lm[3].y and lm[4].y < lm[5].y
    # 其他四指蜷縮：指尖 y > 對應 MCP y
    fingers_down = all([
        lm[8].y  > lm[6].y,   # 食指
        lm[12].y > lm[10].y,  # 中指
        lm[16].y > lm[14].y,  # 無名指
        lm[20].y > lm[18].y,  # 小指
    ])
    return thumb_up and fingers_down

def is_fist(lm):
    """握拳：四指 + 拇指都蜷縮"""
    all_down = all([
        lm[8].y  > lm[6].y,
        lm[12].y > lm[10].y,
        lm[16].y > lm[14].y,
        lm[20].y > lm[18].y,
        lm[4].y  > lm[3].y,   # 拇指也往下
    ])
    return all_down

# ────────────────────────────────────────────
# 題目生成
# ────────────────────────────────────────────
def generate_fill_blank():
    """填空題：成語中某字被挖空，找出正確字"""
    idiom = random.choice(list(idioms.keys()))
    pos   = random.choice(list(idioms[idiom].keys()))
    correct = idiom[pos]
    question = idiom[:pos] + "＿" + idiom[pos+1:]

    wrong_opts = []
    for lvl_list in idioms[idiom][pos].values():
        wrong_opts.extend(lvl_list)
    wrong_opts = list(set(wrong_opts))
    while len(wrong_opts) < 3:
        c = random.choice(options_pool)
        if c != correct and c not in wrong_opts:
            wrong_opts.append(c)
    wrong_opts = random.sample(wrong_opts, 3)

    choices = wrong_opts + [correct]
    random.shuffle(choices)

    return {
        "mode": "fill",
        "idiom": idiom,
        "question": question,
        "choices": choices,
        "correct_char": correct,
        "correct_index": choices.index(correct),
    }

def generate_find_wrong():
    """找錯字：四字成語中某字被換成錯字，找出哪個字錯了"""
    idiom = random.choice(list(idioms.keys()))
    pos   = random.choice(list(idioms[idiom].keys()))

    # 選一個替換字
    all_wrongs = []
    for lvl_list in idioms[idiom][pos].values():
        all_wrongs.extend(lvl_list)
    wrong_char = random.choice(all_wrongs)

    # 建立顯示成語（含錯字）
    display = list(idiom)
    display[pos] = wrong_char
    display_idiom = "".join(display)

    return {
        "mode": "wrong",
        "idiom": idiom,
        "display_idiom": display_idiom,
        "wrong_pos": pos,
        "wrong_char": wrong_char,
    }

def new_question():
    if random.random() < 0.5:
        return generate_fill_blank()
    else:
        return generate_find_wrong()

# ────────────────────────────────────────────
# 繪製工具：支援中文的 putText
# ────────────────────────────────────────────
try:
    from PIL import ImageFont, ImageDraw, Image
    FONT_PATH  = "C:/Windows/Fonts/msjh.ttc"   # 微軟正黑體
    FONT_LARGE = ImageFont.truetype(FONT_PATH, 72)
    FONT_MED   = ImageFont.truetype(FONT_PATH, 52)
    FONT_SMALL = ImageFont.truetype(FONT_PATH, 36)
    FONT_TINY  = ImageFont.truetype(FONT_PATH, 28)
    USE_PIL = True
except Exception:
    USE_PIL = False

def put_chinese(frame, text, x, y, font, color=(255,255,255), anchor="lt"):
    """anchor: lt=左上, cc=中心, ct=上中, rb=右下"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)

    if anchor != "lt":
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        if anchor == "cc":
            x, y = x - tw//2, y - th//2
        elif anchor == "ct":
            x = x - tw//2

    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ────────────────────────────────────────────
# 繪圖輔助
# ────────────────────────────────────────────
def draw_rounded_rect(frame, x1, y1, x2, y2, r, color, thickness=-1, alpha=1.0):
    """畫圓角矩形（支援半透明）"""
    if alpha < 1.0:
        overlay = frame.copy()
        _draw_solid_rounded(overlay, x1, y1, x2, y2, r, color)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    elif thickness == -1:
        _draw_solid_rounded(frame, x1, y1, x2, y2, r, color)
    else:
        cv2.rectangle(frame, (x1+r, y1), (x2-r, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(frame, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)

def _draw_solid_rounded(frame, x1, y1, x2, y2, r, color):
    cv2.rectangle(frame, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(frame, (x1, y1+r), (x2, y2-r), color, -1)
    cv2.ellipse(frame, (x1+r, y1+r), (r,r), 180, 0, 90, color, -1)
    cv2.ellipse(frame, (x2-r, y1+r), (r,r), 270, 0, 90, color, -1)
    cv2.ellipse(frame, (x1+r, y2-r), (r,r),  90, 0, 90, color, -1)
    cv2.ellipse(frame, (x2-r, y2-r), (r,r),   0, 0, 90, color, -1)

def draw_progress_arc(frame, cx, cy, radius, progress, color, thickness=4):
    """畫圓弧進度條"""
    angle = int(360 * progress)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, color, thickness)

# ────────────────────────────────────────────
# 遊戲狀態
# ────────────────────────────────────────────
STATE_WAIT   = "wait"    # 等待開始
STATE_PLAY   = "play"    # 答題中
STATE_RESULT = "result"  # 顯示結果

# 顏色
C_BG       = (15, 15, 30)
C_GOLD     = (0, 200, 255)
C_GREEN    = (0, 220, 120)
C_RED      = (60, 60, 220)
C_BLUE     = (220, 140, 50)
C_WHITE    = (255, 255, 255)
C_GRAY     = (120, 120, 120)
C_DARK     = (30, 30, 60)
C_CYAN     = (220, 220, 0)
C_ORANGE   = (0, 140, 255)
C_HOVER    = (50, 180, 255)

CORNER_POSITIONS = ["TL", "TR", "BL", "BR"]  # 四個角落

# ────────────────────────────────────────────
# 主程式
# ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── 遊戲狀態 ──
state           = STATE_WAIT
question        = None
q_start_time    = None
QUESTION_TIME   = 15.0    # 秒

score           = 0
total           = 0
streak          = 0       # 連續答對

# ── 游標 ──
cross_x, cross_y = W//2, H//2
SMOOTH       = 0.75
last_valid_x = cross_x
last_valid_y = cross_y
last_seen_t  = time.time()
LOST_TIMEOUT = 0.8

# ── 停留 / 開火 ──
HOVER_TIME   = 1.0      # 停留選中
hover_start  = None
hover_target = None     # 目前停留的目標 index

fist_start   = None
FIST_TIME    = 0.5      # 握拳確認時間
selected_idx = None     # 停留完成後鎖定

# ── 結果顯示 ──
result_correct  = None
result_show_t   = None
RESULT_SHOW     = 2.5

# ── 找錯字：四字格子位置 ──
def get_char_boxes(mode="wrong"):
    """四字格子：橫排在中央"""
    cx     = W // 2
    cy     = H // 2
    gap    = 20
    size   = 140
    total_w = 4 * size + 3 * gap
    x0     = cx - total_w // 2
    boxes  = []
    for i in range(4):
        x1 = x0 + i * (size + gap)
        y1 = cy - size // 2
        boxes.append((x1, y1, x1 + size, y1 + size))
    return boxes

def get_corner_boxes():
    """四角落選項框"""
    pad  = 30
    size = 130
    safe_b = int(H * 0.68)  # 下方安全線
    return {
        "TL": (pad,         pad,          pad + size,  pad + size),
        "TR": (W-pad-size,  pad,          W-pad,       pad + size),
        "BL": (pad,         safe_b-size,  pad+size,    safe_b),
        "BR": (W-pad-size,  safe_b-size,  W-pad,       safe_b),
    }

def assign_choices_to_corners(choices):
    """把 4 個選項隨機分配到四個角落"""
    corners = list(CORNER_POSITIONS)
    random.shuffle(corners)
    return {corners[i]: choices[i] for i in range(4)}

# ────────────────────────────────────────────
# 繪製各畫面
# ────────────────────────────────────────────
def draw_bg(frame):
    """深色背景漸層"""
    overlay = np.zeros_like(frame)
    overlay[:] = C_BG
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

def draw_wait_screen(frame):
    draw_bg(frame)
    frame = put_chinese(frame, "成語射擊練習", W//2, H//2-120, FONT_LARGE, C_GOLD, "cc")
    frame = put_chinese(frame, "比出大拇指👍 開始遊戲", W//2, H//2+10, FONT_MED, C_WHITE, "cc")
    frame = put_chinese(frame, "握拳🤜 確認答案", W//2, H//2+80, FONT_SMALL, C_CYAN, "cc")
    # 分數
    frame = put_chinese(frame, f"分數：{score} / {total}", W//2, H//2+160, FONT_SMALL, C_GRAY, "cc")
    return frame

def draw_timer_bar(frame, elapsed, total_time):
    """頂部計時條"""
    ratio  = max(0, 1 - elapsed / total_time)
    bar_w  = int(W * ratio)
    # 底色
    cv2.rectangle(frame, (0, 0), (W, 12), (40, 40, 60), -1)
    # 進度
    if ratio > 0.5:
        color = C_GREEN
    elif ratio > 0.25:
        color = C_ORANGE
    else:
        color = C_RED
    cv2.rectangle(frame, (0, 0), (bar_w, 12), color, -1)
    # 剩餘秒數
    remaining = max(0, total_time - elapsed)
    cv2.putText(frame, f"{remaining:.1f}s", (W-80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_fill_question(frame, q, elapsed, corner_map, hover_target, hover_progress, selected_idx):
    """填空題畫面"""
    draw_bg(frame)
    draw_timer_bar(frame, elapsed, QUESTION_TIME)

    # 中央題目框
    qx1, qy1, qx2, qy2 = W//2-260, H//2-90, W//2+260, H//2+90
    draw_rounded_rect(frame, qx1, qy1, qx2, qy2, 18, C_DARK, alpha=0.9)
    draw_rounded_rect(frame, qx1, qy1, qx2, qy2, 18, C_GOLD, thickness=2)
    frame = put_chinese(frame, "填空題", W//2, qy1+8, FONT_TINY, C_GOLD, "ct")
    frame = put_chinese(frame, q["question"], W//2, H//2-20, FONT_LARGE, C_WHITE, "cc")

    # 四個角落選項
    cb = get_corner_boxes()
    for corner, char in corner_map.items():
        x1, y1, x2, y2 = cb[corner]
        idx = list(corner_map.keys()).index(corner)

        # 狀態顏色
        if selected_idx is not None:
            bg_c   = C_GREEN if char == q["correct_char"] else (40, 40, 80)
            brd_c  = C_GREEN if char == q["correct_char"] else C_GRAY
        elif hover_target == corner:
            bg_c  = (30, 80, 120)
            brd_c = C_HOVER
        else:
            bg_c  = C_DARK
            brd_c = C_BLUE

        draw_rounded_rect(frame, x1, y1, x2, y2, 16, bg_c, alpha=0.92)
        draw_rounded_rect(frame, x1, y1, x2, y2, 16, brd_c, thickness=3)
        frame = put_chinese(frame, char, (x1+x2)//2, (y1+y2)//2, FONT_LARGE, C_WHITE, "cc")

        # 停留進度弧
        if hover_target == corner and hover_progress > 0:
            draw_progress_arc(frame, (x1+x2)//2, (y1+y2)//2, 68, hover_progress, C_CYAN, 5)

    # 右上角提示
    frame = put_chinese(frame, f"連續答對：{streak}", W-10, 50, FONT_TINY, C_GOLD, "rb")
    return frame

def draw_wrong_question(frame, q, elapsed, char_boxes, hover_target, hover_progress, selected_idx):
    """找錯字題畫面"""
    draw_bg(frame)
    draw_timer_bar(frame, elapsed, QUESTION_TIME)

    # 題目說明
    frame = put_chinese(frame, "找出錯字！", W//2, 40, FONT_MED, C_GOLD, "ct")
    frame = put_chinese(frame, "瞄準錯字，停留後握拳確認", W//2, 95, FONT_TINY, C_GRAY, "ct")

    # 四個字格子
    for i, (x1, y1, x2, y2) in enumerate(char_boxes):
        char = q["display_idiom"][i]

        if selected_idx is not None:
            is_wrong_pos = (i == q["wrong_pos"])
            bg_c  = (20, 80, 20) if not is_wrong_pos else (30, 30, 120)
            brd_c = C_GREEN if not is_wrong_pos else C_RED
        elif hover_target == i:
            bg_c  = (30, 80, 130)
            brd_c = C_HOVER
        else:
            bg_c  = C_DARK
            brd_c = (100, 80, 40)

        draw_rounded_rect(frame, x1, y1, x2, y2, 18, bg_c, alpha=0.9)
        draw_rounded_rect(frame, x1, y1, x2, y2, 18, brd_c, thickness=3)
        frame = put_chinese(frame, char, (x1+x2)//2, (y1+y2)//2, FONT_LARGE, C_WHITE, "cc")

        # 停留弧
        if hover_target == i and hover_progress > 0:
            draw_progress_arc(frame, (x1+x2)//2, (y1+y2)//2, 70, hover_progress, C_CYAN, 5)

        # 序號小字
        frame = put_chinese(frame, str(i+1), x1+10, y1+5, FONT_TINY, C_GRAY)

    frame = put_chinese(frame, f"連續答對：{streak}", W-10, 50, FONT_TINY, C_GOLD, "rb")
    return frame

def draw_result_overlay(frame, correct, correct_answer, mode):
    """結果覆蓋層"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if correct:
        frame = put_chinese(frame, "✓ 答對了！", W//2, H//2-60, FONT_LARGE, C_GREEN, "cc")
    else:
        frame = put_chinese(frame, "✗ 答錯了", W//2, H//2-60, FONT_LARGE, C_RED, "cc")
        frame = put_chinese(frame, f"正確答案：{correct_answer}", W//2, H//2+20, FONT_MED, C_GOLD, "cc")

    frame = put_chinese(frame, f"分數 {score} / {total}", W//2, H//2+110, FONT_SMALL, C_WHITE, "cc")
    return frame

def draw_cursor(frame, cx, cy, mode_active):
    if not mode_active:
        return
    cv2.circle(frame, (cx, cy), 22, C_CYAN, 2)
    cv2.circle(frame, (cx, cy), 4,  C_RED, -1)
    cv2.line(frame, (cx-28, cy), (cx+28, cy), C_CYAN, 1)
    cv2.line(frame, (cx, cy-28), (cx, cy+28), C_CYAN, 1)

# ────────────────────────────────────────────
# 額外狀態
# ────────────────────────────────────────────
corner_map   = {}   # fill: {corner: char}
char_boxes   = []   # wrong: 四個字的格子

def load_question():
    global question, q_start_time, corner_map, char_boxes
    global hover_start, hover_target, selected_idx, fist_start
    question      = new_question()
    q_start_time  = time.time()
    hover_start   = None
    hover_target  = None
    selected_idx  = None
    fist_start    = None
    if question["mode"] == "fill":
        corner_map = assign_choices_to_corners(question["choices"])
    else:
        char_boxes = get_char_boxes()

# ────────────────────────────────────────────
# 主迴圈
# ────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ── 手勢偵測 ──
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts       = int(time.time() * 1000)
    result   = landmarker.detect_for_video(mp_img, ts)

    thumb_mode   = False
    fist_mode    = False
    hand_present = False

    if result.hand_landmarks:
        lm           = result.hand_landmarks[0]
        last_seen_t  = time.time()
        hand_present = True

        if is_fist(lm):
            fist_mode = True
        elif is_thumb_up(lm):
            thumb_mode = True
            nx = max(0, min(w, int(lm[4].x * w)))
            ny = max(0, min(h, int(lm[4].y * h)))
            cross_x = int(SMOOTH * cross_x + (1-SMOOTH) * nx)
            cross_y = int(SMOOTH * cross_y + (1-SMOOTH) * ny)
            last_valid_x, last_valid_y = cross_x, cross_y
    else:
        if time.time() - last_seen_t < LOST_TIMEOUT:
            thumb_mode = True
            cross_x, cross_y = last_valid_x, last_valid_y
        else:
            thumb_mode = False

    # ════════════════════════════════════════
    # 狀態機
    # ════════════════════════════════════════

    if state == STATE_WAIT:
        frame = draw_wait_screen(frame)
        draw_cursor(frame, cross_x, cross_y, thumb_mode)
        if thumb_mode:
            state = STATE_PLAY
            score, total, streak = 0, 0, 0
            load_question()

    elif state == STATE_PLAY:
        elapsed = time.time() - q_start_time

        # ── 超時 ──
        if elapsed >= QUESTION_TIME:
            result_correct = False
            if question["mode"] == "fill":
                correct_answer = question["correct_char"]
            else:
                correct_answer = question["idiom"][question["wrong_pos"]]
            total += 1
            streak = 0
            result_show_t = time.time()
            state = STATE_RESULT
            selected_idx = -1   # 超時標記
            continue

        hover_progress = 0.0

        # ── 填空題互動 ──
        if question["mode"] == "fill":
            cb = get_corner_boxes()
            hit_corner = None
            if thumb_mode:
                for corner, (x1,y1,x2,y2) in cb.items():
                    if x1 < cross_x < x2 and y1 < cross_y < y2:
                        hit_corner = corner
                        break

            if hit_corner is not None and selected_idx is None:
                if hover_target != hit_corner:
                    hover_target = hit_corner
                    hover_start  = time.time()
                else:
                    hover_elapsed  = time.time() - hover_start
                    hover_progress = min(hover_elapsed / HOVER_TIME, 1.0)
                    if hover_elapsed >= HOVER_TIME:
                        selected_idx = hit_corner  # 鎖定選項，等握拳確認
            else:
                if selected_idx is None:
                    hover_target = None
                    hover_start  = None

            # 握拳確認
            if selected_idx is not None and fist_mode:
                chosen_char    = corner_map[selected_idx]
                result_correct = (chosen_char == question["correct_char"])
                correct_answer = question["correct_char"]
                total += 1
                if result_correct:
                    score  += 1
                    streak += 1
                else:
                    streak = 0
                result_show_t = time.time()
                state = STATE_RESULT

            frame = draw_fill_question(frame, question, elapsed, corner_map,
                                       hover_target, hover_progress, selected_idx)

        # ── 找錯字題互動 ──
        else:
            char_boxes = get_char_boxes()
            hit_idx = None
            if thumb_mode:
                for i, (x1,y1,x2,y2) in enumerate(char_boxes):
                    if x1 < cross_x < x2 and y1 < cross_y < y2:
                        hit_idx = i
                        break

            if hit_idx is not None and selected_idx is None:
                if hover_target != hit_idx:
                    hover_target = hit_idx
                    hover_start  = time.time()
                else:
                    hover_elapsed  = time.time() - hover_start
                    hover_progress = min(hover_elapsed / HOVER_TIME, 1.0)
                    if hover_elapsed >= HOVER_TIME:
                        selected_idx = hit_idx
            else:
                if selected_idx is None:
                    hover_target = None
                    hover_start  = None

            # 握拳確認
            if selected_idx is not None and fist_mode:
                result_correct = (selected_idx == question["wrong_pos"])
                correct_answer = f"第{question['wrong_pos']+1}字「{question['wrong_char']}」→「{question['idiom'][question['wrong_pos']]}」"
                total += 1
                if result_correct:
                    score  += 1
                    streak += 1
                else:
                    streak = 0
                result_show_t = time.time()
                state = STATE_RESULT

            frame = draw_wrong_question(frame, question, elapsed, char_boxes,
                                        hover_target, hover_progress, selected_idx)

        draw_cursor(frame, cross_x, cross_y, thumb_mode)

    elif state == STATE_RESULT:
        # 保留上題畫面底圖再疊結果
        if question["mode"] == "fill":
            elapsed = min(QUESTION_TIME, time.time() - q_start_time)
            frame = draw_fill_question(frame, question, elapsed, corner_map,
                                       hover_target, 1.0, selected_idx)
        else:
            elapsed = min(QUESTION_TIME, time.time() - q_start_time)
            frame = draw_wrong_question(frame, question, elapsed, char_boxes,
                                        hover_target, 1.0, selected_idx)

        if question["mode"] == "fill":
            ca = question["correct_char"]
        else:
            ca = f"第{question['wrong_pos']+1}字 →「{question['idiom'][question['wrong_pos']]}」"

        frame = draw_result_overlay(frame, result_correct, ca, question["mode"])

        if time.time() - result_show_t > RESULT_SHOW:
            load_question()
            state = STATE_PLAY

    cv2.imshow("成語射擊練習", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
