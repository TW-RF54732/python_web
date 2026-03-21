import cv2
import mediapipe as mp
import time
import random
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
# Pillow 中文渲染
# ========================
try:
    from PIL import Image, ImageDraw, ImageFont
    FONT_PATH = "C:/Windows/Fonts/msjh.ttc"
    PIL_OK = True
except ImportError:
    PIL_OK = False

_font_cache = {}

def _get_font(size):
    if size not in _font_cache:
        from PIL import ImageFont
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]

def measure(text, size=36):
    if not PIL_OK:
        return len(text) * size, size + 4
    from PIL import Image, ImageDraw
    img  = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)
    bb   = draw.textbbox((0, 0), text, font=_get_font(size))
    return bb[2]-bb[0], bb[3]-bb[1]

def put_cn(frame, text, pos, size=36, color=(255,255,255), bg=None, pad=8):
    if not PIL_OK:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return
    from PIL import Image, ImageDraw
    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _get_font(size)
    bb   = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    if bg is not None:
        draw.rectangle([pos[0]-pad, pos[1]-pad,
                        pos[0]+tw+pad, pos[1]+th+pad], fill=bg)
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ========================
# 題目資料（10題）
# ========================
RAW_QUESTIONS = [
    # 選錯字 ─────────────────────────────────────────────────────
    {"type":"wrong","display":"一失二鳥","wrong_char":"失","correct_char":"石",
     "hint":"找出錯字並秒準 1.5 秒"},
    {"type":"wrong","display":"半途兒廢","wrong_char":"兒","correct_char":"而",
     "hint":"找出錯字並秒準 1.5 秒"},
    {"type":"wrong","display":"媽到成功","wrong_char":"媽","correct_char":"馬",
     "hint":"找出錯字並秒準 1.5 秒"},
    {"type":"wrong","display":"畫蛇添族","wrong_char":"族","correct_char":"足",
     "hint":"找出錯字並秒準 1.5 秒"},
    {"type":"wrong","display":"守豬待兔","wrong_char":"豬","correct_char":"株",
     "hint":"找出錯字並秒準 1.5 秒"},
    # 填空 ────────────────────────────────────────────────────────
    {"type":"fill","template":"一＿二鳥","answer":"石",
     "options":["石","失","時","事"],"hint":"秒準正確的字 1.5 秒"},
    {"type":"fill","template":"半途＿廢","answer":"而",
     "options":["而","兒","耳","二"],"hint":"秒準正確的字 1.5 秒"},
    {"type":"fill","template":"畫蛇添＿","answer":"足",
     "options":["族","卒","足","祖"],"hint":"秒準正確的字 1.5 秒"},
    {"type":"fill","template":"守＿待兔","answer":"株",
     "options":["竹","株","豬","主"],"hint":"秒準正確的字 1.5 秒"},
    {"type":"fill","template":"＿到成功","answer":"馬",
     "options":["媽","碼","馬","罵"],"hint":"秒準正確的字 1.5 秒"},
]
random.shuffle(RAW_QUESTIONS)
QUESTIONS = RAW_QUESTIONS[:10]

# ========================
# 攝影機
# ========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ========================
# 游標 / 手勢
# ========================
SMOOTH       = 0.65
HOVER_TIME   = 1.5
LOST_TIMEOUT = 0.8
Q_TIME_LIMIT = 15.0

cross_x, cross_y   = W//2, H//2
last_valid_x       = cross_x
last_valid_y       = cross_y
last_seen_time     = time.time()
hover_start_time   = None
current_target     = None   # int index

# ========================
# 安全區
# ========================
SAFE_T = 55
SAFE_B = int(H * 0.82)
SAFE_L = 35
SAFE_R = W - 35

# ========================
# 遊戲狀態
# ========================
ST_START  = "start"
ST_PLAY   = "play"
ST_RESULT = "result"
ST_FINAL  = "final"

game_state    = ST_START
q_index       = 0
score         = 0
q_start_time  = 0.0
ans_result    = None   # "correct"/"wrong"/"timeout"
chosen_ans    = ""
result_shown  = False

# ========================
# 繪製工具
# ========================
def draw_ring(frame, cx, cy, r, prog, col=(0,215,110), thick=4):
    deg = int(360 * min(prog, 1.0))
    if deg > 0:
        cv2.ellipse(frame, (cx,cy), (r,r), -90, 0, deg, col, thick, cv2.LINE_AA)

def draw_box(frame, x1, y1, x2, y2, label, fsize=48,
             hover=False, prog=0.0):
    """簡約白底方塊，僅在 hover 時顯示進度環"""
    # 背景
    ov = frame.copy()
    cv2.rectangle(ov, (x1,y1), (x2,y2), (248,248,248), -1)
    cv2.addWeighted(ov, 0.93, frame, 0.07, 0, frame)
    # 邊框
    bc    = (0,200,100) if hover else (190,190,190)
    thick = 3 if hover else 1
    cv2.rectangle(frame, (x1,y1), (x2,y2), bc, thick, cv2.LINE_AA)
    # 字
    tw, th = measure(label, fsize)
    put_cn(frame, label,
           (x1+(x2-x1-tw)//2, y1+(y2-y1-th)//2),
           fsize, (30,30,30))
    # 進度環（只在 hover 時）
    if hover and prog > 0:
        draw_ring(frame, (x1+x2)//2, (y1+y2)//2,
                  (x2-x1)//2+12, prog)

def draw_mask(frame, alpha=0.60):
    ov = np.zeros_like(frame)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)

def draw_cursor(frame, cx, cy, prog=0.0, active=True):
    if not active:
        return
    col = (0,215,110)
    cv2.circle(frame, (cx,cy), 6, col, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx,cy), 22, col, 1, cv2.LINE_AA)
    if prog > 0:
        draw_ring(frame, cx, cy, 22, prog, col, 3)

def centered_text(frame, text, y, size, color):
    tw, _ = measure(text, size)
    put_cn(frame, text, (W//2 - tw//2, y), size, color)

# ========================
# 版面計算
# ========================
CHAR_W = 110
CHAR_H = 110
CHAR_GAP = 18

def char_boxes(display):
    n = len(display)
    total = n*CHAR_W + (n-1)*CHAR_GAP
    x0 = W//2 - total//2
    y0 = H//2 - CHAR_H//2
    boxes = []
    for i,ch in enumerate(display):
        x1 = x0 + i*(CHAR_W+CHAR_GAP)
        boxes.append((x1, y0, x1+CHAR_W, y0+CHAR_H, ch))
    return boxes

CORNER_W = 150
CORNER_H = 100
CORNER_MARGIN = 55

def corner_boxes(options):
    pts = [
        (SAFE_L+CORNER_MARGIN,          SAFE_T+CORNER_MARGIN),
        (W-SAFE_L-CORNER_MARGIN-CORNER_W, SAFE_T+CORNER_MARGIN),
        (SAFE_L+CORNER_MARGIN,          SAFE_B-CORNER_MARGIN-CORNER_H),
        (W-SAFE_L-CORNER_MARGIN-CORNER_W, SAFE_B-CORNER_MARGIN-CORNER_H),
    ]
    return [(pts[i][0], pts[i][1],
             pts[i][0]+CORNER_W, pts[i][1]+CORNER_H,
             options[i]) for i in range(min(4,len(options)))]

# ========================
# 主迴圈
# ========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # ─── 手勢偵測 ───────────────────────────────────────────────
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts       = int(time.time()*1000)
    res      = landmarker.detect_for_video(mp_img, ts)

    thumb_mode = False
    if res.hand_landmarks:
        lm = res.hand_landmarks[0]
        last_seen_time = time.time()
        up = lm[4].y < lm[3].y and lm[4].y < lm[5].y
        if up:
            thumb_mode = True
            nx = max(SAFE_L, min(SAFE_R, int(lm[4].x*W)))
            ny = max(SAFE_T, min(SAFE_B, int(lm[4].y*H)))
            cross_x = int(SMOOTH*cross_x + (1-SMOOTH)*nx)
            cross_y = int(SMOOTH*cross_y + (1-SMOOTH)*ny)
            last_valid_x, last_valid_y = cross_x, cross_y
    else:
        if time.time()-last_seen_time < LOST_TIMEOUT:
            thumb_mode = True
            cross_x, cross_y = last_valid_x, last_valid_y

    cur_prog = 0.0
    if thumb_mode and hover_start_time is not None:
        cur_prog = min((time.time()-hover_start_time)/HOVER_TIME, 1.0)

    # ══════════════════════════════════════════════════════════════
    # 開始畫面
    # ══════════════════════════════════════════════════════════════
    if game_state == ST_START:
        centered_text(frame, "成語闖關", H//2-110, 72, (255,255,255))
        centered_text(frame, "豎起大拇指 停留 1.5 秒 開始", H//2-20, 36, (180,210,255))
        centered_text(frame, "共 10 題，每題限時 15 秒", H//2+40, 30, (150,150,150))

        if thumb_mode:
            if hover_start_time is None:
                hover_start_time = time.time()
            draw_cursor(frame, cross_x, cross_y, cur_prog)
            if cur_prog >= 1.0:
                game_state = ST_PLAY
                q_index = 0; score = 0
                q_start_time = time.time()
                hover_start_time = None; current_target = None
        else:
            hover_start_time = None

    # ══════════════════════════════════════════════════════════════
    # 遊戲進行
    # ══════════════════════════════════════════════════════════════
    elif game_state == ST_PLAY:
        q       = QUESTIONS[q_index]
        now     = time.time()
        remain  = max(0.0, Q_TIME_LIMIT-(now-q_start_time))

        # HUD
        put_cn(frame, f"第 {q_index+1}/10 題", (20,18), 26, (180,180,180))
        put_cn(frame, f"分數 {score}", (20,50), 24, (180,180,180))
        timer_c = (0,200,100) if remain>5 else (50,80,255)
        tw,_ = measure(f"{remain:.1f}s", 38)
        put_cn(frame, f"{remain:.1f}s", (W-tw-20, 18), 38, timer_c)
        # 計時條
        bw = int((remain/Q_TIME_LIMIT)*(W-80))
        cv2.rectangle(frame, (40,H-16), (40+bw, H-7), timer_c, -1)
        cv2.rectangle(frame, (40,H-16), (W-40, H-7), (70,70,70), 1)

        # Hint
        hw,_ = measure(q["hint"], 26)
        put_cn(frame, q["hint"], (W//2-hw//2, H-44), 26, (140,140,140))

        # 超時
        if remain <= 0:
            game_state = ST_RESULT
            ans_result = "timeout"; chosen_ans = ""
            hover_start_time = None; current_target = None

        # ── 選錯字 ─────────────────────────────────────────────
        if q["type"] == "wrong":
            cboxes = char_boxes(q["display"])
            hit = None
            if thumb_mode:
                for i,(x1,y1,x2,y2,_) in enumerate(cboxes):
                    if x1<cross_x<x2 and y1<cross_y<y2:
                        hit=i; break

            for i,(x1,y1,x2,y2,ch) in enumerate(cboxes):
                hov  = (hit==i)
                prog = (min((time.time()-hover_start_time)/HOVER_TIME,1.0)
                        if hov and current_target==i and hover_start_time else 0.0)
                draw_box(frame,x1,y1,x2,y2,ch,54,hov,prog)

            if thumb_mode and hit is not None:
                if current_target != hit:
                    current_target=hit; hover_start_time=time.time()
                elif time.time()-hover_start_time >= HOVER_TIME:
                    chosen_ans = cboxes[hit][4]
                    ans_result = "correct" if chosen_ans==q["wrong_char"] else "wrong"
                    if ans_result=="correct": score+=10
                    game_state=ST_RESULT
                    hover_start_time=None; current_target=None
            else:
                current_target=None; hover_start_time=None

        # ── 填空 ───────────────────────────────────────────────
        elif q["type"] == "fill":
            tw2,th2 = measure(q["template"], 68)
            put_cn(frame, q["template"], (W//2-tw2//2, H//2-th2//2), 68, (255,255,255))

            cboxes = corner_boxes(q["options"])
            hit = None
            if thumb_mode:
                for i,(x1,y1,x2,y2,_) in enumerate(cboxes):
                    if x1<cross_x<x2 and y1<cross_y<y2:
                        hit=i; break

            for i,(x1,y1,x2,y2,opt) in enumerate(cboxes):
                hov  = (hit==i)
                prog = (min((time.time()-hover_start_time)/HOVER_TIME,1.0)
                        if hov and current_target==i and hover_start_time else 0.0)
                draw_box(frame,x1,y1,x2,y2,opt,52,hov,prog)

            if thumb_mode and hit is not None:
                if current_target != hit:
                    current_target=hit; hover_start_time=time.time()
                elif time.time()-hover_start_time >= HOVER_TIME:
                    chosen_ans = cboxes[hit][4]
                    ans_result = "correct" if chosen_ans==q["answer"] else "wrong"
                    if ans_result=="correct": score+=10
                    game_state=ST_RESULT
                    hover_start_time=None; current_target=None
            else:
                current_target=None; hover_start_time=None

        # 游標（最後畫，蓋在方塊上）
        draw_cursor(frame, cross_x, cross_y, cur_prog, thumb_mode)

    # ══════════════════════════════════════════════════════════════
    # 單題結果覆蓋層
    # ══════════════════════════════════════════════════════════════
    elif game_state == ST_RESULT:
        # ① 先畫半透明黑色遮罩，隔開背景題目
        draw_mask(frame, alpha=0.62)

        q = QUESTIONS[q_index]

        if ans_result == "correct":
            status_text = "O 答對了！"
            status_col  = (60, 210, 100)
            correct_str = q.get("wrong_char") or q.get("answer","")
            detail      = f"錯字正是「{correct_str}」"
        elif ans_result == "wrong":
            status_text = "X  答錯了"
            status_col  = (80,100,255)
            correct_str = q.get("wrong_char") or q.get("answer","")
            detail      = f"正確答案是「{correct_str}」"
        else:
            status_text = "時間到"
            status_col  = (80,180,255)
            correct_str = q.get("wrong_char") or q.get("answer","")
            detail      = f"正確答案是「{correct_str}」"

        # ② 結果卡片（深底色 + 彩色邊框）
        CW, CH = 560, 270
        cx0 = W//2 - CW//2
        cy0 = H//2 - CH//2
        ov2 = frame.copy()
        cv2.rectangle(ov2, (cx0,cy0), (cx0+CW,cy0+CH), (22,22,32), -1)
        cv2.addWeighted(ov2, 0.97, frame, 0.03, 0, frame)
        cv2.rectangle(frame, (cx0,cy0), (cx0+CW,cy0+CH),
                      status_col, 2, cv2.LINE_AA)

        put_cn(frame, status_text, (cx0+35, cy0+35),  52, status_col)
        put_cn(frame, detail,      (cx0+35, cy0+115), 36, (215,215,215))
        put_cn(frame, "比大拇指 1.5 秒 → 繼續",
               (cx0+35, cy0+190), 28, (130,130,130))

        # ③ 游標 + 進度（任意位置停留即可繼續）
        if thumb_mode:
            if hover_start_time is None:
                hover_start_time = time.time()
            prog2 = min((time.time()-hover_start_time)/HOVER_TIME, 1.0)
            draw_cursor(frame, cross_x, cross_y, prog2)
            if prog2 >= 1.0:
                q_index += 1
                if q_index >= len(QUESTIONS):
                    game_state = ST_FINAL
                else:
                    game_state = ST_PLAY
                    q_start_time = time.time()
                hover_start_time=None; current_target=None
        else:
            hover_start_time = None

    # ══════════════════════════════════════════════════════════════
    # 最終結果
    # ══════════════════════════════════════════════════════════════
    elif game_state == ST_FINAL:
        draw_mask(frame, alpha=0.72)

        CW, CH = 580, 330
        cx0 = W//2 - CW//2
        cy0 = H//2 - CH//2
        ov2 = frame.copy()
        cv2.rectangle(ov2, (cx0,cy0), (cx0+CW,cy0+CH), (18,18,28), -1)
        cv2.addWeighted(ov2, 0.97, frame, 0.03, 0, frame)
        cv2.rectangle(frame, (cx0,cy0), (cx0+CW,cy0+CH), (80,190,255), 2, cv2.LINE_AA)

        put_cn(frame, "遊戲結束！", (cx0+40, cy0+30),  56, (80,190,255))
        score_col = (255,200,50) if score>=70 else (200,200,200)
        put_cn(frame, f"得分：{score} / 100", (cx0+40, cy0+115), 50, score_col)

        if   score >= 90: rank = "成語達人"
        elif score >= 70: rank = "表現良好"
        elif score >= 50: rank = "繼續加油"
        else:             rank = "多多練習"
        put_cn(frame, rank, (cx0+40, cy0+195), 38, (200,200,200))
        put_cn(frame, "按 ESC 離開", (cx0+40, cy0+268), 26, (100,100,100))

    # ─── 顯示 ───────────────────────────────────────────────────
    cv2.imshow("成語遊戲", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
