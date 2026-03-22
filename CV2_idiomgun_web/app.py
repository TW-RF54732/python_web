"""
app.py  ──  成語射擊遊戲 Flask 主程式
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import time
import numpy as np
import json
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from idiom_data import generate_questions

app = Flask(__name__)

# ── 設定 ──────────────────────────────────────────────────
MODEL_PATH   = r"C:\python_web\CV2_idiomgun_web\b_H\hand_landmarker.task"  
DIFFICULTY   = "easy"
WRONG_RATIO  = 0.5
TOTAL_Q      = 10
Q_TIME_LIMIT = 15.0
HOVER_TIME   = 1.5
SMOOTH       = 0.65
LOST_TIMEOUT = 0.8

# ── 全局遊戲狀態（執行緒共享） ──────────────────────────────
game_lock = threading.Lock()
game_state = {
    "state": "start",      # start / play / result / final
    "q_index": 0,
    "score": 0,
    "q_start_time": 0.0,
    "ans_result": None,    # correct / wrong / timeout
    "questions": [],
    "cursor_x": 640,
    "cursor_y": 360,
    "thumb_active": False,
    "hover_target": None,
    "hover_start": None,
    "hover_progress": 0.0,
    "last_seen": time.time(),
    "last_valid_x": 640,
    "last_valid_y": 360,
    "frame_w": 1280,
    "frame_h": 720,
}

# ── MediaPipe 初始化 ────────────────────────────────────────
try:
    BaseOptions           = python.BaseOptions
    HandLandmarker        = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode     = vision.RunningMode

    options    = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = HandLandmarker.create_from_options(options)
    MEDIAPIPE_OK = True
except Exception as e:
    print(f"[警告] MediaPipe 初始化失敗: {e}")
    MEDIAPIPE_OK = False
    landmarker = None

cap = None
cap_lock = threading.Lock()

def get_cap():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def process_frame():
    """每幀處理手勢偵測，更新 game_state"""
    camera = get_cap()
    ret, frame = camera.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    W = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    thumb_mode = False

    if MEDIAPIPE_OK and landmarker:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts     = int(time.time() * 1000)
        res    = landmarker.detect_for_video(mp_img, ts)

        with game_lock:
            gs = game_state
            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
                gs["last_seen"] = time.time()
                if lm[4].y < lm[3].y and lm[4].y < lm[5].y:
                    thumb_mode = True
                    nx = max(35, min(W-35,  int(lm[4].x * W)))
                    ny = max(55, min(int(H*0.82), int(lm[4].y * H)))
                    gs["cursor_x"] = int(SMOOTH * gs["cursor_x"] + (1-SMOOTH)*nx)
                    gs["cursor_y"] = int(SMOOTH * gs["cursor_y"] + (1-SMOOTH)*ny)
                    gs["last_valid_x"] = gs["cursor_x"]
                    gs["last_valid_y"] = gs["cursor_y"]
            else:
                if time.time() - gs["last_seen"] < LOST_TIMEOUT:
                    thumb_mode = True
                    gs["cursor_x"] = gs["last_valid_x"]
                    gs["cursor_y"] = gs["last_valid_y"]

            gs["thumb_active"] = thumb_mode
            gs["frame_w"] = W
            gs["frame_h"] = H

    return frame


def generate_video_stream():
    """MJPEG 串流（只傳原始攝影機畫面，遊戲邏輯在前端）"""
    while True:
        frame = process_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.033)  # ~30fps


# ── 路由 ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           total_q=TOTAL_Q,
                           q_time=int(Q_TIME_LIMIT),
                           difficulty=DIFFICULTY)


@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    """前端輪詢：取得游標位置、拇指狀態"""
    with game_lock:
        gs = game_state
        return jsonify({
            "cursor_x":     gs["cursor_x"],
            "cursor_y":     gs["cursor_y"],
            "thumb_active": gs["thumb_active"],
            "frame_w":      gs["frame_w"],
            "frame_h":      gs["frame_h"],
        })


@app.route("/start_game", methods=["POST"])
def start_game():
    data = request.json or {}
    diff       = data.get("difficulty", DIFFICULTY)
    wrong_r    = float(data.get("wrong_ratio", WRONG_RATIO))
    total      = int(data.get("total_q", TOTAL_Q))

    questions = generate_questions(n=total, difficulty=diff, wrong_ratio=wrong_r)
    with game_lock:
        gs = game_state
        gs["state"]       = "play"
        gs["q_index"]     = 0
        gs["score"]       = 0
        gs["questions"]   = questions
        gs["q_start_time"]= time.time()
        gs["ans_result"]  = None

    return jsonify({"ok": True, "total": len(questions)})


@app.route("/question")
def get_question():
    with game_lock:
        gs = game_state
        idx = gs["q_index"]
        qs  = gs["questions"]
        if idx >= len(qs):
            return jsonify({"done": True})
        q   = qs[idx]
        elapsed = time.time() - gs["q_start_time"]
        remain  = max(0.0, Q_TIME_LIMIT - elapsed)
        return jsonify({
            "done":      False,
            "index":     idx,
            "total":     len(qs),
            "score":     gs["score"],
            "remain":    round(remain, 2),
            "q_time":    Q_TIME_LIMIT,
            "type":      q["type"],
            "idiom":     q["idiom"],
            "display":   q.get("display", ""),
            "template":  q.get("template", ""),
            "options":   q.get("options", []),
            "hint":      q["hint"],
            "difficulty":q["difficulty"],
        })


@app.route("/answer", methods=["POST"])
def submit_answer():
    data   = request.json or {}
    chosen = data.get("chosen", "")

    with game_lock:
        gs  = game_state
        idx = gs["q_index"]
        qs  = gs["questions"]
        if idx >= len(qs):
            return jsonify({"error": "no question"})

        q = qs[idx]
        if q["type"] == "wrong":
            correct = chosen == q["wrong_char"]
            correct_str = q["wrong_char"]
        else:
            correct = chosen == q["answer"]
            correct_str = q["answer"]

        if correct:
            gs["score"] += 10
            result = "correct"
        else:
            result = "wrong"

        gs["ans_result"] = result
        gs["state"]      = "result"

        return jsonify({
            "result":      result,
            "correct_str": correct_str,
            "idiom":       q["idiom"],
            "score":       gs["score"],
        })


@app.route("/timeout", methods=["POST"])
def submit_timeout():
    with game_lock:
        gs  = game_state
        idx = gs["q_index"]
        qs  = gs["questions"]
        if idx >= len(qs):
            return jsonify({"error": "no question"})
        q = qs[idx]
        correct_str = q.get("wrong_char") or q.get("answer", "")
        gs["ans_result"] = "timeout"
        gs["state"]      = "result"
        return jsonify({
            "result":      "timeout",
            "correct_str": correct_str,
            "idiom":       q["idiom"],
            "score":       gs["score"],
        })


@app.route("/next", methods=["POST"])
def next_question():
    with game_lock:
        gs = game_state
        gs["q_index"] += 1
        if gs["q_index"] >= len(gs["questions"]):
            gs["state"] = "final"
            return jsonify({"state": "final", "score": gs["score"],
                            "total": len(gs["questions"])})
        gs["state"]       = "play"
        gs["q_start_time"] = time.time()
        gs["ans_result"]  = None
        return jsonify({"state": "play", "q_index": gs["q_index"]})


@app.route("/reset", methods=["POST"])
def reset_game():
    with game_lock:
        gs = game_state
        gs["state"]    = "start"
        gs["q_index"]  = 0
        gs["score"]    = 0
        gs["questions"]= []
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
