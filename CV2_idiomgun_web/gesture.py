"""
gesture.py  ── 手勢判斷模組
提供 is_thumb_up(landmarks) 給主程式使用
"""


def is_thumb_up(landmarks) -> bool:
    """
    landmarks: MediaPipe HandLandmarker 的 hand_landmarks[0]（NormalizedLandmark list）
    規則：拇指尖端(4) 高於關節(3)，且高於食指根(5) → 拇指朝上
    """
    if not landmarks or len(landmarks) < 6:
        return False
    tip    = landmarks[4]   # 拇指尖
    joint  = landmarks[3]   # 拇指第一關節
    idx_base = landmarks[5] # 食指掌骨根

    # y 座標在影像座標系：值越小越高
    return tip.y < joint.y and tip.y < idx_base.y


def thumb_tip_pos(landmarks, frame_w: int, frame_h: int,
                  safe_l=35, safe_r=None, safe_t=55, safe_b=None) -> tuple[int, int]:
    """
    取得拇指尖在畫面中的像素座標（已 clamp 至安全邊界）
    """
    if safe_r is None:
        safe_r = frame_w - 35
    if safe_b is None:
        safe_b = int(frame_h * 0.82)

    tip = landmarks[4]
    nx  = max(safe_l, min(safe_r, int(tip.x * frame_w)))
    ny  = max(safe_t, min(safe_b, int(tip.y * frame_h)))
    return nx, ny
