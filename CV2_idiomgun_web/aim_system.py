"""
aim_system.py  ── 秒準（Hover-to-Select）判定模組

使用方式：
    aim = AimSystem(hover_time=1.5)
    每幀呼叫 aim.update(target_id, thumb_active)
    用 aim.progress 顯示進度環
    用 aim.fired 判斷是否觸發選擇
"""

import time


class AimSystem:
    """
    追蹤游標在同一目標上的停留時間。
    當停留超過 hover_time 秒時，fire() 觸發並重置。
    """

    def __init__(self, hover_time: float = 1.5):
        self.hover_time     = hover_time
        self.current_target = None
        self.hover_start    = None
        self.progress       = 0.0   # 0.0 ~ 1.0
        self.fired          = False  # 本幀是否觸發

    def update(self, target_id, thumb_active: bool):
        """
        target_id:    目前游標對準的目標（None 表示沒有）
        thumb_active: 拇指是否可見
        回傳 fired（bool）
        """
        self.fired = False

        if not thumb_active or target_id is None:
            self._reset()
            return False

        if target_id != self.current_target:
            self.current_target = target_id
            self.hover_start    = time.time()
            self.progress       = 0.0
            return False

        elapsed       = time.time() - self.hover_start
        self.progress = min(elapsed / self.hover_time, 1.0)

        if self.progress >= 1.0:
            self.fired = True
            self._reset()
            return True

        return False

    def _reset(self):
        self.current_target = None
        self.hover_start    = None
        self.progress       = 0.0

    def get_progress(self, target_id) -> float:
        """取得指定目標的進度（若不是目前目標則回 0）"""
        if target_id != self.current_target:
            return 0.0
        return self.progress
