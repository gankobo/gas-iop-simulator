"""一次文献ベンチマーク（§7-A）と合格判定。

これらの定数・範囲は複数レビュー/原著で一致する定説値（§7-A の表）。
v0 はこの「ガス動態（膨張倍率・ピーク時刻・持続日数・非膨張濃度）」を再現することが
第一の合格条件。テスト（tests/test_benchmarks_v0.py）はここを参照する。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GasDynamicsBenchmark:
    """1 ガスの動態ベンチマーク（許容範囲つき）。時刻は hour、持続は day。"""

    gas: str
    expansion_100pct: tuple  # (min, max) 100%注入時のピーク膨張倍率
    peak_time_h: tuple  # (min, max) ピーク到達時刻 [hour]（air は None）
    nonexpansile_conc: tuple  # (min, max) 非膨張（等膨張）濃度（容積分率）。air は None
    duration_days: tuple  # (min, max) 希釈時の持続日数 [day]


# §7-A の表。範囲は文献の幅＋v0 経験式の許容を見込んで設定。
BENCHMARKS = {
    "air": GasDynamicsBenchmark(
        gas="air",
        expansion_100pct=(1.0, 1.0),  # 膨張なし
        peak_time_h=None,
        nonexpansile_conc=None,
        duration_days=(5.0, 7.0),
    ),
    "SF6": GasDynamicsBenchmark(
        gas="SF6",
        expansion_100pct=(1.9, 2.6),  # 約 2–2.5 倍
        peak_time_h=(24.0, 48.0),  # 1–2 日
        nonexpansile_conc=(0.18, 0.20),  # 18–20%
        duration_days=(10.0, 18.0),  # 約 14 日（20%）
    ),
    "C2F6": GasDynamicsBenchmark(
        gas="C2F6",
        expansion_100pct=(2.8, 3.5),  # 約 3 倍（Lincoff 3.3）
        peak_time_h=(36.0, 60.0),
        nonexpansile_conc=(0.15, 0.17),  # 16%
        duration_days=(28.0, 35.0),
    ),
    "C3F8": GasDynamicsBenchmark(
        gas="C3F8",
        expansion_100pct=(3.6, 4.4),  # 約 4 倍
        peak_time_h=(72.0, 96.0),  # 3–4 日
        nonexpansile_conc=(0.12, 0.14),  # 12–14%
        duration_days=(55.0, 79.0),  # 約 8 週（14%）
    ),
}


def in_range(value: float, rng: tuple, tol_frac: float = 0.0) -> bool:
    """value が [min*(1-tol), max*(1+tol)] に入るか。"""
    lo, hi = rng
    return lo * (1.0 - tol_frac) <= value <= hi * (1.0 + tol_frac)
