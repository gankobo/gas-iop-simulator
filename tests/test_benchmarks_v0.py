"""§7-A 動態ベンチマークの回帰テスト（v0 経験式ガスモデル）。

膨張倍率・ピーク時刻・持続日数・非膨張濃度が文献範囲に入ることを確認する。
"""

import numpy as np
import pytest

from units import days_to_min, min_to_days, min_to_hours
from model.gas_empirical import build_gas_params, expansion_factor, shape
from validation.benchmarks import BENCHMARKS, in_range

GASES = ["air", "SF6", "C2F6", "C3F8"]


def _duration_days(gas, c, medium="vitrectomized", residual=0.05):
    """V_gas が初期体積の residual 倍まで縮む時刻（持続）を日で返す。"""
    p = build_gas_params(gas, c, medium)
    ts = np.linspace(0.0, days_to_min(250), 400000)
    vs = np.array([shape(t, p) for t in ts])
    below = ts[vs <= residual]
    assert len(below) > 0, f"{gas} did not dissolve within 150 days"
    return min_to_days(below[0])


def _peak(gas, c, medium="vitrectomized"):
    p = build_gas_params(gas, c, medium)
    ts = np.linspace(0.0, days_to_min(250), 400000)
    vs = np.array([shape(t, p) for t in ts])
    i = int(vs.argmax())
    return min_to_hours(ts[i]), float(vs[i])


@pytest.mark.parametrize("gas", ["SF6", "C2F6", "C3F8"])
def test_expansion_factor_100pct(gas):
    """100% 注入時のピーク膨張倍率が文献範囲内。"""
    _, peak_fac = _peak(gas, 1.0)
    bm = BENCHMARKS[gas]
    assert in_range(peak_fac, bm.expansion_100pct), (
        f"{gas} expansion {peak_fac:.2f} not in {bm.expansion_100pct}"
    )


@pytest.mark.parametrize("gas", ["SF6", "C2F6", "C3F8"])
def test_peak_time_100pct(gas):
    """100% 注入時のピーク到達時刻が文献範囲内。"""
    peak_h, _ = _peak(gas, 1.0)
    bm = BENCHMARKS[gas]
    assert in_range(peak_h, bm.peak_time_h), (
        f"{gas} peak time {peak_h:.1f}h not in {bm.peak_time_h}"
    )


@pytest.mark.parametrize("gas", ["SF6", "C2F6", "C3F8"])
def test_duration_at_clinical_conc(gas):
    """臨床濃度での持続日数が文献範囲内。"""
    bm = BENCHMARKS[gas]
    c = sum(bm.nonexpansile_conc) / 2.0  # 非膨張濃度の中央
    dur = _duration_days(gas, c)
    assert in_range(dur, bm.duration_days), (
        f"{gas} duration {dur:.1f}d not in {bm.duration_days}"
    )


def test_air_nonexpansion():
    """air は膨張しない（ピーク倍率 ≤ 1）。§6-5。"""
    _, peak_fac = _peak("air", 1.0)
    assert peak_fac <= 1.001, f"air expanded to {peak_fac:.3f}"


def test_air_duration():
    """air の持続が 5–7 日。"""
    dur = _duration_days("air", 1.0)
    assert in_range(dur, BENCHMARKS["air"].duration_days), f"air duration {dur:.1f}d"


@pytest.mark.parametrize("gas", ["SF6", "C2F6", "C3F8"])
def test_nonexpansile_concentration_iso(gas):
    """非膨張濃度では膨張倍率がほぼ 1（等容）。"""
    bm = BENCHMARKS[gas]
    c_ne = sum(bm.nonexpansile_conc) / 2.0
    E = expansion_factor(gas, c_ne)
    assert E <= 1.10, f"{gas} at non-expansile c={c_ne} gives E={E:.2f} (expected ~1)"


@pytest.mark.parametrize("gas", ["SF6", "C2F6", "C3F8"])
def test_concentration_monotonic_expansion(gas):
    """濃度を上げると膨張倍率は単調増加。"""
    bm = BENCHMARKS[gas]
    cs = np.linspace(sum(bm.nonexpansile_conc) / 2.0, 1.0, 10)
    Es = [expansion_factor(gas, c) for c in cs]
    assert all(b >= a - 1e-9 for a, b in zip(Es, Es[1:])), f"{gas} non-monotonic {Es}"


@pytest.mark.parametrize("gas", ["SF6", "C2F6", "C3F8"])
def test_intact_vitreous_longer_life(gas):
    """phakic + intact vitreous は vitrectomized より長寿命（半減期 2–3 倍）。"""
    dur_vit = _duration_days(gas, 1.0, "vitrectomized")
    dur_intact = _duration_days(gas, 1.0, "phakic_intact_vitreous")
    ratio = dur_intact / dur_vit
    assert 2.0 <= ratio <= 3.0, f"{gas} intact/vit duration ratio {ratio:.2f} not 2–3x"
