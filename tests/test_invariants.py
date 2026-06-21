"""§6 物理法則インバリアントの検証（v0 統合シミュレータ）。

現実的な（適正〜中等度）シナリオで、各インバリアントが破れないことを確認する。
極端な過充填（房水枯渇で IOP 非生理域）は v0 剛性外挿の信頼域外として
別途 warnings で扱う（test_overfill_warnings 参照）。
"""

import numpy as np
import pytest

from model.simulator import SimConfig, simulate

# 現実的シナリオ（adequate〜中等度過膨張）。
REALISTIC = [
    ("air", 1.0, 0.5),
    ("SF6", 0.20, 0.6),
    ("SF6", 1.0, 0.4),
    ("C2F6", 0.16, 0.7),
    ("C3F8", 0.14, 0.85),
]


def _run(gas, c, f, **kw):
    kw.setdefault("t_end_days", 120)
    return simulate(SimConfig(gas=gas, c=c, f=f, V_vit=4.5, **kw))


@pytest.mark.parametrize("gas,c,f", REALISTIC)
def test_volumes_nonnegative(gas, c, f):
    """§6-1相当: 気泡体積・残存流体量は非負。"""
    r = _run(gas, c, f)
    assert (r.df["V_gas_mL"] >= -1e-9).all()
    assert (r.df["V_fluid_mL"] >= -1e-6).all(), "V_fluid went negative (floor broken)"


@pytest.mark.parametrize("gas,c,f", REALISTIC)
def test_final_absorption(gas, c, f):
    """§6-4: t→大 で気泡は完全吸収（V_gas→0）。永久に残らない。"""
    r = _run(gas, c, f)
    assert r.df["V_gas_mL"].iloc[-1] < 0.05 * r.ic.V_gas0, "bubble did not absorb"


def test_air_non_expansion():
    """§6-5: air は膨張しない（max V_gas / V_gas0 ≤ 1 + 数%）。"""
    r = _run("air", 1.0, 0.5)
    ratio = r.df["V_gas_mL"].max() / r.ic.V_gas0
    assert ratio <= 1.03, f"air expanded ratio={ratio:.3f}"


@pytest.mark.parametrize("gas,c,f", REALISTIC)
def test_single_peak_no_oscillation(gas, c, f):
    """§6-6: 気泡体積は膨張→単峰→吸収（多峰の数値振動なし）。"""
    v = r = _run(gas, c, f).df["V_gas_mL"].values
    # 符号変化（増→減）の回数を数える。単峰なら高々 1 回。
    dv = np.diff(v)
    sign_changes = np.sum((dv[:-1] > 1e-12) & (dv[1:] < -1e-12))
    assert sign_changes <= 1, f"{gas}: {sign_changes} peaks (oscillation suspected)"


@pytest.mark.parametrize("gas,c,f", REALISTIC)
def test_IOP_lower_bound(gas, c, f):
    """§6-8: IOP は上強膜静脈圧付近を恒常的に下回らない（房水産生で回復）。"""
    r = _run(gas, c, f)
    p_ev = r.config.aqueous.P_ev_mmHg
    assert r.df["IOP_mmHg"].min() >= p_ev - 1.0, (
        "IOP collapsed below episcleral venous P"
    )


@pytest.mark.parametrize("gas,c,f", REALISTIC)
def test_baseline_self_consistent(gas, c, f):
    """ガス外乱が収束したあと、IOP は房水平衡（baseline）へ戻る。"""
    r = _run(gas, c, f)
    final_iop = r.df["IOP_mmHg"].iloc[-1]
    assert abs(final_iop - r.IOP0) < 1.0, (
        f"final IOP {final_iop:.1f} != baseline {r.IOP0:.1f}"
    )


def test_monotonic_IOP_peak_with_fill():
    """§7-C: 置換率 f を上げるとピーク IOP は単調増大。"""
    peaks = []
    for f in [0.3, 0.4, 0.5]:
        peaks.append(_run("SF6", 1.0, f, t_end_days=60).peak_IOP)
    assert all(b >= a - 1e-6 for a, b in zip(peaks, peaks[1:])), (
        f"non-monotonic {peaks}"
    )


def test_monotonic_IOP_peak_with_conc():
    """§7-C: 濃度 c を上げるとピーク IOP は単調増大。"""
    peaks = []
    for c in [0.20, 0.5, 1.0]:
        peaks.append(_run("SF6", c, 0.5, t_end_days=60).peak_IOP)
    assert all(b >= a - 1e-6 for a, b in zip(peaks, peaks[1:])), (
        f"non-monotonic {peaks}"
    )


def test_overfill_warnings():
    """過充填（膨張性ガス高 f）では warnings が立つ。"""
    r = _run("C3F8", 1.0, 0.5, t_end_days=60)
    assert len(r.warnings) >= 1, "expected overfill/non-physical warning"
    assert r.max_fill_fraction > 1.0


def test_adequate_fill_no_warnings():
    """適正充填では warnings が立たない。"""
    r = _run("SF6", 0.20, 0.6)
    assert len(r.warnings) == 0, f"unexpected warnings: {r.warnings}"
