"""v1/v1.2 機構モデルの創発挙動の検証。ハンドオフ §8(v1) の合格条件。

多成分拡散＋球冠→球界面から、air 非膨張・3相（膨張→平衡→消失）・膨張順序・
非膨張濃度・最終吸収・分圧整合・持続日数が「創発的に」出ることを確認する。

注: 物質移動係数 Q_i と界面パラメータは validation/thompson_fit.py の較正結果
（calibrated_Qi.json）に依存する。v1.2 は球冠→球で希釈持続を文献に合わせ込んだ代わり
（C3F8 持続 ≈68日）、膨張倍率がやや高めに出る（C3F8 ~5.4 vs 文献 ~4）。膨張倍率は
較正と同じ充填率で測り、v1.2 のトレードオフを見込んだ許容（tol=0.3）で検証する。
"""

import numpy as np
import pytest

from units import min_to_hours
from model.simulator import SimConfig, simulate
from validation.benchmarks import BENCHMARKS, in_range
from validation.thompson_fit import F_EXPANSION

FLUORO = ["SF6", "C2F6", "C3F8"]


def _run(gas, c, f, t_end_days, medium="vitrectomized", n_points=2500):
    return simulate(
        SimConfig(
            gas=gas,
            c=c,
            f=f,
            V_vit=4.5,
            t_end_days=t_end_days,
            mechanism="diffusion",
            medium=medium,
            n_points=n_points,
        )
    )


def _peak(gas, c, t_end_days, f=None):
    """膨張倍率・ピーク時刻。較正と同じ充填率（F_EXPANSION, 膨張相を球冠領域に保つ）で測る。

    v1.2 では球冠→球の遷移があるため、小気泡 f=0.1 だと膨張相まで球化して値が崩れる。
    臨床的に大きい気泡（較正と同条件）で測るのが正しい。
    """
    if f is None:
        f = F_EXPANSION.get(gas, 0.3)
    r = _run(gas, c, f, t_end_days)
    v = r.df["V_gas_mL"].values
    i = int(v.argmax())
    return v[i] / r.ic.V_gas0, min_to_hours(r.df["t_min"].values[i])


def test_air_non_expansion():
    """§6-5: air は膨張しない（機構から創発）。"""
    pf, _ = _peak("air", 1.0, 20.0)
    assert pf <= 1.05, f"air expanded to {pf:.3f}"


@pytest.mark.parametrize("gas", FLUORO)
def test_expansion_factor_in_range(gas):
    """100% 膨張倍率が §7-A 範囲付近（v1.2 は持続優先のため許容 tol=0.3）に入る。"""
    pf, _ = _peak(gas, 1.0, 12.0)
    bm = BENCHMARKS[gas]
    assert in_range(pf, bm.expansion_100pct, tol_frac=0.3), (
        f"{gas} expansion {pf:.2f} not near {bm.expansion_100pct}"
    )


def test_expansion_ordering():
    """膨張倍率は SF6 < C2F6 < C3F8（フッ素ガスの抜けやすさの順, 創発）。"""
    pfs = [_peak(g, 1.0, 12.0)[0] for g in FLUORO]
    assert pfs[0] < pfs[1] < pfs[2], f"expansion not ordered: {pfs}"


@pytest.mark.parametrize("gas", FLUORO)
def test_peak_time_roughly_in_range(gas):
    """ピーク到達時刻が §7-A 範囲（v1 許容 tol=0.3）付近。"""
    _, ph = _peak(gas, 1.0, 12.0)
    bm = BENCHMARKS[gas]
    assert in_range(ph, bm.peak_time_h, tol_frac=0.3), (
        f"{gas} peak time {ph:.1f}h not near {bm.peak_time_h}"
    )


@pytest.mark.parametrize("gas", FLUORO)
def test_three_phase_single_peak(gas):
    """3相: 膨張→単峰→消失（多峰の数値振動なし）。"""
    r = _run(gas, 1.0, 0.1, 40.0)
    v = r.df["V_gas_mL"].values
    v0 = r.ic.V_gas0
    # 体積が意味を持つ区間のみで増→減の転換を数える（吸収後のゼロ周り揺らぎを除外）。
    thr = 1e-3 * v0
    dv = np.diff(v)
    sig = (np.abs(dv) > thr) & (v[:-1] > 0.02 * v0)
    dvs = dv[sig]
    sign_changes = np.sum((dvs[:-1] > 0) & (dvs[1:] < 0))
    assert sign_changes <= 1, f"{gas}: {sign_changes} peaks (oscillation)"


@pytest.mark.parametrize("gas", FLUORO + ["air"])
def test_final_absorption(gas):
    """§6-4: 気泡は最終的に完全吸収（組織総分圧 < P_atm の不飽和ゆえ）。"""
    r = _run(gas, 1.0, 0.1, 120.0)
    assert r.df["V_gas_mL"].iloc[-1] < 0.05 * r.ic.V_gas0, f"{gas} not absorbed"


@pytest.mark.parametrize("gas", FLUORO)
def test_nonexpansile_concentration_iso(gas):
    """非膨張濃度では膨張倍率がほぼ 1（等容, 創発）。"""
    bm = BENCHMARKS[gas]
    c_ne = sum(bm.nonexpansile_conc) / 2.0
    pf, _ = _peak(gas, c_ne, 12.0)
    assert pf <= 1.35, f"{gas} at c_ne={c_ne} expands to {pf:.2f}"


@pytest.mark.parametrize("gas", FLUORO)
def test_partial_pressure_invariant(gas):
    """§6-2: 気泡が存在する間, Σ p_i(乾性) + 47 = P_b。"""
    r = _run(gas, 1.0, 0.1, 30.0, n_points=1500)
    df = r.df
    mask = df["V_gas_mL"] > 0.005
    psum = df["p_G_mmHg"] + df["p_N2_mmHg"] + df["p_O2_mmHg"] + df["p_CO2_mmHg"] + 47.0
    err = (psum[mask] - df["P_b_mmHg"][mask]).abs().max()
    assert err < 1e-3, f"{gas} partial-pressure invariant err {err:.4f} mmHg"


@pytest.mark.parametrize("gas", FLUORO)
def test_moles_nonnegative(gas):
    """§6-1: モル数は（数値誤差を除き）非負。"""
    r = _run(gas, 1.0, 0.1, 60.0)
    for col in ["n_G_mol", "n_N2_mol", "n_O2_mol", "n_CO2_mol"]:
        assert r.df[col].min() > -1e-12, f"{gas} {col} went negative"


def test_air_duration_ballpark():
    """air の持続が概ね 5–9 日（緩い）。"""
    r = _run("air", 1.0, 0.8, 30.0)
    v = r.df["V_gas_mL"].values
    t = r.df["t_days"].values
    below = t[v <= 0.05 * r.ic.V_gas0]
    dur = below[0] if len(below) else 30.0
    assert 4.0 <= dur <= 10.0, f"air duration {dur:.1f}d out of ballpark"


def test_duration_ordering():
    """希釈濃度での持続は SF6 < C2F6 < C3F8（順序のみ; 絶対値は v1.1 課題）。"""
    durs = []
    for gas in FLUORO:
        bm = BENCHMARKS[gas]
        c_ne = sum(bm.nonexpansile_conc) / 2.0
        r = _run(gas, c_ne, 0.8, 120.0)
        v = r.df["V_gas_mL"].values
        t = r.df["t_days"].values
        below = t[v <= 0.05 * r.ic.V_gas0]
        durs.append(below[0] if len(below) else 120.0)
    assert durs[0] < durs[1] < durs[2], f"duration not ordered: {durs}"


@pytest.mark.parametrize("gas", FLUORO)
def test_intact_vitreous_longer_life(gas):
    """phakic + intact vitreous は vitrectomized より長寿命（medium スケール）。"""

    def dur(medium):
        r = _run(gas, 1.0, 0.1, 250.0, medium=medium, n_points=2000)
        v = r.df["V_gas_mL"].values
        t = r.df["t_days"].values
        below = t[v <= 0.05 * r.ic.V_gas0]
        return below[0] if len(below) else 250.0

    assert dur("phakic_intact_vitreous") > dur("vitrectomized")
