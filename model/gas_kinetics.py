"""v1 機構モデル: 多成分ガスの経界面拡散。ハンドオフ §2.1, §3, §4。

気泡内の各乾性ガス成分のモル数 n_i（i ∈ {G, N2, O2, CO2}）を状態変数とし、
分圧差で駆動される Fick/Henry 型の経界面拡散で時間発展させる:

    dn_i/dt = - Q_i · A(V_gas) · ( p_i − p_i_tissue )

水蒸気は常に飽和（47 mmHg）として代数的に保持し、独立状態にしない。
このため気泡の「圧縮に効く実効圧」は P_b ではなく (P_b − 47) になる。

設計の肝（§4）: Q_i の直接文献値は乏しいので、§7-A のベンチマーク
（膨張倍率・ピーク時刻・持続日数・非膨張濃度）に合うよう最小二乗で同定する
（validation/thompson_fit.py）。本ファイルはその同定結果を既定値として読み込む。
"""

import math
from dataclasses import dataclass

from units import P_H2O_SAT_MMHG

# --- 組織側ガス分圧 [mmHg]（§3, 近似・要文献確認） -----------------------
# フッ素ガスは生体に存在しない → 終点 0。総和 < 760 の「不飽和」が最終吸収の駆動力。
TISSUE_P = {
    "G": 0.0,  # フッ素ガス（SF6/C2F6/C3F8）。air では未使用
    "N2": 570.0,  # 体内 N2 はほぼ大気平衡
    "O2": 45.0,  # 硝子体は低酸素（40–55）
    "CO2": 47.0,  # 45–50
}

DRY_SPECIES = ("G", "N2", "O2", "CO2")

# 術式（media）による拡散速度スケール。vitrectomized=1 を基準、
# phakic + intact vitreous は半減期 2–3 倍 → 速度 ~0.4 倍（§4, Thompson）。
MEDIUM_RATE_SCALE = {
    "vitrectomized": 1.0,
    "phakic_intact_vitreous": 0.4,
}


def bubble_area_mm2(V_gas_mL: float) -> float:
    """気泡表面積 A [mm²]。球近似 A = 4π(3V/4π)^(2/3)。

    V_gas は mL（=1000 mm³）。極小・負体積は 0 でクリップ。
    自由球モデル（v1）。v1.1 の球冠モデルは bubble_area_cap_mm2 を使う。
    """
    if V_gas_mL <= 0.0:
        return 0.0
    V_mm3 = V_gas_mL * 1000.0
    r = (3.0 * V_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    return 4.0 * math.pi * r * r


def cavity_radius_mm(V_vit_mL: float) -> float:
    """硝子体腔を球とみなしたときの半径 R [mm]。V_vit = (4/3)πR³。"""
    V_mm3 = V_vit_mL * 1000.0
    return (3.0 * V_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def _cap_height_from_volume(V_g_mm3: float, R: float) -> float:
    """球冠体積 V_g = (π/3)·a²·(3R − a) を満たす冠高 a [mm] を逆算（0≤a≤2R）。

    a について単調増加なので二分法で解く。
    """
    if V_g_mm3 <= 0.0:
        return 0.0
    V_full = (4.0 / 3.0) * math.pi * R**3
    if V_g_mm3 >= V_full:
        return 2.0 * R

    # V(a) = πR·a² − (π/3)·a³,  V'(a) = 2πR·a − π·a²。Newton 法（単調なので安定）。
    # 初期値: 小 a 近似 V≈πR·a² から a0 = sqrt(Vg/(πR))。
    a = min(math.sqrt(V_g_mm3 / (math.pi * R)), 2.0 * R)
    for _ in range(8):
        f = math.pi * R * a * a - (math.pi / 3.0) * a**3 - V_g_mm3
        fp = 2.0 * math.pi * R * a - math.pi * a * a
        if fp <= 1e-12:
            break
        a_new = a - f / fp
        # 区間 [0, 2R] に拘束
        a = min(max(a_new, 0.0), 2.0 * R)
    return a


def bubble_area_cap_mm2(V_gas_mL: float, V_vit_mL: float) -> float:
    """v1.1 球冠界面モデルの気泡表面積 A [mm²]。

    気泡は浮力で硝子体腔（半径 R の球）の上部に「球冠」として溜まる。
    気泡表面 = 腔壁に接する曲面（gas–retina）＋ 液面の平円板（gas–fluid メニスカス）。
        A_wall = 2πR·a,  A_meniscus = π·a·(2R − a),  a=冠高
    充填率が高いほどメニスカスが縮み、表面/体積比が小さくなる（→長寿命）。
    満充填(a=2R)で A=4πR²（球全体）、メニスカス=0 に収束。
    """
    if V_gas_mL <= 0.0:
        return 0.0
    R = cavity_radius_mm(V_vit_mL)
    V_g_mm3 = V_gas_mL * 1000.0
    a = _cap_height_from_volume(V_g_mm3, R)
    A_wall = 2.0 * math.pi * R * a
    A_meniscus = math.pi * a * (2.0 * R - a)
    return A_wall + A_meniscus


# --- v1.2 球冠→球 移行モデル ------------------------------------------
# Hall et al. 2017（AIChE J）の指摘: 長時間で気泡は球冠→自由球へ移行し、血流のある
# 組織との接触面積が減って吸収が急減速する（彼らは未実装）。これを再現する。
#
# 浮力 vs 表面張力の競合は Bond 数 Bo = Δρ·g·r² / σ で特徴づけられ、毛細管長
# r_cap = sqrt(σ/(Δρ·g)) より大きい気泡は浮力で網膜に押し付けられ球冠（広い組織接触）、
# 小さい気泡は表面張力で球形（液面接触が主で吸収が遅い）になる。
SURFACE_TENSION_N_M = 0.07  # 気液界面張力（おおよそ）
DELTA_RHO_KG_M3 = 1000.0  # ガス vs 眼内液 の密度差
G_M_S2 = 9.8

# 実効的な遷移半径・液面側の相対吸収効率の既定（較正で上書き可能, §較正）。
R_CRIT_MM_DEFAULT = 3.0  # 物理的毛細管長は ~2.7mm。網膜の湾曲等で実効値は較正。
FLUID_EFFICIENCY_DEFAULT = 0.5  # 液面接触の吸収効率（網膜接触=1 に対する比, φ<1）
TRANSITION_SHARPNESS = 4.0


def capillary_length_mm() -> float:
    """毛細管長 r_cap = sqrt(σ/(Δρ·g)) [mm]（物理的な遷移スケールの目安, ~2.7mm）。"""
    return math.sqrt(SURFACE_TENSION_N_M / (DELTA_RHO_KG_M3 * G_M_S2)) * 1000.0


def free_sphere_radius_mm(V_gas_mL: float) -> float:
    if V_gas_mL <= 0.0:
        return 0.0
    V_mm3 = V_gas_mL * 1000.0
    return (3.0 * V_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def _cap_weight(V_gas_mL: float, r_crit_mm: float) -> float:
    """球冠の重み w: 大気泡 →1（球冠）, 小気泡 →0（自由球）。Bond 数（半径比）で遷移。"""
    r = free_sphere_radius_mm(V_gas_mL)
    if r <= 0.0:
        return 0.0
    x = (r / r_crit_mm) ** TRANSITION_SHARPNESS
    return x / (1.0 + x)


def bubble_area_cap_sphere_mm2(
    V_gas_mL: float,
    V_vit_mL: float,
    r_crit_mm: float = R_CRIT_MM_DEFAULT,
    fluid_eff: float = FLUID_EFFICIENCY_DEFAULT,
) -> float:
    """v1.2 実効拡散面積 [mm²]。球冠（網膜接触, 効率1）↔ 自由球（液面接触, 効率φ）を遷移。

        A_eff = w·A_cap + (1−w)·φ·A_sphere
    大気泡 (w→1): A_cap（広い網膜接触）。小気泡 (w→0): φ·A_sphere（液面主体で遅い）。
    A_cap ≥ A_sphere かつ φ<1 なので、小気泡ほど実効面積が小さく吸収が遅延 → 長期残存。
    """
    if V_gas_mL <= 0.0:
        return 0.0
    A_cap = bubble_area_cap_mm2(V_gas_mL, V_vit_mL)
    A_sph = bubble_area_mm2(V_gas_mL)
    w = _cap_weight(V_gas_mL, r_crit_mm)
    return w * A_cap + (1.0 - w) * fluid_eff * A_sph


def bubble_area(
    V_gas_mL: float,
    V_vit_mL: float,
    interface: str = "cap_sphere",
    r_crit_mm: float = R_CRIT_MM_DEFAULT,
    fluid_eff: float = FLUID_EFFICIENCY_DEFAULT,
) -> float:
    """界面積ディスパッチ。
    interface = "cap_sphere"(v1.2) / "cap"(v1.1) / "sphere"(v1)。
    """
    if interface == "sphere":
        return bubble_area_mm2(V_gas_mL)
    if interface == "cap":
        return bubble_area_cap_mm2(V_gas_mL, V_vit_mL)
    return bubble_area_cap_sphere_mm2(V_gas_mL, V_vit_mL, r_crit_mm, fluid_eff)


def dry_total(n: dict) -> float:
    return sum(n[s] for s in DRY_SPECIES)


def gas_volume_mL(n_dry_total: float, P_b: float, RT: float) -> float:
    """乾性ガス総モルと絶対圧から気泡体積 [mL]。

    水蒸気飽和を保つため V = N_dry·R·T/(P_b − 47)。
    """
    return n_dry_total * RT / (P_b - P_H2O_SAT_MMHG)


def partial_pressures(n: dict, P_b: float) -> dict:
    """各乾性ガスの分圧 [mmHg]。Σ p_i(dry) = P_b − 47 を満たす。

    p_i = n_i / N_dry · (P_b − 47)。
    """
    N_dry = dry_total(n)
    if N_dry <= 0.0:
        return {s: 0.0 for s in DRY_SPECIES}
    scale = (P_b - P_H2O_SAT_MMHG) / N_dry
    return {s: n[s] * scale for s in DRY_SPECIES}


@dataclass
class DiffusionParams:
    """物質移動係数 Q_i [mol·mm⁻²·min⁻¹·mmHg⁻¹]。

    Q_G はガス種ごと（フッ素ガスの溶解・拡散の遅さがガス寿命を決める）。
    Q_N2/Q_O2/Q_CO2 は生体ガスで全ガス共通。
    """

    Q_G: float
    Q_N2: float
    Q_O2: float
    Q_CO2: float

    def scaled(self, medium: str) -> "DiffusionParams":
        s = MEDIUM_RATE_SCALE[medium]
        return DiffusionParams(
            self.Q_G * s, self.Q_N2 * s, self.Q_O2 * s, self.Q_CO2 * s
        )

    def as_species_dict(self) -> dict:
        return {"G": self.Q_G, "N2": self.Q_N2, "O2": self.Q_O2, "CO2": self.Q_CO2}


# v1.3 濃度（分圧）依存の foreign-gas 物質移動係数。
# Q_G_eff(p_G) = Q_G · (1 + β·p_G/P0)。β>0 で分圧が高いほど通りやすい
# （高分子膜の自由体積/可塑化理論: 浸透ガス濃度が高いほど拡散係数が増す, Fujita/Frisch）。
# 効果: 膨張相(p_G 高)では Q_G 実効大→ガスが多く漏れ膨張倍率↓。希釈・末期(p_G 低)では
# Q_G ~ Q_G0 のまま→持続を維持。これで膨張倍率と持続の結合を直接ほどく。
# 生体ガス(N2/O2/CO2)は線形のまま。P0 は参照圧(=大気圧)。
P0_CONC_REF_MMHG = 760.0


def dn_dt(
    n: dict, P_b: float, Q: DiffusionParams, A_mm2: float, beta_G: float = 0.0
) -> dict:
    """各乾性ガスのモル変化率 [mol/min]。dn_i/dt = −Q_i·A·(p_i − p_i_tissue)。

    foreign gas G のみ濃度依存: Q_G → Q_G·(1 + β·p_G/P0)。β=0 で従来（定数）。
    界面積 A は呼び出し側で算出して渡す（球/球冠の選択を simulator が持つ）。
    """
    p = partial_pressures(n, P_b)
    Qd = Q.as_species_dict()
    out = {}
    for s in DRY_SPECIES:
        q = Qd[s]
        if s == "G" and beta_G != 0.0:
            q = q * (1.0 + beta_G * max(p[s], 0.0) / P0_CONC_REF_MMHG)
        out[s] = -q * A_mm2 * (p[s] - TISSUE_P[s])
    return out
