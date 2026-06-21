"""v0 経験式ガスモデル。ハンドオフ §8(v0) / §7-A。

気泡体積 V_gas(t) を、文献の経験パラメータ（膨張倍率・ピーク時刻・一次消失）から
直接生成する。機構（多成分拡散）は v1 で扱う。

形状（無次元 shape(t) = V_gas(t)/V_gas0）は区分連続:

    膨張相  0 ≤ t ≤ t_peak :  shape = 1 + (E-1) * s(t/t_peak)
            s(x) = (1 - e^{-a·x}) / (1 - e^{-a})            （0→1, 初期が最速）
    消失相  t > t_peak     :  shape = E * exp(-k_diss·(t - t_peak))   （一次反応）

設計意図（ハンドオフ §6/§7-A を満たす）:
  - ピーク膨張倍率 = E、ピーク時刻 = t_peak を厳密に再現（§7-A ベンチマーク直撃）
  - 立ち上がり s は初期が最速 →「膨張は最初の6–8時間が最速」に整合
  - 消失相は指数減衰（一次反応）→ t→∞ で完全吸収（§6-4）
  - air は E=1 → 膨張相が平坦、即消失（§6-5 非膨張）
  - 単峰（多峰振動なし, §6-6）

注: ピークで dshape/dt に角（不連続）が生じるが、値は連続で単峰性は保たれる。
v1 の機構モデルでは滑らかな3相が創発的に得られる。
"""

import math
from dataclasses import dataclass

from units import days_to_min, hours_to_min
from io_eye.eye_geometry import GAS_MAX_EXPANSION, NONEXPANSILE_CONC_CLINICAL

# --- 文献由来パラメータ（§7-A） -----------------------------------------
# ピーク到達時刻 [hour]（範囲の中央付近）。air はピークなし → t_peak=0（即消失）。
PEAK_TIME_HOURS = {
    "air": 0.0,
    "SF6": 36.0,  # 24–48 h
    "C2F6": 48.0,  # 36–60 h
    "C3F8": 84.0,  # 72–96 h
}

# 臨床（希釈）濃度での持続日数 [day]。dissolution の一次速度 k_diss の較正アンカー。
#   air: 5–7 日 / SF6(20%): ~14 日 / C2F6(16%): ~31 日 / C3F8(14%): ~67 日
DURATION_DAYS_CLINICAL = {
    "air": 6.0,
    "SF6": 14.0,
    "C2F6": 31.0,
    "C3F8": 67.0,
}

# 「持続」を気泡が初期体積の何割まで縮むまでとみなすか（消失判定）。
RESIDUAL_AT_DURATION = 0.05  # 5% 残存 = ほぼ消失

# 膨張相 s(x) の早さパラメータ（大きいほど初期集中）。
RISE_SHARPNESS = 2.5

# 術式（media）による速度スケール。vitrectomized を基準 1.0、
# phakic + intact vitreous は拡散が遅く半減期 2–3 倍長い（Thompson）→ 速度 ~0.4 倍。
# 速度が遅い = 消失が遅い（寿命長い）かつピーク到達も遅い。
MEDIUM_RATE_SCALE = {
    "vitrectomized": 1.0,
    "phakic_intact_vitreous": 0.4,
}


def expansion_factor(gas: str, c: float) -> float:
    """濃度 c におけるピーク膨張倍率 E。

    100%(c=1) で文献の最大膨張倍率、非膨張濃度 c_ne で 1.0（等容）に線形補間。
    c_ne 未満は v0 では 1.0 で床留め（即 dissolution）。
    """
    E_max = GAS_MAX_EXPANSION[gas]
    if gas == "air" or E_max <= 1.0:
        return 1.0
    c_ne = NONEXPANSILE_CONC_CLINICAL[gas]
    if c <= c_ne:
        return 1.0
    return 1.0 + (E_max - 1.0) * (c - c_ne) / (1.0 - c_ne)


@dataclass
class EmpiricalGasParams:
    """gas/c/medium から導いた形状パラメータ（分単位）。"""

    gas: str
    E: float  # ピーク膨張倍率
    t_peak_min: float  # ピーク到達時刻 [min]（air は 0）
    k_diss_per_min: float  # 一次消失速度 [1/min]
    a_rise: float = RISE_SHARPNESS

    @property
    def has_peak(self) -> bool:
        return self.t_peak_min > 0.0 and self.E > 1.0


def build_gas_params(
    gas: str, c: float, medium: str = "vitrectomized"
) -> EmpiricalGasParams:
    if gas not in GAS_MAX_EXPANSION:
        raise ValueError(f"unknown gas {gas!r}")
    if medium not in MEDIUM_RATE_SCALE:
        raise ValueError(f"unknown medium {medium!r}; choose {list(MEDIUM_RATE_SCALE)}")
    scale = MEDIUM_RATE_SCALE[medium]

    E = expansion_factor(gas, c)

    # ピーク時刻: 遅い media では到達も遅い（1/scale 倍）。
    t_peak = hours_to_min(PEAK_TIME_HOURS[gas]) / scale

    # k_diss: 消失相が D_min（持続）で残存 RESIDUAL に達するよう較正。
    #   E·exp(-k·(D - t_peak)) = RESIDUAL  →  k = ln(E/RESIDUAL)/(D - t_peak)
    D_min = days_to_min(DURATION_DAYS_CLINICAL[gas]) / scale  # intact は長寿命
    span = max(D_min - t_peak, days_to_min(0.5))  # 安全下限
    k_diss = math.log(E / RESIDUAL_AT_DURATION) / span

    return EmpiricalGasParams(gas=gas, E=E, t_peak_min=t_peak, k_diss_per_min=k_diss)


def _s(x: float, a: float) -> float:
    """膨張相の立ち上がり関数 s(x): [0,1]→[0,1]、初期が最速。"""
    return (1.0 - math.exp(-a * x)) / (1.0 - math.exp(-a))


def _ds(x: float, a: float) -> float:
    """s'(x)。"""
    return a * math.exp(-a * x) / (1.0 - math.exp(-a))


def shape(t_min: float, p: EmpiricalGasParams) -> float:
    """無次元体積形状 shape(t) = V_gas(t)/V_gas0。"""
    if t_min <= 0.0:
        return 1.0
    if p.t_peak_min > 0.0 and t_min <= p.t_peak_min:
        x = t_min / p.t_peak_min
        return 1.0 + (p.E - 1.0) * _s(x, p.a_rise)
    # 消失相
    return p.E * math.exp(-p.k_diss_per_min * (t_min - p.t_peak_min))


def dshape_dt(t_min: float, p: EmpiricalGasParams) -> float:
    """shape の時間微分 [1/min]（IOP-ODE の体積外乱項に使う）。"""
    if t_min < 0.0:
        return 0.0
    if p.t_peak_min > 0.0 and t_min <= p.t_peak_min:
        x = t_min / p.t_peak_min
        return (p.E - 1.0) * _ds(x, p.a_rise) / p.t_peak_min
    return (
        -p.k_diss_per_min * p.E * math.exp(-p.k_diss_per_min * (t_min - p.t_peak_min))
    )


def V_gas(t_min: float, V_gas0: float, p: EmpiricalGasParams) -> float:
    return V_gas0 * shape(t_min, p)


def dV_gas_dt(t_min: float, V_gas0: float, p: EmpiricalGasParams) -> float:
    return V_gas0 * dshape_dt(t_min, p)


def peak_info(V_gas0: float, p: EmpiricalGasParams):
    """ピーク（最大体積）の時刻[min]・倍率を返す。区分形なので解析的に既知。"""
    if p.has_peak:
        return p.t_peak_min, p.E
    return 0.0, 1.0
