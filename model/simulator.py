"""統合シミュレータ。ハンドオフ §2.5。

v0 では経験式ガスモデル（gas_empirical）が気泡の「物質的体積トレンド」を与え、
房水動態（aqueous）と眼剛性（globe）と連立して IOP(t) を解く。

§2.5 の IOP-ODE:
    dIOP/dt = [ dV_fluid/dt + (R·T/P_b)·dN_total/dt ]
              / [ C_oc(IOP) + V_gas / P_b ]

v0 でのモル数の扱い:
  経験式 V_gas_emp(t) を「基準圧 P_ref = P_atm + IOP0 における体積」と解釈し、
      N_total(t)   = V_gas_emp(t) · P_ref / (R·T)
      dN_total/dt  = dV_gas_emp/dt · P_ref / (R·T)
  とする。現在圧 P_b での実気泡体積は
      V_gas_actual = N_total · R·T / P_b = V_gas_emp · P_ref / P_b
  となり、IOP 上昇で気泡が圧縮される結合（双方向）が自然に入る。
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from units import (
    P_ATM_MMHG,
    R_MMHG_ML,
    T_BODY_K,
    days_to_min,
    min_to_days,
    min_to_hours,
)
from model.aqueous import AqueousParams, dV_fluid_dt, equilibrium_IOP, outflow
from model.globe import K_DEFAULT_PER_ML, ocular_compliance
from model.gas_empirical import (
    EmpiricalGasParams,
    build_gas_params,
    dV_gas_dt,
    V_gas as V_gas_emp_fn,
)
from model.gas_kinetics import (
    DRY_SPECIES,
    DiffusionParams,
    bubble_area,
    dn_dt,
    dry_total,
    gas_volume_mL,
    partial_pressures,
)
from model.calibrated_params import (
    get_calibrated_Q,
    get_conc_dependence,
    get_interface_params,
)
from io_eye.eye_geometry import InitialConditions, build_initial_conditions


@dataclass
class SimConfig:
    """シミュレーション設定。"""

    gas: str
    c: float  # ガス濃度（容積分率）
    f: float  # 置換率（fill fraction）
    V_vit: float  # 硝子体腔容積 [mL]
    IOP0: float | None = None  # 初期 IOP [mmHg]。None なら房水平衡 IOP を使う
    medium: str = "vitrectomized"  # or "phakic_intact_vitreous"
    P_atm: float = P_ATM_MMHG  # 大気圧（高地テストで可変）
    K_per_mL: float = K_DEFAULT_PER_ML
    aqueous: AqueousParams = field(default_factory=AqueousParams)
    t_end_days: float = 90.0  # シミュレーション期間
    n_points: int = 1500  # 出力サンプル数
    mechanism: str = "empirical"  # "empirical"(v0) or "diffusion"(v1/v1.1/v1.2)
    diffusion: DiffusionParams | None = None  # None なら較正済み既定値を使う
    interface: str = "cap_sphere"  # "cap_sphere"(v1.2) / "cap"(v1.1) / "sphere"(v1)
    r_crit_mm: float | None = None       # 球冠→球 遷移半径。None なら較正/既定値
    fluid_efficiency: float | None = None  # 液面の相対吸収効率 φ。None なら較正/既定値
    beta_G: float | None = None  # foreign-gas 濃度依存係数(v1.3)。None なら較正/既定値


# 警告閾値: これを超える IOP は「v0 剛性外挿の信頼域外（破局的過充填）」とみなす。
# 実眼ではこの域で網膜灌流停止・ガス放出・組織損傷が起こり、絶対値は当てにならない。
IOP_NONPHYSICAL_MMHG = 60.0


@dataclass
class SimResult:
    df: pd.DataFrame
    ic: InitialConditions
    gas_params: EmpiricalGasParams | None
    config: SimConfig
    IOP0: float = 0.0  # 実際に用いた初期 IOP
    warnings: list = field(default_factory=list)

    @property
    def peak_IOP(self) -> float:
        return float(self.df["IOP_mmHg"].max())

    @property
    def peak_IOP_time_h(self) -> float:
        i = int(self.df["IOP_mmHg"].values.argmax())
        return float(self.df["t_hours"].iloc[i])

    @property
    def max_fill_fraction(self) -> float:
        return float(self.df["fill_fraction"].max())


def _build_warnings(df: pd.DataFrame) -> list:
    warnings = []
    peak = float(df["IOP_mmHg"].max())
    max_fill = float(df["fill_fraction"].max())
    if max_fill > 1.001:
        warnings.append(
            f"硝子体腔を超える過充填（最大充填 {max_fill * 100:.0f}%）。房水が枯渇し "
            "IOP が急騰します。臨床的には破局的状況で、緊急のガス抜き等が必要な領域です。"
        )
    if peak > IOP_NONPHYSICAL_MMHG:
        warnings.append(
            f"ピーク IOP {peak:.0f} mmHg は剛性モデルの信頼域（〜{IOP_NONPHYSICAL_MMHG:.0f} "
            "mmHg）を超えています。実眼では網膜灌流停止・組織損傷が起こる領域で、"
            "絶対値そのものは参考程度に留めてください（『危険な高 IOP になる』という"
            "定性的結論は妥当）。"
        )
    return warnings


def _solve(rhs, y0, cfg) -> tuple:
    t_end = days_to_min(cfg.t_end_days)
    t_eval = np.linspace(0.0, t_end, cfg.n_points)
    sol = solve_ivp(
        rhs, (0.0, t_end), y0,
        method="LSODA", t_eval=t_eval,
        rtol=1e-7, atol=1e-10, max_step=t_end / 200,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y


def simulate(cfg: SimConfig) -> SimResult:
    if cfg.mechanism == "diffusion":
        return _simulate_diffusion(cfg)
    return _simulate_empirical(cfg)


def _simulate_empirical(cfg: SimConfig) -> SimResult:
    """v0: 経験式ガスモデル（§7-A）を体積外乱として IOP-ODE を解く。"""
    aq = cfg.aqueous
    IOP0 = cfg.IOP0 if cfg.IOP0 is not None else equilibrium_IOP(aq)

    ic = build_initial_conditions(
        V_vit=cfg.V_vit, gas=cfg.gas, c=cfg.c, f=cfg.f, IOP0=IOP0, P_atm=cfg.P_atm,
    )
    gp = build_gas_params(cfg.gas, cfg.c, cfg.medium)

    P_ref = cfg.P_atm + IOP0  # 経験式体積の基準圧
    RT = R_MMHG_ML * T_BODY_K  # noqa: F841 (対称性のため明示)

    def rhs(t, y):
        V_fluid, IOP = y
        IOP_eff = max(IOP, 0.0)
        P_b = cfg.P_atm + IOP_eff
        Vge = V_gas_emp_fn(t, ic.V_gas0, gp)
        dVge = dV_gas_dt(t, ic.V_gas0, gp)
        V_gas_actual = Vge * P_ref / P_b
        gas_vol_rate = dVge * P_ref / P_b
        dVf = dV_fluid_dt(IOP_eff, V_fluid, aq)
        C_oc = ocular_compliance(IOP_eff, cfg.K_per_mL)
        denom = C_oc + V_gas_actual / P_b
        dIOP = (dVf + gas_vol_rate) / denom
        return [dVf, dIOP]

    t, y = _solve(rhs, [ic.V_fluid0, IOP0], cfg)
    V_fluid, IOP = y[0], y[1]
    P_b = cfg.P_atm + np.maximum(IOP, 0.0)
    Vge = np.array([V_gas_emp_fn(ti, ic.V_gas0, gp) for ti in t])
    V_gas_actual = Vge * P_ref / P_b
    F_out = np.array([outflow(max(ip, 0.0), aq) for ip in IOP])
    df = pd.DataFrame({
        "t_min": t, "t_hours": min_to_hours(t), "t_days": min_to_days(t),
        "IOP_mmHg": IOP, "V_gas_mL": V_gas_actual, "V_gas_ref_mL": Vge,
        "V_fluid_mL": V_fluid, "P_b_mmHg": P_b,
        "fill_fraction": V_gas_actual / cfg.V_vit, "F_out_uL_min": F_out * 1000.0,
    })
    return SimResult(df=df, ic=ic, gas_params=gp, config=cfg,
                     IOP0=IOP0, warnings=_build_warnings(df))


def _simulate_diffusion(cfg: SimConfig) -> SimResult:
    """v1: 多成分拡散の機構モデル。各乾性ガスのモル数を状態変数として解く。

    状態 y = [n_G, n_N2, n_O2, n_CO2, V_fluid, IOP]。
    IOP-ODE（水蒸気飽和を保つため実効圧 = P_b − 47）:
        dIOP/dt = [ dV_fluid/dt + R·T/(P_b−47)·dN_dry/dt ]
                  / [ C_oc(IOP) + V_gas/(P_b−47) ]
    """
    aq = cfg.aqueous
    IOP0 = cfg.IOP0 if cfg.IOP0 is not None else equilibrium_IOP(aq)
    dp = (cfg.diffusion or get_calibrated_Q(cfg.gas)).scaled(cfg.medium)
    # 球冠→球 遷移パラメータ（cfg 指定 > 較正値 > 既定）
    ifp = get_interface_params()
    r_crit = cfg.r_crit_mm if cfg.r_crit_mm is not None else ifp["r_crit_mm"]
    fluid_eff = (cfg.fluid_efficiency if cfg.fluid_efficiency is not None
                 else ifp["fluid_efficiency"])
    beta_G = cfg.beta_G if cfg.beta_G is not None else get_conc_dependence(cfg.gas)

    ic = build_initial_conditions(
        V_vit=cfg.V_vit, gas=cfg.gas, c=cfg.c, f=cfg.f, IOP0=IOP0, P_atm=cfg.P_atm,
    )
    RT = R_MMHG_ML * T_BODY_K
    P_H2O = 47.0

    # 初期モル数（CO2 は注入ガスに無いので 0）
    n0 = {
        "G": ic.n_fluoro0, "N2": ic.n_N2_0,
        "O2": ic.n_O2_0, "CO2": 0.0,
    }
    y0 = [n0["G"], n0["N2"], n0["O2"], n0["CO2"], ic.V_fluid0, IOP0]

    def rhs(t, y):
        nG, nN2, nO2, nCO2, V_fluid, IOP = y
        n = {"G": max(nG, 0.0), "N2": max(nN2, 0.0),
             "O2": max(nO2, 0.0), "CO2": max(nCO2, 0.0)}
        IOP_eff = max(IOP, 0.0)
        P_b = cfg.P_atm + IOP_eff
        N_dry = dry_total(n)
        V_gas = gas_volume_mL(N_dry, P_b, RT) if N_dry > 0 else 0.0

        A = bubble_area(V_gas, cfg.V_vit, cfg.interface, r_crit, fluid_eff)
        dn = dn_dt(n, P_b, dp, A, beta_G)
        dN_dry = sum(dn[s] for s in DRY_SPECIES)

        dVf = dV_fluid_dt(IOP_eff, V_fluid, aq)
        C_oc = ocular_compliance(IOP_eff, cfg.K_per_mL)
        denom = C_oc + V_gas / (P_b - P_H2O)
        dIOP = (dVf + RT / (P_b - P_H2O) * dN_dry) / denom
        return [dn["G"], dn["N2"], dn["O2"], dn["CO2"], dVf, dIOP]

    t, y = _solve(rhs, y0, cfg)
    nG, nN2, nO2, nCO2, V_fluid, IOP = y
    P_b = cfg.P_atm + np.maximum(IOP, 0.0)

    # 後処理: 各 step の体積・分圧を再構成
    V_gas = np.zeros_like(t)
    pG = np.zeros_like(t); pN2 = np.zeros_like(t)
    pO2 = np.zeros_like(t); pCO2 = np.zeros_like(t)
    for i in range(len(t)):
        n = {"G": max(nG[i], 0.0), "N2": max(nN2[i], 0.0),
             "O2": max(nO2[i], 0.0), "CO2": max(nCO2[i], 0.0)}
        N_dry = dry_total(n)
        V_gas[i] = gas_volume_mL(N_dry, P_b[i], RT) if N_dry > 0 else 0.0
        p = partial_pressures(n, P_b[i])
        pG[i], pN2[i], pO2[i], pCO2[i] = p["G"], p["N2"], p["O2"], p["CO2"]

    F_out = np.array([outflow(max(ip, 0.0), aq) for ip in IOP])
    # 物理的にモル数は非負。吸収末端の数値的な極小負値（〜1e-11）は 0 でクリップ。
    df = pd.DataFrame({
        "t_min": t, "t_hours": min_to_hours(t), "t_days": min_to_days(t),
        "IOP_mmHg": IOP, "V_gas_mL": V_gas,
        "V_fluid_mL": V_fluid, "P_b_mmHg": P_b,
        "fill_fraction": V_gas / cfg.V_vit, "F_out_uL_min": F_out * 1000.0,
        "p_G_mmHg": pG, "p_N2_mmHg": pN2, "p_O2_mmHg": pO2, "p_CO2_mmHg": pCO2,
        "n_G_mol": np.maximum(nG, 0.0), "n_N2_mol": np.maximum(nN2, 0.0),
        "n_O2_mol": np.maximum(nO2, 0.0), "n_CO2_mol": np.maximum(nCO2, 0.0),
    })
    return SimResult(df=df, ic=ic, gas_params=None, config=cfg,
                     IOP0=IOP0, warnings=_build_warnings(df))
