"""房水動態（Goldmann 式）。ハンドオフ §2.3。

    dV_fluid/dt = F_prod - F_out
    F_out       = C_tm * max(IOP - P_ev, 0) + F_uveo

V_fluid は「排出可能な残存流体量（残存BSS＋房水）」[mL]。0 を床（floor）とし、
枯渇したら流出を止めて流入のみとする（負の体積を許さない）。
内部単位: 流量は mL/min、圧は mmHg、体積は mL。
"""

from dataclasses import dataclass

from units import ul_per_min_to_ml_per_min


@dataclass
class AqueousParams:
    """房水動態パラメータ。文献既定値はハンドオフ §4。"""

    F_prod_ul_min: float = 2.4  # 房水産生 [µL/min]（2.0–2.5）
    C_tm_ul_min_mmHg: float = 0.26  # 線維柱帯流出能 [µL/min/mmHg]（0.22–0.30）
    F_uveo_ul_min: float = 0.2  # ぶどう膜強膜流出 [µL/min]（0–0.5, 圧非依存近似）
    P_ev_mmHg: float = 8.5  # 上強膜静脈圧 [mmHg]（8–9）

    # 内部 mL/min 換算（プロパティで都度変換し単位の取り違えを防ぐ）
    @property
    def F_prod(self) -> float:
        return ul_per_min_to_ml_per_min(self.F_prod_ul_min)

    @property
    def C_tm(self) -> float:
        return ul_per_min_to_ml_per_min(self.C_tm_ul_min_mmHg)

    @property
    def F_uveo(self) -> float:
        return ul_per_min_to_ml_per_min(self.F_uveo_ul_min)


def equilibrium_IOP(p: AqueousParams) -> float:
    """ガス外乱がないときの定常 IOP [mmHg]（F_prod = F_out を解く）。

    F_prod = C_tm·(IOP - P_ev) + F_uveo  ⇒  IOP = P_ev + (F_prod - F_uveo)/C_tm。
    シミュレーションの自己無撞着な baseline（初期 IOP の既定）に使う。
    """
    return p.P_ev_mmHg + (p.F_prod - p.F_uveo) / p.C_tm


def outflow(IOP_mmHg: float, p: AqueousParams) -> float:
    """房水流出 F_out [mL/min]。IOP が P_ev 以下なら線維柱帯流出はゼロ。"""
    pressure_driven = p.C_tm * max(IOP_mmHg - p.P_ev_mmHg, 0.0)
    return pressure_driven + p.F_uveo


# 流出が枯渇床に近づくと滑らかに 0 へ向かわせるスケール幅 [mL]。
# 硬い if 分岐（V_fluid<=0 で打ち切り）は不連続を生み、過充填時に solve_ivp が
# 床付近で数値振動（chatter）してハングする。連続なランプで回避する。
V_FLOOR_RAMP_mL = 0.01


def _floor_ramp(V_fluid_mL: float) -> float:
    """V_fluid→0 で 0、V_fluid≫床幅 で 1 に滑らかに飽和するランプ係数。"""
    if V_fluid_mL <= 0.0:
        return 0.0
    return min(V_fluid_mL / V_FLOOR_RAMP_mL, 1.0)


def dV_fluid_dt(IOP_mmHg: float, V_fluid_mL: float, p: AqueousParams) -> float:
    """残存流体量の変化率 [mL/min]。

    枯渇床: 流出は「逃がせる流体があるぶんだけ」起こる。V_fluid が床幅に近づくと
    流出を連続的に減衰させ、枯渇点では流入（産生）のみが残る。これにより
    V_fluid は負へ行かず、不連続による数値振動も生じない。
    """
    f_out = outflow(IOP_mmHg, p) * _floor_ramp(V_fluid_mL)
    return p.F_prod - f_out
