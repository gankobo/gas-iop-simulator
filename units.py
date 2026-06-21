"""単位変換の単一窓口（ハンドオフ §4 の「単位の罠」対策）。

このプロジェクトの内部単位を以下に統一する。全モジュールはここの定数・関数を通す。
  - 圧力 : mmHg
  - 体積 : mL
  - 時間 : min（分）
  - 物質量: mol
  - 温度 : K

理想気体則・PV 仕事は「mmHg·mL 単位の気体定数 R_mmHg_mL」を1つ定義して扱う。
これにより SI への往復変換を各所に散らさずに済む。
"""

# --- 物理定数 -------------------------------------------------------------
PA_PER_MMHG = 133.322  # 1 mmHg = 133.322 Pa
M3_PER_ML = 1.0e-6  # 1 mL  = 1e-6 m^3
R_SI = 8.314462618  # J/(mol·K) = Pa·m^3/(mol·K)

# mmHg·mL 単位の気体定数:
#   R_SI [Pa·m^3] を (mmHg, mL) に変換する。
#   R_mmHg_mL = R_SI / (PA_PER_MMHG * M3_PER_ML)
R_MMHG_ML = R_SI / (PA_PER_MMHG * M3_PER_ML)  # ≈ 62363.6  mmHg·mL/(mol·K)

# --- 生理・環境の既定定数 -------------------------------------------------
T_BODY_K = 310.15  # 体温 37℃
P_ATM_MMHG = 760.0  # 標準大気圧（高地テストで可変）
P_H2O_SAT_MMHG = 47.0  # 37℃ 飽和水蒸気圧

# --- 時間変換ヘルパ -------------------------------------------------------
MIN_PER_HOUR = 60.0
MIN_PER_DAY = 1440.0


def hours_to_min(h: float) -> float:
    return h * MIN_PER_HOUR


def days_to_min(d: float) -> float:
    return d * MIN_PER_DAY


def min_to_hours(m: float) -> float:
    return m / MIN_PER_HOUR


def min_to_days(m: float) -> float:
    return m / MIN_PER_DAY


# --- 房水流量の単位（µL/min → mL/min） -----------------------------------
UL_PER_ML = 1000.0


def ul_per_min_to_ml_per_min(q_ul: float) -> float:
    """房水産生・流出能は文献値が µL 単位。内部 mL に統一する。"""
    return q_ul / UL_PER_ML


# --- 理想気体ヘルパ（内部単位で完結） -------------------------------------
def ideal_gas_volume(n_mol: float, p_abs_mmHg: float, T_K: float = T_BODY_K) -> float:
    """V[mL] = nRT/P （P は絶対圧 mmHg）。"""
    return n_mol * R_MMHG_ML * T_K / p_abs_mmHg


def ideal_gas_moles(V_mL: float, p_abs_mmHg: float, T_K: float = T_BODY_K) -> float:
    """n[mol] = PV/RT。"""
    return p_abs_mmHg * V_mL / (R_MMHG_ML * T_K)
