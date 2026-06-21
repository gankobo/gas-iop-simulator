"""入力変換: 眼軸長・ガス条件 → 初期条件。ハンドオフ §5。

注: パッケージ名は `io` だと Python 標準ライブラリ `io` と衝突するため `io_eye` とする。
"""

from dataclasses import dataclass

from units import P_ATM_MMHG, P_H2O_SAT_MMHG, ideal_gas_moles

# --- 5.1 眼軸長 → 硝子体腔容積 -------------------------------------------
# 各モデルは V_vit[mL] = a * AL[mm] + b の形。


def vitreous_volume_from_AL(AL_mm: float, model: str = "linear") -> float:
    """眼軸長 AL[mm] から硝子体腔容積 V_vit[mL] を推定。

    model:
      - "linear"      : phakic/pseudophakic 実測コホート  V = 0.37·AL − 4.16
      - "mri_emmetrope": MRI 正視眼                       V = 0.458·AL − 6.33
      - "mri_myopia"  : MRI 病的近視                       V = 0.546·AL − 6.98
    """
    coeffs = {
        "linear": (0.37, -4.16),
        "mri_emmetrope": (0.458, -6.33),
        "mri_myopia": (0.546, -6.98),
    }
    if model not in coeffs:
        raise ValueError(f"unknown AL→V model: {model!r}; choose {list(coeffs)}")
    a, b = coeffs[model]
    V = a * AL_mm + b
    if V <= 0:
        raise ValueError(f"non-physical vitreous volume {V:.2f} mL for AL={AL_mm} mm")
    return V


# --- ガス物性: 100% 純ガスの最大膨張倍率（経験式 v0 用; §7-A） -----------
# Lincoff 1980 等の原典値。air は膨張しない。
GAS_MAX_EXPANSION = {
    "air": 1.0,
    "SF6": 2.0,  # 約 2–2.5 倍（保守的に 2.0）
    "C2F6": 3.3,  # Lincoff 1980: 3.3 倍
    "C3F8": 4.0,  # 約 4 倍
}

# 非膨張（等膨張）濃度の臨床慣用値（容積分率）。設定で切替可能（benchmarks 参照）。
NONEXPANSILE_CONC_CLINICAL = {
    "air": None,
    "SF6": 0.20,
    "C2F6": 0.16,
    "C3F8": 0.13,
}


@dataclass
class InitialConditions:
    """シミュレーションの初期状態（v0/v1 共通の入口）。"""

    V_vit: float  # 硝子体腔容積 [mL]
    V_gas0: float  # 初期気泡体積 [mL]
    V_fluid0: float  # 初期残存流体量 [mL]
    N_total0: float  # 初期気泡総モル数 [mol]
    n_fluoro0: float  # フッ素ガス初期モル数 [mol]
    n_N2_0: float  # 気泡内 N2 初期モル数 [mol]
    n_O2_0: float  # 気泡内 O2 初期モル数 [mol]
    n_H2O0: float  # 気泡内 水蒸気モル数 [mol]
    IOP0: float  # 初期 IOP [mmHg]
    gas: str
    c: float  # ガス濃度（容積分率）
    f: float  # 置換率（fill fraction）


def build_initial_conditions(
    *,
    V_vit: float,
    gas: str,
    c: float,
    f: float,
    IOP0: float = 15.0,
    P_atm: float = P_ATM_MMHG,
) -> InitialConditions:
    """ハンドオフ §5.2 の手順で初期気泡モル数を組む。

    濃度 c の定義: 乾燥1atm基準の容積分率（§9 で確定）。
    """
    if gas not in GAS_MAX_EXPANSION:
        raise ValueError(f"unknown gas {gas!r}; choose {list(GAS_MAX_EXPANSION)}")
    if not (0.0 < f <= 1.0):
        raise ValueError(f"fill fraction f must be in (0,1]; got {f}")
    if not (0.0 < c <= 1.0):
        raise ValueError(f"concentration c must be in (0,1]; got {c}")

    # 1. 初期気泡体積
    V_gas0 = f * V_vit
    # 2. 初期絶対圧
    P_b0 = P_atm + IOP0
    # 3. 全モル（湿性: 水蒸気込み）
    N_total0 = ideal_gas_moles(V_gas0, P_b0)
    # 4. 水蒸気が 47/P_b0 を占有 → 乾性ガスは残り
    n_H2O0 = (P_H2O_SAT_MMHG / P_b0) * N_total0
    n_dry = N_total0 - n_H2O0
    # 5. 乾性ガスの内訳: フッ素ガス分率 = c、空気部分 (1−c) を N2:O2 ≈ 0.79:0.21
    n_fluoro0 = c * n_dry
    n_air = (1.0 - c) * n_dry
    n_N2_0 = 0.79 * n_air
    n_O2_0 = 0.21 * n_air
    # air の場合: c は「空気の容積分率」ではなく、純空気注入なら c=... の扱いに注意。
    # 本モデルでは gas='air' のとき n_fluoro0=0 とし、乾性ガス全量を空気組成にする。
    if gas == "air":
        n_fluoro0 = 0.0
        n_air = n_dry
        n_N2_0 = 0.79 * n_air
        n_O2_0 = 0.21 * n_air
    # 6. 残存流体量（v0: 硝子体腔のみ）
    V_fluid0 = V_vit - V_gas0

    return InitialConditions(
        V_vit=V_vit,
        V_gas0=V_gas0,
        V_fluid0=V_fluid0,
        N_total0=N_total0,
        n_fluoro0=n_fluoro0,
        n_N2_0=n_N2_0,
        n_O2_0=n_O2_0,
        n_H2O0=n_H2O0,
        IOP0=IOP0,
        gas=gas,
        c=c,
        f=f,
    )
