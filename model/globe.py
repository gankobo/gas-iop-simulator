"""眼球の圧–体積関係（眼剛性 / コンプライアンス）。ハンドオフ §2.4。

Friedenwald の眼剛性式:
    log10(IOP / IOP_ref) = K * (V_globe - V_ref)

これを V について解くと眼球容積 V_globe(IOP) が得られ、IOP で微分すると
眼コンプライアンス（単位圧あたりの容積変化）が得られる:
    C_oc(IOP) = dV_globe/dIOP = 1 / (ln(10) * K * IOP)

高 IOP ほど C_oc が小さく（眼が硬く）なり、同じ体積増で IOP が急上昇する挙動を表す。
"""

import math

# Friedenwald 係数 K の既定値（1/µL → 1/mL に直すと 1000 倍）。
# ハンドオフ §4: K = 0.0125–0.025 [1/µL]。中央付近 0.0184 [1/µL] を既定とする。
# 内部体積単位は mL なので 1/mL に変換: K[1/mL] = K[1/µL] * 1000。
K_DEFAULT_PER_UL = 0.0184
K_DEFAULT_PER_ML = K_DEFAULT_PER_UL * 1000.0  # ≈ 18.4 [1/mL]

LN10 = math.log(10.0)


def ocular_compliance(IOP_mmHg: float, K_per_mL: float = K_DEFAULT_PER_ML) -> float:
    """眼コンプライアンス C_oc(IOP) = dV_globe/dIOP [mL/mmHg]。

    C_oc = 1 / (ln(10) * K * IOP)。IOP は gauge ではなく実効的な眼内圧として扱う。
    IOP→0 で発散するため下限でクリップする（低圧域での数値安定化）。
    """
    iop = max(IOP_mmHg, 1.0)  # 1 mmHg 未満では剛性式は意味を持たない → クリップ
    return 1.0 / (LN10 * K_per_mL * iop)


def globe_volume_change(
    IOP_mmHg: float, IOP_ref_mmHg: float, K_per_mL: float = K_DEFAULT_PER_ML
) -> float:
    """基準圧 IOP_ref からの眼球容積の増分 ΔV_globe [mL]。

    log10(IOP/IOP_ref) = K * ΔV_globe  ⇒  ΔV_globe = log10(IOP/IOP_ref) / K。
    体積保存の初期条件合わせ・検証用。
    """
    iop = max(IOP_mmHg, 1e-6)
    iop_ref = max(IOP_ref_mmHg, 1e-6)
    return math.log10(iop / iop_ref) / K_per_mL
