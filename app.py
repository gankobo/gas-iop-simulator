"""眼内膨張性ガス IOP シミュレーター — Streamlit 対話UI。

起動: プロジェクトフォルダで
    streamlit run app.py
ブラウザが開き、スライダーで条件を変えると IOP(t) などが即座に更新される。
"""

import sys
import os

# プロジェクトルートを import パスに追加（streamlit run でも動くように）
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import streamlit as st

from model.aqueous import AqueousParams, equilibrium_IOP
from model.simulator import SimConfig, simulate
from io_eye.eye_geometry import (GAS_MAX_EXPANSION, NONEXPANSILE_CONC_CLINICAL,
                                 vitreous_volume_from_AL)
from validation.benchmarks import BENCHMARKS

st.set_page_config(page_title="眼内ガス IOP シミュレーター", layout="wide")

st.title("眼内膨張性ガス IOP シミュレーター")
st.caption(
    "硝子体手術 / pneumatic retinopexy で膨張性ガス（air・SF6・C2F6・C3F8）を入れたときの"
    "眼圧（IOP）の時間変化を、気液動態の物理に基づいて再現します。"
)
st.error(
    "⚠️ 教育・研究用のデモです（医療機器ではありません）。臨床判断・患者個別の予測には"
    "使用しないでください。特に過充填時の絶対 IOP 値は外挿で、数値より傾向の方が信頼できます。\n\n"
    "Educational/research demo — **not a medical device**. Do not use for clinical decisions "
    "or individual prediction."
)

# ============================ サイドバー: 入力 ============================
with st.sidebar:
    st.header("入力条件")

    mech_label = st.radio(
        "計算モデル",
        ["v0（経験式・高速）", "v1.2（多成分拡散・球冠→球移行）"],
        help="v0 は文献の経験式でガス体積を生成。v1.2 は気体の拡散物理から"
             "膨張・3相・吸収を創発的に計算。気泡は大きいうちは網膜に押し付けられた"
             "球冠、小さくなると球形へ移行する（吸収が遅れ長期残存）。各ガス分圧も出力。",
    )
    mechanism = "diffusion" if mech_label.startswith("v1") else "empirical"

    gas = st.selectbox("ガスの種類", ["air", "SF6", "C2F6", "C3F8"], index=1)

    # 濃度（air は純空気＝1.0 固定）
    if gas == "air":
        c = 1.0
        st.info("air は希釈しないため濃度は 100% 固定です。")
    else:
        c_ne = NONEXPANSILE_CONC_CLINICAL[gas]
        c = st.slider(
            "ガス濃度（容積分率）", min_value=0.05, max_value=1.0,
            value=float(c_ne), step=0.01,
            help=f"{gas} の非膨張（等膨張）濃度の目安は約 {c_ne*100:.0f}%。"
                 "これより高いと膨張、低いとほぼ等容で吸収。",
        )

    f = st.slider(
        "硝子体腔の置換率 f（fill fraction）", min_value=0.05, max_value=1.0,
        value=0.6, step=0.05,
        help="注入直後にガスが硝子体腔の何割を占めるか。",
    )

    medium = st.radio(
        "眼の状態（拡散速度に影響）",
        ["vitrectomized", "phakic_intact_vitreous"],
        format_func=lambda m: "硝子体切除眼（vitrectomized）"
        if m == "vitrectomized" else "硝子体・水晶体あり（吸収が約2.5倍遅い）",
    )

    st.divider()
    st.subheader("眼のサイズ")
    size_mode = st.radio("入力方法", ["眼軸長 AL から推定", "硝子体腔容積を直接入力"])
    if size_mode == "眼軸長 AL から推定":
        AL = st.slider("眼軸長 AL [mm]", 20.0, 32.0, 23.4, 0.1)
        al_model = st.selectbox(
            "推定式", ["linear", "mri_emmetrope", "mri_myopia"],
            format_func={
                "linear": "線形（実測コホート）",
                "mri_emmetrope": "MRI 正視眼",
                "mri_myopia": "MRI 病的近視",
            }.get,
        )
        V_vit = vitreous_volume_from_AL(AL, al_model)
        st.metric("推定 硝子体腔容積", f"{V_vit:.2f} mL")
    else:
        V_vit = st.slider("硝子体腔容積 [mL]", 2.5, 10.0, 4.5, 0.1)

    st.divider()
    st.subheader("初期眼圧（t=0）")
    use_eq = st.checkbox(
        "房水平衡値を自動で使う（既定）", value=True,
        help="チェックを外すと、t=0 の眼圧を手動で指定できます。",
    )
    if use_eq:
        IOP0 = None
        st.caption("房水動態から自動計算した平衡 IOP を初期値にします。")
    else:
        IOP0 = st.slider("初期 IOP [mmHg]", 5.0, 40.0, 15.0, 1.0)

    st.divider()
    with st.expander("詳細設定（上級者向け）"):
        K_uL = st.slider(
            "眼剛性 K [1/µL]（高いほど眼が硬い）", 0.0125, 0.025, 0.0184, 0.0005,
        )
        P_atm = st.slider(
            "大気圧 [mmHg]（高地・気圧変化テスト）", 500.0, 800.0, 760.0, 5.0,
        )
        F_prod = st.slider("房水産生 [µL/min]", 1.0, 3.5, 2.4, 0.1)
        C_tm = st.slider("線維柱帯流出能 [µL/min/mmHg]", 0.10, 0.40, 0.26, 0.01)
        t_end_days = st.slider("シミュレーション期間 [日]", 7, 200, 90, 1)

# ============================ シミュレーション ============================
aq = AqueousParams(F_prod_ul_min=F_prod, C_tm_ul_min_mmHg=C_tm)
cfg = SimConfig(
    gas=gas, c=c, f=f, V_vit=V_vit, IOP0=IOP0, medium=medium,
    P_atm=P_atm, K_per_mL=K_uL * 1000.0, aqueous=aq, t_end_days=float(t_end_days),
    mechanism=mechanism,
)
result = simulate(cfg)
df = result.df

# ============================ 警告 ============================
for w in result.warnings:
    st.warning("⚠️ " + w)

# ============================ サマリ指標 ============================
col1, col2, col3, col4 = st.columns(4)
col1.metric("ベースライン IOP", f"{result.IOP0:.1f} mmHg")
col2.metric("ピーク IOP", f"{result.peak_IOP:.1f} mmHg",
            delta=f"+{result.peak_IOP - result.IOP0:.1f}")
col3.metric("ピーク到達", f"{result.peak_IOP_time_h:.1f} 時間後")
col4.metric("初期気泡体積", f"{result.ic.V_gas0:.2f} mL")

# ============================ グラフ ============================
st.subheader("時間変化")
tab1, tab2 = st.tabs(["全期間", "最初の7日間（IOPスパイク拡大）"])


def _plot(frame):
    left, right = st.columns(2)
    with left:
        st.markdown("**眼圧 IOP [mmHg]**")
        st.line_chart(frame.set_index("t_days")[["IOP_mmHg"]])
    with right:
        st.markdown("**充填率 と 気泡体積 [mL]**")
        st.line_chart(frame.set_index("t_days")[["fill_fraction"]])
        st.line_chart(frame.set_index("t_days")[["V_gas_mL"]])


with tab1:
    _plot(df)
with tab2:
    _plot(df[df["t_days"] <= 7.0])

# v1（機構モデル）では気泡内の各ガス分圧を表示
if mechanism == "diffusion" and "p_G_mmHg" in df.columns:
    st.subheader("気泡内のガス分圧 [mmHg]（v1.2 機構モデル）")
    st.caption(
        "純ガス注入直後は窒素（N2）がほぼ 0。組織から N2 が流入して気泡が膨張し、"
        "N2 が組織値（約570）に達すると膨張が止まり、その後フッ素ガス（G）が"
        "ゆっくり抜けて気泡全体が吸収されます——この3相が物理から自然に出ています。"
    )
    pcols = {"p_G_mmHg": "フッ素ガス G", "p_N2_mmHg": "N2",
             "p_O2_mmHg": "O2", "p_CO2_mmHg": "CO2"}
    pp = df[df["V_gas_mL"] > 1e-3].rename(columns=pcols).set_index("t_days")
    st.line_chart(pp[list(pcols.values())])

# ============================ ベンチマーク照合 ============================
with st.expander("文献ベンチマーク（§7-A）との照合"):
    bm = BENCHMARKS[gas]
    rows = []
    if gas != "air":
        rows.append(("100%膨張倍率", f"{bm.expansion_100pct[0]}–{bm.expansion_100pct[1]} 倍"))
        rows.append(("ピーク到達", f"{bm.peak_time_h[0]:.0f}–{bm.peak_time_h[1]:.0f} 時間"))
        rows.append(("非膨張濃度",
                     f"{bm.nonexpansile_conc[0]*100:.0f}–{bm.nonexpansile_conc[1]*100:.0f}%"))
    rows.append(("希釈時の持続", f"{bm.duration_days[0]:.0f}–{bm.duration_days[1]:.0f} 日"))
    st.table(pd.DataFrame(rows, columns=["項目", "文献値"]))
    st.caption(
        f"最大膨張倍率（100%注入の文献値）: {GAS_MAX_EXPANSION[gas]} 倍 / "
        f"このモデルの平衡IOP: {equilibrium_IOP(aq):.1f} mmHg"
    )

# ============================ 生データ ============================
with st.expander("計算データ（CSV ダウンロード可）"):
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "CSV をダウンロード", df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"iop_sim_{gas}_c{c}_f{f}.csv", mime="text/csv",
    )

st.caption(
    "※ v0 は §7-A の文献値からガス体積を生成、v1 は多成分拡散の物理から膨張・3相・"
    "吸収を創発的に計算します（v1 の希釈持続日数は定数モデルの限界でやや短め）。"
    "いずれも IOP は房水動態＋眼剛性で計算。極端な過充填での非常に高い IOP は"
    "剛性モデルの外挿であり、絶対値より『危険な高 IOP になる』という傾向の方が信頼できます。"
)
