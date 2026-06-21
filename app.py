"""眼内膨張性ガス IOP シミュレーター — Streamlit 対話UI（日本語 / English 切替）。

起動: プロジェクトフォルダで
    streamlit run app.py
ブラウザが開き、スライダーで条件を変えると IOP(t) などが即座に更新される。
サイドバー上部の「言語 / Language」で日本語・英語を切り替えられる。
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

# ============================ 文言テーブル（日本語 / English）============================
TXT = {
    "日本語": {
        "title": "眼内膨張性ガス IOP シミュレーター",
        "caption": "硝子体手術 / pneumatic retinopexy で膨張性ガス（air・SF6・C2F6・C3F8）を入れたときの"
                   "眼圧（IOP）の時間変化を、気液動態の物理に基づいて再現します。",
        "disclaimer": "⚠️ 教育・研究用のデモです（医療機器ではありません）。臨床判断・患者個別の予測には"
                      "使用しないでください。特に過充填時の絶対 IOP 値は外挿で、数値より傾向の方が信頼できます。",
        "sidebar_header": "入力条件",
        "model_label": "計算モデル",
        "model_v0": "v0（経験式・高速）",
        "model_v1": "v1.2（多成分拡散・球冠→球移行）",
        "model_help": "v0 は文献の経験式でガス体積を生成。v1.2 は気体の拡散物理から"
                      "膨張・3相・吸収を創発的に計算。気泡は大きいうちは網膜に押し付けられた"
                      "球冠、小さくなると球形へ移行する（吸収が遅れ長期残存）。各ガス分圧も出力。",
        "gas_label": "ガスの種類",
        "air_info": "air は希釈しないため濃度は 100% 固定です。",
        "conc_label": "ガス濃度（容積分率）",
        "conc_help": "{gas} の非膨張（等膨張）濃度の目安は約 {pct:.0f}%。"
                     "これより高いと膨張、低いとほぼ等容で吸収。",
        "fill_label": "硝子体腔の置換率 f（fill fraction）",
        "fill_help": "注入直後にガスが硝子体腔の何割を占めるか。",
        "medium_label": "眼の状態（拡散速度に影響）",
        "medium_vit": "硝子体切除眼（vitrectomized）",
        "medium_phakic": "硝子体・水晶体あり（吸収が約2.5倍遅い）",
        "size_header": "眼のサイズ",
        "size_mode_label": "入力方法",
        "size_mode_AL": "眼軸長 AL から推定",
        "size_mode_direct": "硝子体腔容積を直接入力",
        "AL_label": "眼軸長 AL [mm]",
        "al_model_label": "推定式",
        "al_linear": "線形（実測コホート）",
        "al_mri_emm": "MRI 正視眼",
        "al_mri_myo": "MRI 病的近視",
        "V_vit_metric": "推定 硝子体腔容積",
        "V_vit_direct": "硝子体腔容積 [mL]",
        "iop_header": "初期眼圧（t=0）",
        "use_eq": "房水平衡値を自動で使う（既定）",
        "use_eq_help": "チェックを外すと、t=0 の眼圧を手動で指定できます。",
        "eq_caption": "房水動態から自動計算した平衡 IOP を初期値にします。",
        "IOP0_label": "初期 IOP [mmHg]",
        "advanced": "詳細設定（上級者向け）",
        "K_label": "眼剛性 K [1/µL]（高いほど眼が硬い）",
        "P_atm_label": "大気圧 [mmHg]（高地・気圧変化テスト）",
        "F_prod_label": "房水産生 [µL/min]",
        "C_tm_label": "線維柱帯流出能 [µL/min/mmHg]",
        "t_end_label": "シミュレーション期間 [日]",
        "metric_baseline": "ベースライン IOP",
        "metric_peak": "ピーク IOP",
        "metric_peak_time": "ピーク到達",
        "metric_peak_time_val": "{h:.1f} 時間後",
        "metric_bubble": "初期気泡体積",
        "time_change": "時間変化",
        "tab_all": "全期間",
        "tab_7d": "最初の7日間（IOPスパイク拡大）",
        "chart_iop": "**眼圧 IOP [mmHg]**",
        "chart_fill_vol": "**充填率 と 気泡体積 [mL]**",
        "pp_subheader": "気泡内のガス分圧 [mmHg]（v1.2 機構モデル）",
        "pp_caption": "純ガス注入直後は窒素（N2）がほぼ 0。組織から N2 が流入して気泡が膨張し、"
                      "N2 が組織値（約570）に達すると膨張が止まり、その後フッ素ガス（G）が"
                      "ゆっくり抜けて気泡全体が吸収されます——この3相が物理から自然に出ています。",
        "pp_G": "フッ素ガス G",
        "bm_expander": "文献ベンチマーク（§7-A）との照合",
        "bm_col_item": "項目",
        "bm_col_lit": "文献値",
        "bm_row_exp": "100%膨張倍率",
        "bm_exp_val": "{a}–{b} 倍",
        "bm_row_peak": "ピーク到達",
        "bm_peak_val": "{a:.0f}–{b:.0f} 時間",
        "bm_row_ne": "非膨張濃度",
        "bm_ne_val": "{a:.0f}–{b:.0f}%",
        "bm_row_dur": "希釈時の持続",
        "bm_dur_val": "{a:.0f}–{b:.0f} 日",
        "bm_caption": "最大膨張倍率（100%注入の文献値）: {ratio} 倍 / このモデルの平衡IOP: {iop:.1f} mmHg",
        "data_expander": "計算データ（CSV ダウンロード可）",
        "download_btn": "CSV をダウンロード",
        "footer": "※ v0 は §7-A の文献値からガス体積を生成、v1 は多成分拡散の物理から膨張・3相・"
                  "吸収を創発的に計算します（v1 の希釈持続日数は定数モデルの限界でやや短め）。"
                  "いずれも IOP は房水動態＋眼剛性で計算。極端な過充填での非常に高い IOP は"
                  "剛性モデルの外挿であり、絶対値より『危険な高 IOP になる』という傾向の方が信頼できます。",
    },
    "English": {
        "title": "Intraocular Expanding-Gas IOP Simulator",
        "caption": "Reproduces the time course of intraocular pressure (IOP) after intravitreal "
                   "injection of expanding gases (air, SF6, C2F6, C3F8) in vitrectomy / "
                   "pneumatic retinopexy, based on gas–liquid dynamics.",
        "disclaimer": "⚠️ Educational/research demo — **not a medical device**. Do not use for "
                      "clinical decisions or individual prediction. Absolute IOP under overfill is "
                      "an extrapolation; the trend is more reliable than the exact number.",
        "sidebar_header": "Inputs",
        "model_label": "Model",
        "model_v0": "v0 (empirical, fast)",
        "model_v1": "v1.2 (multi-component diffusion, cap→sphere)",
        "model_help": "v0 generates bubble volume from literature empirical curves. v1.2 derives "
                      "expansion, the 3 phases, and absorption emergently from gas-diffusion "
                      "physics. The bubble is a cap pressed against the retina while large, "
                      "transitioning to a sphere when small (slower absorption, longer residual). "
                      "Partial pressures are also output.",
        "gas_label": "Gas type",
        "air_info": "Air is not diluted, so concentration is fixed at 100%.",
        "conc_label": "Gas concentration (volume fraction)",
        "conc_help": "The nonexpansile (iso-volume) concentration of {gas} is around {pct:.0f}%. "
                     "Higher → expansion; lower → roughly iso-volume absorption.",
        "fill_label": "Vitreous fill fraction f",
        "fill_help": "Fraction of the vitreous cavity occupied by gas right after injection.",
        "medium_label": "Eye status (affects diffusion rate)",
        "medium_vit": "Vitrectomized eye",
        "medium_phakic": "Phakic, intact vitreous (~2.5× slower absorption)",
        "size_header": "Eye size",
        "size_mode_label": "Input method",
        "size_mode_AL": "Estimate from axial length AL",
        "size_mode_direct": "Enter vitreous volume directly",
        "AL_label": "Axial length AL [mm]",
        "al_model_label": "Estimation formula",
        "al_linear": "Linear (measured cohort)",
        "al_mri_emm": "MRI emmetrope",
        "al_mri_myo": "MRI pathological myopia",
        "V_vit_metric": "Estimated vitreous volume",
        "V_vit_direct": "Vitreous volume [mL]",
        "iop_header": "Initial IOP (t=0)",
        "use_eq": "Use aqueous equilibrium value automatically (default)",
        "use_eq_help": "Uncheck to set the t=0 IOP manually.",
        "eq_caption": "Uses the equilibrium IOP computed from aqueous dynamics as the initial value.",
        "IOP0_label": "Initial IOP [mmHg]",
        "advanced": "Advanced settings",
        "K_label": "Ocular rigidity K [1/µL] (higher = stiffer eye)",
        "P_atm_label": "Atmospheric pressure [mmHg] (altitude / pressure test)",
        "F_prod_label": "Aqueous production [µL/min]",
        "C_tm_label": "Trabecular outflow facility [µL/min/mmHg]",
        "t_end_label": "Simulation duration [days]",
        "metric_baseline": "Baseline IOP",
        "metric_peak": "Peak IOP",
        "metric_peak_time": "Time to peak",
        "metric_peak_time_val": "{h:.1f} h",
        "metric_bubble": "Initial bubble volume",
        "time_change": "Time course",
        "tab_all": "Full period",
        "tab_7d": "First 7 days (IOP spike, zoomed)",
        "chart_iop": "**IOP [mmHg]**",
        "chart_fill_vol": "**Fill fraction & bubble volume [mL]**",
        "pp_subheader": "Partial pressures inside the bubble [mmHg] (v1.2 mechanistic model)",
        "pp_caption": "Right after pure-gas injection, nitrogen (N2) is near 0. N2 flows in from "
                      "tissue and the bubble expands; when N2 reaches the tissue value (~570), "
                      "expansion stops, then the fluorinated gas (G) slowly leaves and the whole "
                      "bubble is absorbed — these 3 phases emerge naturally from the physics.",
        "pp_G": "Fluorinated gas G",
        "bm_expander": "Comparison with literature benchmarks (§7-A)",
        "bm_col_item": "Quantity",
        "bm_col_lit": "Literature value",
        "bm_row_exp": "100% expansion ratio",
        "bm_exp_val": "{a}–{b}×",
        "bm_row_peak": "Time to peak",
        "bm_peak_val": "{a:.0f}–{b:.0f} h",
        "bm_row_ne": "Nonexpansile concentration",
        "bm_ne_val": "{a:.0f}–{b:.0f}%",
        "bm_row_dur": "Duration when diluted",
        "bm_dur_val": "{a:.0f}–{b:.0f} days",
        "bm_caption": "Max expansion ratio (literature, 100% fill): {ratio}× / "
                      "model equilibrium IOP: {iop:.1f} mmHg",
        "data_expander": "Computed data (CSV download)",
        "download_btn": "Download CSV",
        "footer": "v0 generates bubble volume from §7-A literature values; v1 computes expansion, "
                  "the 3 phases, and absorption emergently from multi-component diffusion physics "
                  "(v1's diluted-duration is slightly short due to the constant-coefficient limit). "
                  "In both, IOP is computed from aqueous dynamics + ocular rigidity. Very high IOP "
                  "under extreme overfill is an extrapolation of the rigidity model; the trend "
                  "('dangerously high IOP can occur') is more reliable than the absolute value.",
    },
}

st.set_page_config(page_title="眼内ガス IOP シミュレーター / Intraocular Gas IOP Simulator",
                   layout="wide")

# 言語選択（サイドバー最上部）
lang = st.sidebar.radio("言語 / Language", ["日本語", "English"], horizontal=True)


def t(key, **kw):
    """現在の言語の文言を返す（必要なら .format で埋め込み）。"""
    s = TXT[lang][key]
    return s.format(**kw) if kw else s


st.title(t("title"))
st.caption(t("caption"))
st.error(t("disclaimer"))

# ============================ サイドバー: 入力 ============================
with st.sidebar:
    st.header(t("sidebar_header"))

    model_choices = {0: t("model_v0"), 1: t("model_v1")}
    mech_idx = st.radio(
        t("model_label"), [0, 1], format_func=lambda i: model_choices[i],
        help=t("model_help"),
    )
    mechanism = "diffusion" if mech_idx == 1 else "empirical"

    gas = st.selectbox(t("gas_label"), ["air", "SF6", "C2F6", "C3F8"], index=1)

    # 濃度（air は純空気＝1.0 固定）
    if gas == "air":
        c = 1.0
        st.info(t("air_info"))
    else:
        c_ne = NONEXPANSILE_CONC_CLINICAL[gas]
        c = st.slider(
            t("conc_label"), min_value=0.05, max_value=1.0,
            value=float(c_ne), step=0.01,
            help=t("conc_help", gas=gas, pct=c_ne * 100),
        )

    f = st.slider(
        t("fill_label"), min_value=0.05, max_value=1.0,
        value=0.6, step=0.05, help=t("fill_help"),
    )

    medium_labels = {
        "vitrectomized": t("medium_vit"),
        "phakic_intact_vitreous": t("medium_phakic"),
    }
    medium = st.radio(
        t("medium_label"), ["vitrectomized", "phakic_intact_vitreous"],
        format_func=lambda m: medium_labels[m],
    )

    st.divider()
    st.subheader(t("size_header"))
    size_modes = {"AL": t("size_mode_AL"), "direct": t("size_mode_direct")}
    size_mode = st.radio(t("size_mode_label"), ["AL", "direct"],
                         format_func=lambda m: size_modes[m])
    if size_mode == "AL":
        AL = st.slider(t("AL_label"), 20.0, 32.0, 23.4, 0.1)
        al_labels = {
            "linear": t("al_linear"),
            "mri_emmetrope": t("al_mri_emm"),
            "mri_myopia": t("al_mri_myo"),
        }
        al_model = st.selectbox(
            t("al_model_label"), ["linear", "mri_emmetrope", "mri_myopia"],
            format_func=lambda m: al_labels[m],
        )
        V_vit = vitreous_volume_from_AL(AL, al_model)
        st.metric(t("V_vit_metric"), f"{V_vit:.2f} mL")
    else:
        V_vit = st.slider(t("V_vit_direct"), 2.5, 10.0, 4.5, 0.1)

    st.divider()
    st.subheader(t("iop_header"))
    use_eq = st.checkbox(t("use_eq"), value=True, help=t("use_eq_help"))
    if use_eq:
        IOP0 = None
        st.caption(t("eq_caption"))
    else:
        IOP0 = st.slider(t("IOP0_label"), 5.0, 40.0, 15.0, 1.0)

    st.divider()
    with st.expander(t("advanced")):
        K_uL = st.slider(t("K_label"), 0.0125, 0.025, 0.0184, 0.0005)
        P_atm = st.slider(t("P_atm_label"), 500.0, 800.0, 760.0, 5.0)
        F_prod = st.slider(t("F_prod_label"), 1.0, 3.5, 2.4, 0.1)
        C_tm = st.slider(t("C_tm_label"), 0.10, 0.40, 0.26, 0.01)
        t_end_days = st.slider(t("t_end_label"), 7, 200, 90, 1)

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
col1.metric(t("metric_baseline"), f"{result.IOP0:.1f} mmHg")
col2.metric(t("metric_peak"), f"{result.peak_IOP:.1f} mmHg",
            delta=f"+{result.peak_IOP - result.IOP0:.1f}")
col3.metric(t("metric_peak_time"), t("metric_peak_time_val", h=result.peak_IOP_time_h))
col4.metric(t("metric_bubble"), f"{result.ic.V_gas0:.2f} mL")

# ============================ グラフ ============================
st.subheader(t("time_change"))
tab1, tab2 = st.tabs([t("tab_all"), t("tab_7d")])


def _plot(frame):
    left, right = st.columns(2)
    with left:
        st.markdown(t("chart_iop"))
        st.line_chart(frame.set_index("t_days")[["IOP_mmHg"]])
    with right:
        st.markdown(t("chart_fill_vol"))
        st.line_chart(frame.set_index("t_days")[["fill_fraction"]])
        st.line_chart(frame.set_index("t_days")[["V_gas_mL"]])


with tab1:
    _plot(df)
with tab2:
    _plot(df[df["t_days"] <= 7.0])

# v1（機構モデル）では気泡内の各ガス分圧を表示
if mechanism == "diffusion" and "p_G_mmHg" in df.columns:
    st.subheader(t("pp_subheader"))
    st.caption(t("pp_caption"))
    pcols = {"p_G_mmHg": t("pp_G"), "p_N2_mmHg": "N2",
             "p_O2_mmHg": "O2", "p_CO2_mmHg": "CO2"}
    pp = df[df["V_gas_mL"] > 1e-3].rename(columns=pcols).set_index("t_days")
    st.line_chart(pp[list(pcols.values())])

# ============================ ベンチマーク照合 ============================
with st.expander(t("bm_expander")):
    bm = BENCHMARKS[gas]
    rows = []
    if gas != "air":
        rows.append((t("bm_row_exp"),
                     t("bm_exp_val", a=bm.expansion_100pct[0], b=bm.expansion_100pct[1])))
        rows.append((t("bm_row_peak"),
                     t("bm_peak_val", a=bm.peak_time_h[0], b=bm.peak_time_h[1])))
        rows.append((t("bm_row_ne"),
                     t("bm_ne_val", a=bm.nonexpansile_conc[0] * 100,
                       b=bm.nonexpansile_conc[1] * 100)))
    rows.append((t("bm_row_dur"),
                 t("bm_dur_val", a=bm.duration_days[0], b=bm.duration_days[1])))
    st.table(pd.DataFrame(rows, columns=[t("bm_col_item"), t("bm_col_lit")]))
    st.caption(t("bm_caption", ratio=GAS_MAX_EXPANSION[gas], iop=equilibrium_IOP(aq)))

# ============================ 生データ ============================
with st.expander(t("data_expander")):
    st.dataframe(df, use_container_width=True)
    st.download_button(
        t("download_btn"), df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"iop_sim_{gas}_c{c}_f{f}.csv", mime="text/csv",
    )

st.caption(t("footer"))
