"""v1 物質移動係数 Q_i の較正。ハンドオフ §4, §7-A/§7-B。

§9.6 の方針決定: Thompson 1989 の図を目視で数値化する代わりに、複数文献で一致する
§7-A の定説スカラー（膨張倍率・ピーク時刻・持続日数）を較正ターゲットとして
Q_i を最小二乗同定する。図のCSVが用意できれば、同じ枠組みで残差を曲線フィットに
差し替えられる（TODO: refs/data/thompson1989_*.csv）。

較正は vitrectomized・膨張倍率測定は小気泡（f=0.1, 過充填と圧縮の影響を排除）で行う。

実行:
    python validation/thompson_fit.py
結果は validation/calibrated_Qi.json に書き出され、以降のシミュレーションが自動利用する。
"""

import json
import os

import numpy as np
from scipy.optimize import least_squares

from model.gas_kinetics import DiffusionParams
from model.simulator import SimConfig, simulate
from units import min_to_days, min_to_hours

# CO2 は溶解・拡散が速い → Q_N2 の倍率で固定。O2 は N2 と同程度。
Q_CO2_RATIO = 20.0
Q_O2_RATIO = 1.0

_V_VIT = 4.5

# 膨張倍率・ピーク測定用の充填率（ガスごと）。
# 臨床ベンチマークは「大きい気泡」での値。膨張のピーク充填率が ~0.75 になるよう
# f = 0.75/膨張倍率 とし、(1) 過充填・圧縮を避け、(2) 膨張相を球冠領域に保って
# 球冠→球 遷移が膨張測定へ誤干渉しないようにする（v1.2 で重要）。
F_EXPANSION = {
    "air": 0.6,
    "SF6": 0.75 / 2.0,  # ~0.375
    "C2F6": 0.75 / 3.3,  # ~0.23
    "C3F8": 0.75 / 4.0,  # ~0.19
}

# §7-A ターゲット（中央値）。
TARGETS = {
    "air_duration_d": 6.0,
    "SF6": {"peakfac": 2.0, "peak_h": 36.0, "ne_conc": 0.20, "duration_d": 14.0},
    "C2F6": {"peakfac": 3.3, "peak_h": 48.0, "ne_conc": 0.16, "duration_d": 31.0},
    "C3F8": {"peakfac": 4.0, "peak_h": 84.0, "ne_conc": 0.13, "duration_d": 67.0},
}


def _peak_window_days(gas):
    """ピーク到達を確実に捉える十分長い窓 [日]。"""
    return max(15.0, TARGETS[gas]["peak_h"] / 24.0 * 6.0)


def _dp_from_log(theta):
    """theta = [log10 Q_N2, log10 QG_SF6, log10 QG_C2F6, log10 QG_C3F8,
    log10 r_crit_mm, phi]（r_crit,phi = v1.2 球冠→球 遷移パラメータ）。

    注: foreign-gas 濃度依存 β（v1.3）はモデル側に実装済みだが、較正では β=0 を既定とする
    （β は膨張倍率を下げるが持続・ピーク時刻を悪化させ、総合では v1.2(β=0) が最良だった）。
    実験的に β を使うときは SimConfig.beta_G を手動指定する。
    """
    qn2 = 10.0 ** theta[0]
    return {
        "Q_N2": qn2,
        "Q_O2": qn2 * Q_O2_RATIO,
        "Q_CO2": qn2 * Q_CO2_RATIO,
        "SF6": 10.0 ** theta[1],
        "C2F6": 10.0 ** theta[2],
        "C3F8": 10.0 ** theta[3],
        "r_crit_mm": 10.0 ** theta[4],
        "phi": theta[5],
    }


def _run(gas, c, f, dpq, t_end_days, n_points=3000):
    dp = DiffusionParams(
        Q_G=dpq.get(gas, 0.0) if gas != "air" else 0.0,
        Q_N2=dpq["Q_N2"],
        Q_O2=dpq["Q_O2"],
        Q_CO2=dpq["Q_CO2"],
    )
    cfg = SimConfig(
        gas=gas,
        c=c,
        f=f,
        V_vit=_V_VIT,
        t_end_days=t_end_days,
        mechanism="diffusion",
        interface="cap_sphere",
        diffusion=dp,
        r_crit_mm=dpq["r_crit_mm"],
        fluid_efficiency=dpq["phi"],
        beta_G=0.0,  # 較正は β=0（v1.2）。ディスク上の較正値を読ませない。
        n_points=n_points,
    )
    return simulate(cfg)


def _peak(gas, c, dpq, t_end_days):
    f = F_EXPANSION.get(gas, 0.3)
    r = _run(gas, c, f, dpq, t_end_days)
    v = r.df["V_gas_mL"].values
    i = int(v.argmax())
    return v[i] / r.ic.V_gas0, min_to_hours(r.df["t_min"].values[i])


def _peaktime(gas, dpq, t_end_days):
    """ピーク到達時刻[h]: 臨床的な気泡サイズ(F_EXPANSION 充填, cap_sphere, 臨床眼)で測る。

    ピーク時刻は気泡サイズ(A/V)依存なので、臨床充填の実モデル構成で測るのが正しい。
    """
    f = F_EXPANSION.get(gas, 0.3)
    r = _run(gas, 1.0, f, dpq, t_end_days, interface="cap_sphere")
    v = r.df["V_gas_mL"].values
    return min_to_hours(r.df["t_min"].values[int(v.argmax())])


# 持続は臨床的な充填サイズ（大きい気泡ほど表面/体積比が小さく長寿命）で測る。
# 非膨張濃度では過充填しないので f=0.8 で安全。air も臨床的サイズで。
_F_DUR = 0.8


# 持続は「気泡が消えるまで＝球冠→球の残存しっぽ」を表すため深い閾値（2%）で測る。
_DUR_RESIDUAL = 0.02


def _duration(gas, c, dpq, t_end_days, residual=_DUR_RESIDUAL, f=_F_DUR):
    r = _run(gas, c, f, dpq, t_end_days)
    v = r.df["V_gas_mL"].values
    v0 = r.ic.V_gas0
    t = r.df["t_min"].values
    below = t[v <= residual * v0]
    if len(below) == 0:
        return None
    return min_to_days(below[0])


# 残差の重み（相対残差にかける）。持続と膨張倍率を重視、ピーク時刻は緩め。
# 膨張倍率・ピーク時刻（§7-A の看板数値）と air を最優先。持続は定数Qモデルでは
# 膨張倍率と過度に結合するため低重みにし、「順序・桁が合う」ことを狙う（厳密一致は v1.1）。
# v1.2 較正の重み。持続を球冠→球で合わせ込みつつ膨張倍率・ピーク時刻・air を両立。
# r_crit は「膨張相(大気泡)を球冠に保ちつつ末期(小気泡)だけ球化する」窓に入る必要があり、
# この重みバランス（持続0.5, 膨張1.0）で最良の同時適合が得られた（cost≈0.19）。
_W_PEAKFAC = 1.0
_W_PEAKTIME = 0.6
_W_DURATION = 0.5
_W_AIR = 1.0


def residuals(theta, verbose=False):
    dpq = _dp_from_log(theta)
    res = []

    # air: 持続 6 日（小気泡, 30 日窓）
    air_dur = _duration("air", 1.0, dpq, 30.0)
    air_dur = air_dur if air_dur is not None else 30.0
    res.append(
        _W_AIR * (air_dur - TARGETS["air_duration_d"]) / TARGETS["air_duration_d"]
    )

    # 各フッ素ガス: 100% 膨張倍率・ピーク時刻 ＋ 希釈濃度での持続日数
    for gas in ("SF6", "C2F6", "C3F8"):
        pf, ph = _peak(gas, 1.0, dpq, _peak_window_days(gas))
        res.append(
            _W_PEAKFAC * (pf - TARGETS[gas]["peakfac"]) / TARGETS[gas]["peakfac"]
        )
        res.append(_W_PEAKTIME * (ph - TARGETS[gas]["peak_h"]) / TARGETS[gas]["peak_h"])

        tgt_dur = TARGETS[gas]["duration_d"]
        dur = _duration(gas, TARGETS[gas]["ne_conc"], dpq, tgt_dur * 2.5)
        dur = dur if dur is not None else tgt_dur * 2.5
        res.append(_W_DURATION * (dur - tgt_dur) / tgt_dur)

    if verbose:
        print(
            "theta=",
            theta,
            "res=",
            np.round(res, 3),
            "cost=",
            0.5 * np.sum(np.array(res) ** 2),
        )
    return res


def calibrate(save=True):
    # 初期推定: [logQN2, logQG_SF6, logQG_C2F6, logQG_C3F8, log10 r_crit, phi]
    theta0 = np.array(
        list(np.log10([2.0e-13, 5e-14, 2.5e-14, 1.2e-14])) + [np.log10(6.0), 0.35]
    )
    lb = np.array(list(np.log10([1e-14, 1e-16, 1e-16, 1e-16])) + [np.log10(2.0), 0.1])
    ub = np.array(list(np.log10([1e-11, 1e-13, 1e-13, 1e-13])) + [np.log10(10.0), 1.0])

    sol = least_squares(
        residuals,
        theta0,
        bounds=(lb, ub),
        diff_step=0.05,
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=250,
    )
    dpq = _dp_from_log(sol.x)

    # 検証メトリクス（希釈濃度での持続・非膨張濃度の膨張）
    report = {"cost": float(0.5 * np.sum(sol.fun**2)), "gases": {}}
    for gas in ("air", "SF6", "C2F6", "C3F8"):
        entry = {}
        if gas == "air":
            entry["duration_d"] = _duration("air", 1.0, dpq, 30.0)
        else:
            pf, ph = _peak(gas, 1.0, dpq, _peak_window_days(gas))
            entry["peakfac_100"] = round(pf, 3)
            entry["peak_h_100"] = round(ph, 1)
            ne = TARGETS[gas]["ne_conc"]
            ne_pf, _ = _peak(gas, ne, dpq, _peak_window_days(gas))
            entry["peakfac_at_ne_conc"] = round(ne_pf, 3)
            dur = _duration(gas, ne, dpq, TARGETS[gas]["duration_d"] * 3.0)
            entry["duration_at_ne_d"] = round(dur, 1) if dur else None
        report["gases"][gas] = entry

    out = {
        "shared": {"Q_N2": dpq["Q_N2"], "Q_O2": dpq["Q_O2"], "Q_CO2": dpq["Q_CO2"]},
        "per_gas": {
            "air": {"Q_G": 0.0},
            "SF6": {"Q_G": dpq["SF6"]},
            "C2F6": {"Q_G": dpq["C2F6"]},
            "C3F8": {"Q_G": dpq["C3F8"]},
        },
        "interface": {"r_crit_mm": dpq["r_crit_mm"], "fluid_efficiency": dpq["phi"]},
        "report": report,
    }

    if save:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "calibrated_Qi.json"
        )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
    return out


if __name__ == "__main__":
    result = calibrate(save=True)
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_log.txt"),
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(json.dumps(result, indent=2, ensure_ascii=False))
    print("calibration done")
