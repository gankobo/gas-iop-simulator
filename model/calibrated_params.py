"""較正済み物質移動係数 Q_i のローダ。

実際の値は validation/thompson_fit.py が §7-A ベンチマークに最小二乗フィットして
`validation/calibrated_Qi.json` に書き出す。本モジュールはそれを読み込み、
無ければフォールバックの初期推定値を返す。

JSON 形式:
{
  "shared": {"Q_N2": ..., "Q_O2": ..., "Q_CO2": ...},
  "per_gas": {"SF6": {"Q_G": ...}, "C2F6": {...}, "C3F8": {...}, "air": {"Q_G": 0.0}}
}
"""

import json
import os

from model.gas_kinetics import DiffusionParams

_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "validation",
    "calibrated_Qi.json",
)

# フォールバック初期推定値（桁合わせ済み。較正前のブートストラップ用）。
_FALLBACK_SHARED = {"Q_N2": 1.5e-13, "Q_O2": 1.5e-13, "Q_CO2": 3.0e-12}
_FALLBACK_QG = {"air": 0.0, "SF6": 7.0e-15, "C2F6": 2.5e-15, "C3F8": 1.4e-15}


def _load_json():
    if os.path.exists(_JSON_PATH):
        with open(_JSON_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def get_calibrated_Q(gas: str) -> DiffusionParams:
    """ガス種の較正済み（無ければフォールバック）拡散パラメータを返す。"""
    data = _load_json()
    if data is not None:
        shared = data["shared"]
        qg = data["per_gas"].get(gas, {}).get("Q_G", _FALLBACK_QG.get(gas, 0.0))
        return DiffusionParams(
            Q_G=qg,
            Q_N2=shared["Q_N2"],
            Q_O2=shared["Q_O2"],
            Q_CO2=shared["Q_CO2"],
        )
    return DiffusionParams(
        Q_G=_FALLBACK_QG.get(gas, 0.0),
        Q_N2=_FALLBACK_SHARED["Q_N2"],
        Q_O2=_FALLBACK_SHARED["Q_O2"],
        Q_CO2=_FALLBACK_SHARED["Q_CO2"],
    )


# v1.2 球冠→球 遷移の界面パラメータ（フォールバック）。
_FALLBACK_INTERFACE = {"r_crit_mm": 3.0, "fluid_efficiency": 0.5}


def get_interface_params() -> dict:
    """較正済み（無ければフォールバック）の球冠→球 遷移パラメータを返す。"""
    data = _load_json()
    if data is not None and "interface" in data:
        return {
            "r_crit_mm": data["interface"].get(
                "r_crit_mm", _FALLBACK_INTERFACE["r_crit_mm"]
            ),
            "fluid_efficiency": data["interface"].get(
                "fluid_efficiency", _FALLBACK_INTERFACE["fluid_efficiency"]
            ),
        }
    return dict(_FALLBACK_INTERFACE)


# v1.3 foreign-gas 濃度依存係数 β のフォールバック（0 = 従来の定数）。
_FALLBACK_BETA_G = 0.0


def get_conc_dependence(gas: str) -> float:
    """較正済み（無ければ 0）の foreign-gas 濃度依存係数 β（ガス別）を返す。"""
    data = _load_json()
    if data is not None and "conc_dependence" in data:
        per_gas = data["conc_dependence"].get("beta_G_per_gas", {})
        return per_gas.get(gas, _FALLBACK_BETA_G)
    return _FALLBACK_BETA_G


def is_calibrated() -> bool:
    return _load_json() is not None
