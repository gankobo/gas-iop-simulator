# 眼内膨張性ガス IOP シミュレーター / Intraocular Expanding-Gas IOP Simulator

硝子体手術・pneumatic retinopexy で膨張性ガス（air・SF6・C2F6・C3F8）を注入したときの
**眼圧（IOP）の時間変化**を、気液動態の物理から計算する教育・研究用シミュレーターです。

A physics-based educational/research simulator that reproduces the **time course of intraocular
pressure (IOP)** after intravitreal injection of expanding gases (air, SF6, C2F6, C3F8) in
vitrectomy or pneumatic retinopexy.

---

> ## ⚠️ 重要な但し書き / Disclaimer
>
> **日本語** — これは気体の物理を可視化する **教育・研究用のデモ**であり、同時に
> 「定数係数モデルとして正直にどこまで再現できるか」を示す研究的試作でもあります。
> **臨床判断・診療・患者個別の予測には使用しないでください。医療機器ではありません。**
> 特に過充填時の絶対的な IOP 値は眼剛性モデルの外挿であり、数値そのものより
> 「危険な高 IOP になりうる」という傾向の方が信頼できます。無保証で提供します。
>
> **English** — This is an **educational/research demonstration** that visualizes gas physics,
> and at the same time a research prototype showing how far an honest *constant-coefficient*
> model can go. **Do not use it for clinical decisions, patient care, or individual prediction.
> It is not a medical device.** Absolute IOP values under overfill are extrapolations of the
> ocular-rigidity model; the qualitative trend (“dangerously high IOP can occur”) is more
> trustworthy than the exact number. Provided without warranty.

---

## 触ってみる / Try it

- **オンライン（リンクで誰でも）/ Online:** **https://gas-iop-simulator-nbcyuehnq2nlhu7hxynappy.streamlit.app/**
- **手元で動かす / Run locally:**

  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```

  ブラウザが開き、左のスライダーでガス・濃度・置換率・眼のサイズを変えると、
  IOP・気泡体積・充填率のグラフがその場で更新されます。

Python から直接呼ぶこともできます / You can also call it from Python:

```python
from model.simulator import SimConfig, simulate

cfg = SimConfig(gas="SF6", c=0.20, f=0.6, V_vit=4.5)  # SF6 20%, fill 60%, vitreous 4.5 mL
r = simulate(cfg)
print(r.peak_IOP, "mmHg @", r.peak_IOP_time_h, "h")
print(r.df.head())
for w in r.warnings:
    print("warning:", w)
```

---

## 何ができるか / What it does

- ガス種・濃度・置換率・眼のサイズ（眼軸長 or 硝子体腔容積）を指定して **IOP(t)** を計算
- 気泡体積 V_gas(t)・充填率・気泡内の各ガス分圧など内部量も出力
- 過充填や非生理的に高い IOP には**警告**を表示
- 2 つの計算モデルを切替：
  - **v0（経験式）** — 文献の消失曲線からガス体積を直接生成（高速）
  - **v1.2（多成分拡散）** — 気体拡散の物理から膨張・3 相・吸収を*創発的に*再現

---

## モデルの前提 / Model assumptions

物理的に分離した 2 つの時間スケールを連立して解いています（`scipy.solve_ivp`, LSODA）。

The simulator couples two physically separated timescales (solved with `scipy.solve_ivp`, LSODA):

1. **遅い系（時〜週）/ Slow (hours–weeks): gas bubble.**
   気泡内の 4 成分（フッ素ガス G・N2・O2・CO2）が界面を通って拡散：
   `dn_i/dt = −Q_i·A·(p_i − p_i,tissue)`。純フッ素ガス注入直後は気泡内 N2 ≒ 0 のため、
   組織側 N2（約 570 mmHg）が流入して気泡が膨張し（**膨張相**）、N2 が組織値に達すると
   膨張が止まり（**平衡相**）、その後フッ素ガスが抜けて気泡が吸収されます（**吸収相**）。
   組織の総ガス分圧が大気圧より低い（不飽和, 約 709 < 760 mmHg）ことが、最終的にどんな
   気泡も完全吸収される根本的な駆動力です。

2. **気泡の形 / Bubble geometry.**
   気泡は硝子体腔（球状の空洞）の上部に浮き、大きいうちは網膜に押し付けられた**球冠**
   （血流組織と広く接触＝吸収が速い）、小さくなると表面張力で**球形**へ移行（液面接触が主で
   吸収が遅い）します。この球冠→球移行（Bond 数で遷移）が、希釈ガスの長期残存を生みます。

3. **速い系（分〜時）/ Fast (minutes–hours): aqueous + globe.**
   房水の産生・流出（Goldmann）と眼剛性（Friedenwald）で IOP を緩衝。
   ガス体積の増加が IOP を押し上げ、上がった IOP が房水流出を増やして緩和する**負帰還**。
   IOP 上昇は気泡を圧縮するので双方向結合です。

主な単純化 / Main simplifications:
- 物質移動係数 Q_i は**ガスごとに定数**（濃度・履歴に依存しない）。
- V_fluid は硝子体腔のみ（前房房水は未実装）。眼剛性 K は固定既定値（UI で可変）。
- 組織側ガス分圧は近似値を採用。個体別の較正は行っていません。

---

## 数値の根拠 / Where the numbers come from

このモデルは特定の患者データに当てはめたものではなく、**複数の一次文献で一致する
定説スカラー**（膨張倍率・ピーク時刻・持続日数・非膨張濃度）を較正・検証ターゲットにしています。

| 量 / Quantity | 値の根拠 / Source |
|---|---|
| 膨張倍率（CF4 1.9 / C2F6 3.3 / C3F8 4 倍）| Lincoff et al. 1980 (PMID 7425930) |
| ガス消失動態・半減期 | Thompson 1989 (PMID 2719578) |
| 非膨張濃度・独立クロスチェック | Williamson et al. 2018, gas eye model (PMID 29232331) |
| 球冠→球移行（長期残存の機構）| Hall et al. 2017, AIChE J (DOI 10.1002/aic.15739) |
| 房水動態（Goldmann 標準値）| Brubaker 1991 (Friedenwald Lecture) |
| 眼剛性 C_oc(IOP)=1/(ln10·K·IOP) | Friedenwald 1937 |
| 眼軸長→硝子体腔容積 | 線形コホート回帰 / MRI 回帰 Zhou et al. 2020 (PMID 33080714) |

v1.2 の文献適合（vitrectomized 較正, 4 成分定数 Q + 球冠→球）/ Benchmark fit:

| ガス | 膨張倍率(文献) | ピーク時刻(文献) | 希釈持続(文献) |
|---|---|---|---|
| air  | —（非膨張）   | —             | 6.0 日 (≈6) |
| SF6  | 2.7 倍 (2–2.5) | 37 h (24–48) | 13.8 日 (≈14) |
| C2F6 | 4.4 倍 (≈3.3) | 61 h (36–60) | 29.3 日 (28–35) |
| C3F8 | 5.4 倍 (≈4)   | 78 h (72–96) | 67.8 日 (≈67) |

持続日数・ピーク時刻・air の非膨張は文献とほぼ一致します。膨張倍率は C3F8 でやや高めに出ます
（下記「限界」参照）。較正は `python -m validation.thompson_fit` で再現できます。

The model is **not fitted to any individual patient**. It is calibrated and validated against
consensus scalars (expansion ratio, peak time, duration, nonexpansile concentration) that agree
across multiple primary sources, listed above.

---

## 限界（正直な前提）/ Limitations (the honest part)

このシミュレーターは「**定数係数クラスのモデルとして正直にどこまで到達できるか**」を示すものです。
以下は便宜的に隠さず、構造的な限界として明示します。

This simulator is meant to show the honest reach of a **constant-coefficient class** model.
The following limits are stated openly rather than hidden behind tuning:

1. **膨張倍率と持続日数を完全には両立できない / Expansion ratio vs. duration cannot be fully decoupled.**
   定数 Q モデルでは両者が結合します。v1.2 は希釈持続を文献値にほぼ一致させた代償として、
   膨張倍率がやや高め（C3F8 ≈5.4 vs 文献 ≈4）になります。膨張倍率を優先したい場合は
   `interface="cap"` を使うと膨張倍率は文献どおりですが持続が短くなります。

2. **過充填時の絶対 IOP は外挿 / Absolute IOP under overfill is an extrapolation.**
   膨張性ガスを高い置換率で入れると房水が枯渇し、IOP が非生理域まで上昇します。
   「過充填＝危険」という定性的結論は正しいものの、絶対値は眼剛性モデルの外挿であり
   参考に留めてください（該当時は警告を表示）。

3. **濃度依存の消失速度は約 1.9 倍までで頭打ち / Concentration-dependent half-life plateaus.**
   Thompson 1992 は「C3F8 濃度が高いほど消失半減期が長い（5%→20% で約 3 倍）」を報告。
   本モデルは向き（濃いほど遅い）は幾何だけで自然に再現しますが、傾きは約 1.9 倍で頭打ちになり、
   較正・充填率・濃度依存係数のいずれでも文献の約 3 倍には届きません。これは「減衰相ではどの初期
   濃度の気泡も同じ組成へ収束し、"今の状態"で係数を決める仕組みでは初期用量を区別できない」ためです。
   最後の差を埋めるには用量/履歴を記憶する状態変数が要りますが、それは現象を合わせる便宜的装置で
   物理的・生物学的な独立根拠を欠くため、**あえて導入していません**。

4. **未実装 / Not modeled yet.** 前房房水を含めた流体容積、個体別の眼剛性較正、
   圧–体積実測カーブの差し込み。

---

## テスト / Tests

```bash
python -m pytest tests/ -q
```

- `tests/test_benchmarks_v0.py` — 文献動態ベンチマーク（膨張倍率・ピーク時刻・持続日数・非膨張濃度）
- `tests/test_invariants.py` — 物理インバリアント（非負・最終吸収・air 非膨張・単峰・単調性・警告）
- `tests/test_v1_emergence.py` — v1 機構モデルの創発（air 非膨張・3 相・膨張順序・分圧整合・最終吸収）

---

## ファイル構成 / Layout

```
app.py                        Streamlit 対話 UI（v0 / v1.2 切替）
units.py                      単位変換の単一窓口（mmHg / mL / min / mol）
model/gas_empirical.py        v0 経験式ガスモデル
model/gas_kinetics.py         v1.2 多成分拡散 + 球冠→球界面
model/calibrated_params.py    較正済み Q_i のローダ
model/aqueous.py              房水動態（Goldmann）+ 平衡 IOP
model/globe.py                眼剛性・コンプライアンス（Friedenwald）
model/simulator.py            連立 ODE の統合
io_eye/eye_geometry.py        眼軸長→容積、初期条件生成
validation/benchmarks.py      文献ベンチマーク定数・合格判定
validation/thompson_fit.py    物質移動係数 Q_i の較正
tests/                        pytest（インバリアント + ベンチマーク + 創発）
```

参考文献の一覧は [REFERENCES.md](REFERENCES.md) を参照してください。
See [REFERENCES.md](REFERENCES.md) for the full bibliography.

---

## ライセンス / License

MIT License — see [LICENSE](LICENSE).
コードは自由に利用・改変できます（無保証）。ただし上記の通り臨床利用は想定していません。
