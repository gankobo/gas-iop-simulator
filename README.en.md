🌐 [日本語](README.md) | **English**

# Intraocular Expanding-Gas IOP Simulator

A physics-based educational/research simulator that reproduces the **time course of intraocular
pressure (IOP)** after intravitreal injection of expanding gases (air, SF6, C2F6, C3F8) in
vitrectomy or pneumatic retinopexy.

---

> ## ⚠️ Disclaimer
>
> This is an **educational/research demonstration** that visualizes gas physics, and at the
> same time a research prototype showing how far an honest *constant-coefficient* model can go.
> **Do not use it for clinical decisions, patient care, or individual prediction. It is not a
> medical device.** Absolute IOP values under overfill are extrapolations of the ocular-rigidity
> model; the qualitative trend (“dangerously high IOP can occur”) is more trustworthy than the
> exact number. Provided without warranty.

---

## Try it

- **Online (anyone, via link):** **https://gas-iop-simulator-nbcyuehnq2nlhu7hxynappy.streamlit.app/**
- **Run locally:**

  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```

  A browser opens; change gas, concentration, fill fraction, and eye size with the sliders on
  the left, and the IOP / bubble-volume / fill-fraction charts update in place. Use the
  “言語 / Language” selector at the top of the sidebar to switch between Japanese and English.

You can also call it from Python:

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

## What it does

- Computes **IOP(t)** for a given gas, concentration, fill fraction, and eye size
  (axial length or vitreous-cavity volume).
- Also outputs internal quantities: bubble volume V_gas(t), fill fraction, and the partial
  pressures of each gas inside the bubble.
- Shows **warnings** for overfill or non-physiological IOP.
- Switch between two models:
  - **v0 (empirical)** — generates bubble volume directly from literature disappearance curves (fast).
  - **v1.2 (multi-component diffusion)** — reproduces expansion, the 3 phases, and absorption
    *emergently* from gas-diffusion physics.

---

## Model assumptions

The simulator couples two physically separated timescales (solved with `scipy.solve_ivp`, LSODA):

1. **Slow (hours–weeks): gas bubble.**
   Four components inside the bubble (fluorinated gas G, N2, O2, CO2) diffuse across the
   interface: `dn_i/dt = −Q_i·A·(p_i − p_i,tissue)`. Right after pure fluorinated-gas injection
   the bubble's N2 ≈ 0, so tissue N2 (~570 mmHg) flows in and the bubble expands (**expansion
   phase**); once N2 reaches the tissue value, expansion stops (**equilibrium phase**); then the
   fluorinated gas leaves and the bubble is absorbed (**absorption phase**). The fact that total
   tissue gas tension is below atmospheric (undersaturation, ~709 < 760 mmHg) is the fundamental
   driving force that ultimately fully absorbs any bubble.

2. **Bubble geometry.**
   The bubble floats at the top of the (spherical) vitreous cavity: while large it is a
   **spherical cap** pressed against the retina (broad contact with perfused tissue → fast
   absorption); as it shrinks, surface tension makes it transition to a **sphere** (mostly
   fluid contact → slow absorption). This cap→sphere transition (governed by the Bond number)
   produces the long residual of diluted gas.

3. **Fast (minutes–hours): aqueous + globe.**
   Aqueous production/outflow (Goldmann) and ocular rigidity (Friedenwald) buffer the IOP.
   A rise in gas volume pushes IOP up, and the elevated IOP increases aqueous outflow, relaxing
   it — a **negative feedback**. The IOP rise also compresses the bubble, so the coupling is
   bidirectional.

Main simplifications:
- The mass-transfer coefficients Q_i are **constant per gas** (independent of concentration or history).
- V_fluid is the vitreous cavity only (anterior-chamber aqueous not modeled). Ocular rigidity K
  has a fixed default (adjustable in the UI).
- Tissue-side gas tensions use approximate values. No per-individual calibration is performed.

---

## Where the numbers come from

The model is **not fitted to any individual patient**. It is calibrated and validated against
consensus scalars (expansion ratio, peak time, duration, nonexpansile concentration) that agree
across multiple primary sources.

| Quantity | Source |
|---|---|
| Expansion ratio (CF4 1.9 / C2F6 3.3 / C3F8 4×) | Lincoff et al. 1980 (PMID 7425930) |
| Gas disappearance kinetics / half-life | Thompson 1989 (PMID 2719578) |
| Nonexpansile concentration / independent cross-check | Williamson et al. 2018, gas eye model (PMID 29232331) |
| Cap→sphere transition (long-residual mechanism) | Hall et al. 2017, AIChE J (DOI 10.1002/aic.15739) |
| Aqueous dynamics (Goldmann standard values) | Brubaker 1991 (Friedenwald Lecture) |
| Ocular rigidity C_oc(IOP)=1/(ln10·K·IOP) | Friedenwald 1937 |
| Axial length → vitreous volume | Linear cohort regression / MRI regression Zhou et al. 2020 (PMID 33080714) |

v1.2 benchmark fit (vitrectomized calibration, 4-component constant Q + cap→sphere):

| Gas | Expansion (lit.) | Peak time (lit.) | Diluted duration (lit.) |
|---|---|---|---|
| air  | — (nonexpansile) | —             | 6.0 d (≈6) |
| SF6  | 2.7× (2–2.5)     | 37 h (24–48)  | 13.8 d (≈14) |
| C2F6 | 4.4× (≈3.3)      | 61 h (36–60)  | 29.3 d (28–35) |
| C3F8 | 5.4× (≈4)        | 78 h (72–96)  | 67.8 d (≈67) |

Duration, peak time, and air's nonexpansile behavior closely match the literature. The expansion
ratio runs slightly high for C3F8 (see "Limitations"). The calibration is reproducible with
`python -m validation.thompson_fit`.

---

## Limitations (the honest part)

This simulator is meant to show the honest reach of a **constant-coefficient class** model.
The following limits are stated openly rather than hidden behind tuning:

1. **Expansion ratio vs. duration cannot be fully decoupled.**
   In a constant-Q model the two are coupled. v1.2 matches the diluted duration to the literature,
   at the cost of a slightly high expansion ratio (C3F8 ≈5.4 vs literature ≈4). If you prefer to
   prioritize the expansion ratio, use `interface="cap"`: the expansion matches the literature but
   the duration becomes shorter.

2. **Absolute IOP under overfill is an extrapolation.**
   Injecting expanding gas at a high fill fraction depletes the aqueous and pushes IOP into the
   non-physiological range. The qualitative conclusion ("overfill = dangerous") is correct, but the
   absolute value is an extrapolation of the ocular-rigidity model and should be treated as
   indicative only (a warning is shown in such cases).

3. **Concentration-dependent half-life plateaus at ~1.9×.**
   Thompson 1992 reports that higher C3F8 concentration gives a longer disappearance half-life
   (~3× from 5%→20%). This model reproduces the direction (denser = slower) naturally from geometry
   alone, but the slope plateaus at ~1.9× and reaches neither the literature ~3× through calibration,
   fill fraction, nor a concentration-dependent coefficient. This is because "in the decay phase every
   bubble converges to the same composition regardless of initial concentration, so a scheme that sets
   coefficients from the *current* state cannot distinguish the initial dose." Closing the last gap
   would require a state variable that remembers dose/history, but that is an expedient device for
   fitting the phenomenon, lacking independent physical/biological grounding — so it is **deliberately
   not introduced**.

4. **Not modeled yet.** Fluid volume including anterior-chamber aqueous, per-individual ocular-rigidity
   calibration, and plugging in a measured pressure–volume curve.

---

## Tests

```bash
python -m pytest tests/ -q
```

- `tests/test_benchmarks_v0.py` — literature kinetic benchmarks (expansion, peak time, duration, nonexpansile conc.)
- `tests/test_invariants.py` — physical invariants (non-negativity, eventual absorption, air nonexpansile, single peak, monotonicity, warnings)
- `tests/test_v1_emergence.py` — emergence of the v1 mechanistic model (air nonexpansile, 3 phases, expansion order, partial-pressure consistency, eventual absorption)

---

## Layout

```
app.py                        Streamlit interactive UI (v0 / v1.2 switch, JP/EN switch)
units.py                      Single point for unit conversion (mmHg / mL / min / mol)
model/gas_empirical.py        v0 empirical gas model
model/gas_kinetics.py         v1.2 multi-component diffusion + cap→sphere interface
model/calibrated_params.py    Loader for calibrated Q_i
model/aqueous.py              Aqueous dynamics (Goldmann) + equilibrium IOP
model/globe.py                Ocular rigidity / compliance (Friedenwald)
model/simulator.py            Integration of the coupled ODEs
io_eye/eye_geometry.py        Axial length → volume, initial-condition generation
validation/benchmarks.py      Literature benchmark constants / pass criteria
validation/thompson_fit.py    Calibration of mass-transfer coefficients Q_i
tests/                        pytest (invariants + benchmarks + emergence)
```

See [REFERENCES.md](REFERENCES.md) for the full bibliography.

---

## License

MIT License — see [LICENSE](LICENSE).
The code may be freely used and modified (without warranty). As stated above, clinical use is not intended.
