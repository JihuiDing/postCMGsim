# Estimating Exceedance Probability of Induced Seismicity

**A synthesis of methods and workflows**
Compiled 2026-08-25 from 28 primary sources (peer-reviewed papers, USGS/IEAGHG/DOE technical reports, and software documentation).

**How to read the confidence tags:**
- **[V]** — claim survived 3-vote adversarial verification (independent agents tried to refute it and failed), 3–0 or 2–1
- **[E]** — extracted verbatim from the source with a supporting quote, but not put through the refutation panel
- **[?]** — extracted, but verification could not be completed

---

## 1. What "exceedance probability" actually means here

Plain version: you want a number like *"there is a 2% chance that shaking at this town exceeds 0.1 g in the next year"* or *"there is a 5% chance of an earthquake bigger than magnitude 3 during this injection stage."*

Those are **two different questions**, and the literature computes both, often confusingly under the same name:

| Type | Quantity | What it's used for |
|---|---|---|
| **Ground-motion exceedance** | P(shaking intensity > threshold) in a time window | Building codes, damage/nuisance risk, regulatory hazard maps |
| **Magnitude exceedance** | P(M > M\*) in a time window | Traffic-light thresholds, permit conditions, operational go/no-go |
| **Fault-slip exceedance** | P(a mapped fault reactivates) given pressure change | Site screening before you ever inject (common in CO₂ storage) |

The third is a *precursor* to the other two: it tells you whether a source exists at all, not how big or how often. Several CO₂ projects stop at this level because they have no seismic catalogue to work with. **[E]**

**Vocabulary you'll hit constantly:**
- **PSHA** — Probabilistic Seismic Hazard Analysis. The standard machinery for computing ground-motion exceedance, invented for natural earthquakes.
- **PSRA** — the same thing plus damage/loss (Probabilistic Seismic Risk Analysis).
- **Gutenberg–Richter (G-R)** — the empirical law that small earthquakes vastly outnumber big ones: log₁₀N(≥M) = a − bM. The **b-value** is the slope (higher b = relatively more small events); the **a-value** is overall productivity.
- **Mmax** — maximum magnitude, the point where you truncate G-R. The single most contested parameter in the whole field.
- **Aleatory** uncertainty = irreducible randomness (which fault ruptures, scatter in ground motion). **Epistemic** uncertainty = ignorance you could in principle reduce (which model is right, what b-value is). They are propagated differently — see §8.

---

## 2. The canonical workflow, and the four things you must change

### The base chain (six steps)

The field has converged on one skeleton, stated near-identically in the CO₂-storage review (White & Foxall 2016) and the EGS guideline (Grigoli et al. 2024): **[V]**

1. **Source characterisation** — where can earthquakes happen? (geometry)
2. **Occurrence rate** — how often, and with what magnitude distribution?
3. **Ground motion model** — given a magnitude and distance, what shaking arrives?
4. **Hazard curve** — integrate 1–3 → annual probability of exceeding each shaking level ← *this is the exceedance probability*
5. **Vulnerability / fragility** — given that shaking, what fraction of buildings are damaged?
6. **Risk / consequence** — convert to injury, cost, or nuisance complaints

Steps 1–4 are PSHA. Adding 5–6 makes it PSRA. **[V]**

The EGS guideline (Grigoli et al. 2024, *Reviews of Geophysics*) names the same six as: seismic source model, ground shaking intensity model (GSIM), site amplification, exposure, fragility curves, consequence curves — and recommends the **OpenQuake** engine to run it. **[V]**

### The four induced-specific modifications

**(a) There is no pre-existing catalogue.** For a natural-hazard PSHA you use centuries of seismicity. For induced seismicity, *by definition nothing has happened yet before you inject*. So source geometry and recurrence rates must come from a physical model, an analogue site, or be updated live as the operation proceeds. **[?]** *(White & Foxall 2016 — verification hit the session limit, but Grigoli et al. and the Bedretto study state the same premise independently.)*

**(b) The magnitude of concern drops.** Natural PSHA cares about ~M4.5+ (damage). Induced seismicity is regulated on **nuisance**, so the lower bound falls to about **M1.5–M2**. **[E]** This matters enormously: the Oklahoma study found that minimum magnitude strongly controls short-term hazard while Mmax barely does (§6). **[V]**

**(c) Rates are non-stationary in time and space.** They track the injection schedule. So the time- and space-dependence must be threaded through the *entire* chain, not bolted on at the end. **[E]**

**(d) Forecast windows shrink from decades to months.** The Oklahoma implementation states hazard forecasts are only meaningful for **6–24 months**, versus 50 years for national models, and reports only annual exceedance rates ≥ 0.01 for that reason. **[V]** The USGS one-year model made the same call: a **1-year** forecast, not the standard 50-year. **[V]**

### Worked example: USGS 2016 one-year CEUS forecast

The canonical demonstration that you can bend classical PSHA to this purpose. **[V]**

- Kept the Frankel-style **smoothed-seismicity + truncated G-R** framework unchanged
- Substituted very short catalogue durations (**1-year, 2-year, 36-year** branches) for the long-term catalogue
- Assumed rate stationarity *over the forecast year only*
- Output: ground motion with 1% probability of exceedance **in 1 year** reaches **0.6 g** in northern Oklahoma / southern Kansas, ~0.2 g in Raton Basin, central Arkansas, north-central Texas — more than **3× the induced-free 2014 national model** in places. Annual chance of MMI VI+ was **5–12%** in north-central Oklahoma. **[E]**
- "Chance of damage" was computed as the average of two exceedance probabilities at fixed thresholds: **PGA 0.12 g** and **1-Hz spectral acceleration 0.10 g**. **[E]**

---

## 3. Family A — Statistical models (the seismogenic index lineage)

**The idea in one sentence:** don't model the rock physics; find the empirical constant that says *how many earthquakes this particular reservoir gives you per cubic metre of fluid*, then extrapolate with Gutenberg–Richter.

### The seismogenic index Σ

The workhorse formulation (Shapiro; implemented operationally by Király-Proag et al.):

```
log₁₀ N(≥M, t) = log₁₀ Qc(t) − b·M + Σ
```

where `Qc(t)` is cumulative injected volume and **Σ** is the seismogenic index — "the seismogenic response of rock to a unit volume of fluid." **[E]** Higher Σ = higher induced hazard, and Σ is designed to be comparable across tectonic settings and across planned operations. **[E]**

Key properties from Shapiro (2018): **[E]**
- Originally injection-only, now extended to **fluid production/extraction**, where **poroelastic coupling becomes a first-order control** rather than a correction
- Derived analytically for two end-members: a point source in an infinite poroelastic continuum (≈ EGS stimulation) and a poroelastic layer in an elastic half-space (≈ large-area hydrocarbon reservoir)
- **Σ calibrated from production-induced seismicity transfers to forecast injection-induced seismicity and vice versa** — practically useful when your site has depletion history but no injection history

**Operational implementation** (Király-Proag et al. 2016, "Induced Seismicity Test Bench"): Σ and b re-estimated from each learning period **every six hours**, injected volume forecast from the planned schedule, post-shut-in decay from Langenbruch & Shapiro (2010) with the Omori decay exponent *p* **forced to 2** whenever it can't be estimated or comes out below 2. **[E]**

### The Mignan closed-form exceedance criterion

This is the cleanest bridge from a statistical model to an actual traffic-light number. Mignan et al. (2017, *Scientific Reports*): **[E]**

Probability of an event at or above a "safety magnitude" `m_saf` is Poisson:

```
Pr(m ≥ m_saf) = 1 − exp{ −10^(a_fb − b·m_saf) · [ V(t_si) + τ·V̇(t_si) ] } = Y
```

- `a_fb` = activation feedback (a volume-normalised productivity, seismogenic-index-like)
- `τ` = post-injection relaxation time (captures **trailing seismicity** after shut-in)
- `V(t_si)`, `V̇(t_si)` = injected volume and rate at shut-in

**Then invert it.** Instead of picking a red-light magnitude by expert judgement, *solve* for the threshold that satisfies your safety criterion:

```
m_th = (1/b)·log₁₀[ Y − 10^(a_fb − b·m_saf)·τ·V̇(t_si) ] + m_saf
```

Updating `m_th` on the fly guarantees the safety criterion is met. **[E]** The safety anchor is actuarial, not seismological: **individual risk ≤ 10⁻⁶ per year**, the standard threshold for hazardous installations — explicitly replacing heuristic magnitude thresholds that vary between ~0.5 and 4 across jurisdictions with no link to any risk standard. **[E]**

Only **three parameters**. Fitted to six fluid-injection experiments: 0.8 ≤ b ≤ 1.6, −2.8 ≤ a_fb ≤ 0.1 m⁻³, 0.2 ≤ τ ≤ 20 days. **[E]**

Uncertainty handled by: re-estimating in a moving window of constant event count (n = 100); using **generic priors from previous experiments** before site data accumulate; Monte Carlo simulation of the Basel sequence. Important caveat the authors themselves report — the safety target is met **on average across realisations, not in every individual sequence**. **[E]**

### ETAS (Epidemic Type Aftershock Sequence) with Bayesian parameters

Used at St Gallen for very short-horizon forecasting: **[V]**
- 24-hour forecasts of the rate of events above completeness magnitude Mct = −0.5
- Parameter uncertainty propagated by **integrating the ETAS rate over the joint posterior of θ = [b, K, a, c, p, d, q, Kt, Kr]** using adaptive MCMC
- Output: 2nd–98th and 16th–84th percentile bands on forecast counts per 0.001° × 0.001° cell
- Validation: observed counts fell within ±2σ of the mean estimate

### Bedretto: seismogenic index inside a full PSHA logic tree

The most recent full worked example (Solid Earth 2025), at the Bedretto Underground Laboratory: **[V]**
- Framework called **PISHA** (probabilistic *induced* seismic hazard analysis), updated sequentially as stimulation experiments proceed
- Rates from a seismogenic-index-type formulation (Mignan et al. 2017): magnitude-frequency rate scales with injected volume via `a_fb` combined with b
- Explicit design goal: *"PSHA must appropriately represent the uncertainties in the assessment and represent the range of technically defensible interpretations"* — i.e. span the defensible range, don't produce a best estimate

**The headline uncertainty result, and it's sobering:** a priori hazard estimates span **2–3 orders of magnitude between the 10th and 90th percentiles** for deep injections (up to **6 orders** if you include underground-lab analogues). That collapses substantially only once site-specific ground-motion models and observed seismic response are folded in — after which P(PGV > 30 mm/s) < 0.001 for 100 m³ injections at the second update. Results feed a TLS with fixed PGV thresholds of **0.5, 2.5, 15 mm/s**. **[V, 2–1]**

**Practical implication:** before you inject, a statistically-grounded hazard number is uncertain to within a factor of ~1000. It becomes decision-grade only after the first stimulation gives you site data. This is the single strongest argument in the corpus for staged, adaptive workflows over one-shot pre-operational assessments.

---

## 4. Family B — Physics-based / geomechanical models

**The idea in one sentence:** simulate the pressure and stress the injection actually creates, and compute whether specific faults reach failure.

### B1. Fault Slip Potential (FSP) — the CO₂-storage screening standard

Originated as a Stanford SCITS/ExxonMobil tool, publicly released 2017 (FSP 2.0); stewardship moved to UT-Austin's Center for Injection and Seismicity Research in 2023; web-based FSP 3.x now maintained with the BEG TexNet program. **[E]**

**Method:** Mohr–Coulomb frictional failure analysis of mapped faults + pore-pressure modelling of injection → *probability of slip on each mapped fault*. Explicitly probabilistic, not deterministic. **[E]** Pressure input either imported from an external reservoir model or generated internally by a semi-analytic confined-aquifer solution with linear superposition of multiple wells. **[E]**

**How the probability is produced:** Monte Carlo sampling over geomechanical parameters, **2000 realisations per fault scenario**; each realisation is a point on a CDF giving slip likelihood as a function of pore-pressure change. **[E]**

**Parameters randomised in a representative CO₂ study:** 20 near-vertical faults (dip 75–85°, strike ±10°, length 1–4 km in a 10×10 km area); ±0.1 psi/ft on Sv, Shmin and pore pressure; ±10° on SHmax azimuth; ±0.1 on friction coefficient and A-φ. **[E]**

**Making it time-dependent:** couple a **3D hydromechanical reservoir simulation** (in the cited study, a CMG model: 18 years of injection at ~0.68 Mt CO₂/yr plus 100 years of post-injection monitoring) to the Monte Carlo CDF. Map the pressure change at each fault midpoint through time onto the CDF. Result: slip potential on the closest-to-failure faults rises from **~0.3 to ~0.5** (a ~20 percentage-point absolute increase) for maximum pressure changes of ~200–400 psi. **[E]**

**Sensitivity (tornado diagrams):** in-situ **stress magnitude and orientation dominate** — SHmax azimuth for poorly-oriented faults, Shmin gradient for well-oriented ones. Fault dip and vertical stress gradient contribute little. Derived operational criterion: keep pressure below **500 psi** within 1–2 km of the injector, versus the ~600–1000 psi needed to slip well-oriented faults. **[E]**

**Scope limits that bound its use in exceedance workflows:** FSP assumes faults intersect the injection interval, and **neglects out-of-zone pressure effects and poroelastic stress transfer**. **[E]**

### B2. Rate-and-state seismicity coupled to poroelasticity

Segall & Lu (2015, *JGR*) — the most physically complete route to a time-dependent *rate*, and it produces a genuine magnitude-exceedance expression: **[E]**

```
R(M,t) = k · 10^(−bM) · P_in( r_s(M) / √(4ct) ) · ∫ R(x,t) dV
```

Three factors: nucleation rate × G-R probability that nucleation lands on a fault segment big enough for magnitude M × **P_in**, the probability that the rupture segment lies *entirely inside* the pressure-perturbed volume (radius growing as √(4ct)).

Notable results: **[E]**
- Poroelastic coupling makes the *sign* of the seismicity-rate change **fault-orientation dependent**
- **Abrupt shut-in can produce a local sharp INCREASE in seismicity rate** on faults whose slip was inhibited during injection; tapering the flux reduces the spike. The authors state this conclusion is independent of the Dieterich model assumptions.
- In the low-ambient-shear-stress limit, **Mmax is controlled by the perturbed volume size** and grows with injection duration, producing a *time-advancing rollover* in the frequency-magnitude distribution (apparent b > 1 at intermediate magnitudes). In the high-stress limit, Mmax is limited only by fault dimensions and tectonic prestress.
- Triggering-front scaling t ~ r²/(c·f_c) persists even with poroelasticity and nucleation included, and the model predicts a **finite, magnitude-dependent quiescent period** before events of a given size become observable

The shut-in spike is operationally important: it's a mechanism by which the risk *rises* at exactly the moment a traffic-light system tells you to stop.

### B3. Stochastic poromechanical analysis (SPA) — Pohang

The clearest demonstration that physics-based models can yield an honest a priori exceedance probability, but only if you make them stochastic. **[E]**

- Couples pore-pressure/poroelastic physics with Monte Carlo sampling of uncertain geomechanical parameters
- **30,000 valid realisations per slip-pattern scenario**, kept tractable by using analytical solutions (Theis pressure + poroelastic stress) rather than full numerical simulation
- Magnitude derived **geometrically** from the simulated size of the reactivated slipping patch
- Result for the 2017 Pohang Mw 5.5: **7–15% a priori exceedance probability** (7% reverse-slip-dominated, 15% strike-slip-dominated); ~32–34% probability of exceeding Mw 3.0; maximum possible induced magnitude near **Mw 7.0**
- **The deterministic version of the same model predicts a stable fault** and fails to reproduce the mainshock entirely

That last point is the methodological punchline of the whole family: a single best-estimate parameter set can tell you an earthquake is impossible when the honest answer is "one chance in ten."

### B4. Deterministic THM screening (the CO₂-storage counter-tradition)

Vilarrasa et al. (2019, *Solid Earth*) argue for coupled thermo-hydro-mechanical predictive models plus pressure management, and contain **no PSHA, no seismogenic index, and no exceedance probability at all**. **[E]** Worth knowing because it's influential in the CO₂ community and represents a genuine methodological disagreement, not an oversight.

- Uncertainty propagated as a **deterministic bounding range**, not a distribution: fault-property uncertainty maps to a range of critical pressure increases; operational limits are set by not exceeding it
- Screening metric: mobilised friction coefficient **μ_mob = tan(φ'_mob)** from measured principal stresses and pore pressure, compared to a reference fault friction of 0.6. All eight tabulated CO₂ sites fall below: In Salah 0.48, Weyburn 0.50, Otway 0.41, Snøhvit 0.49, St. Lawrence Lowland 0.54, Decatur 0.51, **Pohang 0.27**
- Mmax treated as geometry-controlled (M scales with shear modulus × rupture area × mean slip), bounded by fault constitutive behaviour: brittle slip-weakening faults can produce M > 4, ductile faults give aseismic slip or swarms
- Argues epistemic uncertainty must be reduced by treating **operation itself as continuous characterisation**, because the injection-affected region grows as √t and eventually exceeds what any pre-operational test could sample

> ⚠️ **Source-attribution error caught during extraction:** the URL `se.copernicus.org/articles/10/871/2019` was listed in the search results as Verdon & Stork. It is not — it is Vilarrasa, Carrera, Olivella, Rutqvist & Laloui (2019), "Induced seismicity in geologic carbon storage," *Solid Earth* 10, 871–892. **[E]**

---

## 5. Family C — Hybrid, Bayesian, and ensemble approaches

The 2016 CO₂ review framed the field as two families (statistical vs. simulation-based geomechanical). **[?]** Everything since has been about combining them.

### Hybrid physical-statistical: Langenbruch et al., Oklahoma/Kansas

Documented in the IEAGHG state-of-knowledge report: **[E]**
- **MODFLOW** 3-D hydrogeologic model of 20 years of injection → pore-pressure *rates*
- P(triggering M ≥ 3) computed as the **ratio of two histograms**: pressure rates that triggered observed earthquakes ÷ monthly pressure rates at all model seed points
- Output: **1-year M ≥ 4 exceedance-probability maps** over 1257 km² (20-km radius) areas

This is a neat trick — it uses physics to make the spatial field and statistics to calibrate the trigger, avoiding both the "no catalogue" problem and the need for a full rupture model.

### Hybrid at a CO₂ site: Illinois Basin–Decatur

Zhang et al. (2025, *Engineering Geology* 362:108536): **[E]**
- Coupled hydromechanical fault-stability model (slip tendency index + Coulomb failure stress evolution) **combined with** seismogenic index and magnitude probability distributions on the same site
- Result: **Σ ≈ −4**, translating to **>50% probability that induced magnitude stays below M 2.27**
- Used for injection-schedule decision support: a controlled **ramp-up** shifts the magnitude distribution toward low magnitudes, whereas high initial rates increase fault instability relative to constant or gradually-increasing schedules
- Physics finding: permeability-controlled **pore-pressure diffusion dominates** reactivation for both near-well and far-field faults; poroelastic stress produces only localised slip and **can stabilise faults during shut-in**

Note this contradicts Segall & Lu's shut-in spike (§B2) — the sign of the poroelastic shut-in effect is site- and orientation-specific, not universal. Treat any general claim about shut-in as unresolved.

### Bayesian time-varying rates: the Oklahoma framework

The most complete Bayesian implementation, and the best template if you want a running hazard estimate: **[V]**

- Substitute a **time-dependent rate λ(Mn ≥ mmin)(t)** into the standard Cornell hazard integral, then time-average hazard over the forecast window
- Rates from a **Bayesian multiple-change-point model** (sequential bisection to detect multiple change points) fit to a **Reasenberg-declustered** M ≥ 3 catalogue on a 0.1° grid, **updated every 6 months**
- Exceedance computed by **stochastic Monte Carlo event-set simulation — 10,000 one-year simulations** in OpenQuake — rather than site-by-site numerical integration. This is faster *and* permits **spatially correlated ground-motion fields**, which you need for regional loss exceedance curves. **[V]**

The change-point model is the key move: rather than assuming a rate or fitting a smooth trend, it lets the data say "the rate stepped up in March 2015 and stepped down in November 2016."

### Bayesian extreme-value + ETAS for Mmax

The IEAGHG report describes combining Bayesian inference with **extreme value theory** and an ETAS occurrence model specifically to constrain the magnitude of the largest expected earthquake in a future interval. **[E]**

### Non-Poisson recurrence (St Gallen)

Most implementations assume Poisson occurrence. St Gallen tested it and found it wrong: **[V]**

1. Fit candidate inter-arrival-time PDFs **per project phase**: exponential (Poisson), Brownian Passage Time, Weibull, gamma
2. Select best by **Kolmogorov–Smirnov test**
3. Convert the conditional time-dependent probability to an **equivalent annual rate r = −ln(1−P)/t** and substitute for the Poisson rate α_i in the hazard integral

**4 of 6 phases were best fit by non-Poisson models** (Weibull in phases 2, 3, 5; BPT in phase 4).

Validation against withheld stations SGT03 and SGT09 at 10%, 20%, 50% exceedance over 15-day / 1-month / 2-month windows: **Poisson-based hazard systematically overpredicted observed PGVs**, while Weibull/BPT and ETAS came closer. **[E]** *(Phase-specific GMPEs of form ln PGV = a + b·M + c·ln R were refit per phase and accepted only on a one-sided F-test at 95%, cutting σ_lnPGV from the Douglas et al. reference value of 1.863 down to ~0.82–1.01.)*

The Test Bench paper independently flags the same problem: both its forecast models **and the entire testing framework** assume Poisson occurrence, the authors state explicitly that this is violated because events are not independent in time or space, and note it may itself cause forecast failures. **[E]**

### Ensembles: ORION

DOE/LLNL/LBNL open-source toolkit built specifically for **CO₂ storage** forecasting, under the NRAP and SMART initiatives. **[E]**
- Core output *is* an exceedance-probability forecast: spatially and temporally resolved P(magnitude exceeded) within a defined window
- Propagates **model-form uncertainty by ensembling three method families** — statistical, physics-based, and machine-learning — on the explicit rationale that ensembling compensates for the limitations of any single class
- Treats **Mmax as a probability distribution** derived statistically from observed seismicity, not a deterministic cap
- Couples the output to a traffic-light-style control scheme with well-level injection-change recommendations

### Forecast validation: the Induced Seismicity Test Bench

Király-Proag et al. (2016) ported the **CSEP** tectonic-forecast validation protocol to induced seismicity. **[E]** Any forecast is evaluated on event number, magnitude distribution, and 3D spatial distribution with pseudo-prospective learning/forecast periods, using likelihood-based consistency tests (**N-test, M-test, S-test**) plus information-gain-per-earthquake for ranking.

**The result is a warning.** Pseudo-prospective testing on Basel 2006 and Soultz 2004: **neither the statistical model (SaSS) nor the hybrid hydro-mechanical model (HySei) forecast the largest event (mw 3.1)**, and both systematically underpredict rate around and shortly after shut-in. That is precisely the magnitude-exceedance episode a traffic light system exists to catch. **[E]**

---

## 6. Maximum magnitude — the central unresolved fight

Mmax truncates the G-R distribution and therefore controls the tail of every exceedance curve. There is **no solved data-driven estimator.** **[?]**

### The three classical estimators

1. **Fault-dimension scaling** — rupture length/stress-drop scaling from the largest identified favorably-oriented fault (a 1 km fault ⟹ roughly M3–M4) **[E]**
2. **Shapiro et al. (2011)** — scaling to the dimensions of the pressure-perturbed volume **[E]**
3. **McGarr (2014)** — upper-bound seismic moment proportional to net cumulative injected volume **[E]**

**All three can be exceeded** when a rupture nucleating inside the pressure plume propagates beyond it. **[?]** *(Stated independently in both the 2016 CO₂ review and McGarr's own paper.)*

### McGarr (2014) in detail

```
M₀max = G · ΔV        with G = 3×10¹⁰ Pa
```

Envelopes 18 case histories across wastewater disposal, EGS, hydraulic fracturing, and scientific injection. **[E]**

**Five explicit assumptions** — worth checking against your site before using it: **[E]**
1. Hydraulic communication with a seismogenic (often crystalline basement) formation
2. Pre-existing well-oriented faults stressed to within one stress drop of failure
3. Full saturation before injection
4. Gutenberg–Richter with **b = 1**
5. Induced events confined to the pore-pressure-perturbed volume

Because it assumes **all injection-induced deformation is seismic**, it is an upper bound, not an expected value. **[E]**

**It is b-value dependent.** With B = b/1.5 constrained to 0.5 ≤ B < 1: **[E]**
```
M₀max = [(1−B)/B] · [2μ(3λ+2G)/3] · ΔV
```
Higher b lowers Mmax. Demonstrated at Paralana, Australia: b = 1.32 (B = 0.88) predicts M2.7 against the observed M2.5, versus M3.1 predicted using b = 1.

McGarr's own stated caveats: the dominant epistemic uncertainty is rupture propagating **outside** the perturbed volume, so the bound is **"plausible but not an absolute physical limit"** for hazard use. **[E]** And although the bound grows monotonically with cumulative volume, **the timing of the largest event is decoupled from injection history** — the M3.9 Ashtabula, Ohio event occurred ~7 years after injection ceased, 6.5–7 km from the well. **[E]**

### The three attacks on volume-based Mmax

**Attack 1 — it's empirically non-conservative.** The 2017 Pohang Mw 5.5 **exceeded McGarr's predicted maximum moment by roughly three orders of magnitude**. **[E]** At St Gallen, McGarr's bound gave Mmax 2.6–2.9 for the phase that actually produced an observed **ML 3.5**. **[V]**

**Attack 2 — the missing parameter is antecedent stress.** The Pohang paper proposes: **[E]**
```
M₀max = ½(1 − c)·G·ΔV
```
where **c** is the antecedent stress ratio locating initial shear stress within the stress-drop window. c = 0.5 recovers the classic case; **c > 0.5 (faults already primed near failure, up to c ≈ 0.999) permits maximum moments orders of magnitude larger for the same injected volume**. The implication is a change of practice: constraining Mmax requires **characterising the pre-injection in-situ stress state on the target fault**, not just monitoring or capping volume. A lab kinematic discriminator exists (cumulative moment exceeds the classic bound once fault shear slip reaches ~2× fault dilation, shown in double-direct-shear experiment CSS1 at 60% of peak shear strength) — but the authors concede it is **not practically observable in real fault systems**. **[E]**

**Attack 3 — the calibration set was the wrong population.** Bommer & Verdon (2024) point out McGarr calibrated on a compilation that was **predominantly *triggered* seismicity**, while the bound is only valid for industrially **driven** seismicity. **[E]**

### The "no bound at all" position

van der Elst et al. (2016, *JGR*) argue from the data that: **[E]**
- Observed maximum magnitudes at injection sites are **statistically consistent with random sampling from an unbounded G-R distribution** — no evidence for an injection-imposed bound distinct from tectonic earthquakes
- Therefore induced earthquakes should be assigned the **same Mmax as tectonic earthquakes**, and the largest event observed to date must never be taken as the site maximum
- Injected volume controls the **number** of induced earthquakes but **not the total released seismic moment**, which falsifies deterministic volume-based bounds and supports rate-based (seismogenic-index-style) forecasting combined with a tectonic magnitude distribution
- The largest observed event scales with the **logarithm of the total number** of induced events — the signature of extreme-value sampling from a G-R population, and a testable basis for magnitude-exceedance forecasting
- The largest event occurs at a **random position in time** within a sequence — which directly undercuts traffic-light systems that assume progressive escalation

### The rebuttal to "no bound at all"

Bommer & Verdon (2024) cite **Watkins et al. (2023)**: the observed distribution of magnitude *jumps* is **statistically significantly smaller than random G-R draws**, supporting gradual magnitude growth and hence TLS effectiveness. **[E]** Verdon & Bommer (2020) independently found from a global hydraulic-fracturing compilation (USA, Canada, UK, China) that magnitude jumps are **in most cases below 2 magnitude units**, and that a statistical test of where the largest event falls within each sequence **rejects random timing at ~4% significance** — largest events tend to fall later in injection sequences. **[E]**

**This is a live, unresolved disagreement.** Van der Elst says magnitude timing is random and Mmax is tectonic; Verdon/Bommer/Watkins say growth is gradual and Mmax is bounded below tectonic. Which you believe determines whether a traffic light system is a safety control or a comfort blanket. **A defensible workflow puts both on a logic tree with weights rather than picking one.**

### The MmaxE / MmaxT distinction

Bommer & Verdon (2024)'s most useful contribution for practical work: **Mmax is two different quantities** that enter the calculation differently. **[E]**

| | Meaning | Where it enters |
|---|---|---|
| **MmaxT** | Physical upper bound | Truncates the G-R recurrence relation |
| **MmaxE** | *Expected* largest event magnitude | What actually controls hazard and risk over a short operation |

For short-term operations (hydraulic fracturing, EGS), especially under a TLS, **MmaxE is appreciably lower than MmaxT** and is the one that matters. Conflating them is a common source of over-conservatism.

**Their recommended structure — a two-node logic tree:** **[E]**
- **Node 1:** driven-only vs. driven-plus-triggered seismicity, with the triggered branch weight set equal to the **probability of tectonic triggering**
- **Node 2:** separate Mmax ranges for each scenario
- **Warning:** Mmax distributions taken from natural-seismicity PSHA are generally **inappropriate for the triggered branch**, because injection/extraction wells are shallow

### Is hazard sensitive to Mmax? Two answers, and they're reconcilable

- **Oklahoma:** Mmax has **little effect above ~M6** at the annual exceedance rates of interest, because larger events are too infrequent to contribute at those rates. **Minimum** magnitude matters strongly. Recommendation: mmin ≤ 3 and Mmax ≥ 6 for Oklahoma. (b = 1.3 baseline, varied 0.8–1.4.) **[V]**
- **Bommer & Verdon:** the standard practice of assigning a **narrow range of large Mmax values** in a logic tree manufactures the conclusion that hazard is insensitive to Mmax. Use a **broad** Mmax distribution — e.g. Bayesian updating of a prior with observed seismicity, per USNRC (2012) — and hazard curves **vary significantly**. **[E]**
- **Eagle Ford TLS design:** in sensitivity analysis of the risk model inputs, **forecast maximum magnitude dominates the outcome**, making Mmax the leading uncertainty. **[E]**

**These are consistent.** Mmax is insensitive at *high* annual exceedance rates (frequent, moderate shaking) and highly sensitive at *low* rates and in *risk*/consequence calculations, where the tail is what does the damage. Know which regime you're in before quoting either result.

### A closed-form magnitude-exceedance estimator for Mmax

St Gallen offers a usable formula from a lower-truncated exponential (G-R) magnitude PDF: **[V, 2–0]**

```
P(Mmax > M* | Mmax_obs < Mmax < Mmax_TOT) =
    [ exp(−β·M*) − exp(−β·Mmax_TOT) ] / [ exp(−β·Mmax_obs) − exp(−β·Mmax_TOT) ]
```

They adopt the **90% non-exceedance value** with Mmax_TOT = 5.0 (justified by two historical Mw ≥ 4.0 events), giving phase Mmax of **1.4–4.7** — versus McGarr's 2.6–2.9 for the phase that produced the observed ML 3.5.

### Expert elicitation: the Groningen precedent

Groningen's Mmax input was produced as a **logic tree formulated by a dedicated expert panel**, informed by two international workshops (2016 and 2022) devoted solely to the Mmax question — formal elicitation feeding a probability-weighted tree, not a deterministic estimate. **[E]**

The authors (Bommer, van Elk, Zoback) argue the result was **overly conservative on two counts**: the size of events inducible within the reservoir, and — in their view most importantly — the assigned likelihood of **triggering larger tectonic earthquakes on faults outside the reservoir**. Their methodological argument: assigning non-zero weight to Mmax values that are **not physically realisable** injects unwarranted conservatism, so **physics-based feasibility should bound the tails of Mmax distributions**. **[E]**

The stakes are not academic. In October 2023 the Dutch government **closed the Groningen field**, justified by the potential threat of larger earthquakes — despite the largest observed induced event being **M 3.5** (2012), with production from 1963 and induced seismicity from 1991. **[E]**

---

## 7. Ground motion models — the acknowledged weak link

Named explicitly as a specific weak point for induced-seismicity exceedance calculations: **[E]**

- Most empirical GMPEs **do not extend below M4.5**
- **NGA-West2** reaches only M3.0, with minimal data below M6 at distances under 5 km
- The geothermal-specific GMPEs of **Douglas et al. (2013)** for M < 3.5 carry large aleatory uncertainty **dominated by site-to-site variability**
- Suggested site-specific alternative: **empirical Green's functions**

You are extrapolating GMPEs into exactly the magnitude–distance corner where induced seismicity lives (small M, very short R) and where there is least data. The St Gallen result quantifies the payoff from fixing this: refitting phase-specific GMPEs cut σ_lnPGV from Douglas's 1.863 to **~0.82–1.01** — a factor-of-2 reduction in the dominant aleatory term. **[E]**

A related, under-appreciated problem: **magnitude estimates themselves are uncertain.** Velocity-model and station site-effect uncertainty can shift induced-event local magnitude by **more than a whole magnitude unit** — enough to move an event between TLS zones and reverse the operational decision. **[E]**

---

## 8. How uncertainty is propagated

### The standard split

| | What it is | How it's handled |
|---|---|---|
| **Aleatory** | Irreducible randomness | Embedded **inside** the hazard integral as PDFs — GMPE sigma, focal-depth PMF, magnitude PDF |
| **Epistemic** | Model/parameter ignorance | **Weighted logic tree** over alternative models; each end branch is a full hazard curve; results reported as a **distribution** (mean, median, fractiles) |

Concrete example of aleatory handling: the Oklahoma model embeds focal depth as a probability mass function directly in the integral — depths 3/4/5/6/7 km with probabilities 0.05/0.15/0.60/0.15/0.05. **[E]**

### Logic-tree structures actually used

**USGS 2016** — **two parallel trees**, one for sites inside induced-seismicity zones and one for sites outside. Branch levels: **[V]**
1. Catalogue / 2014 NSHM fault and area sources
2. Induced-vs-natural classification and rate estimation
3. Catalogue duration (1 / 2 / 36 years)
4. Smoothing distance
5. Mmax
6. Ground motion model

Two features worth copying:
- Branch weights assigned by **degree of acceptance in the scientific community**, not by data fitting
- **Zone classification is itself weighted** — 100 / 80 / 50 / 0 percent on the "induced" branch, from expert elicitation at workshops and with state geological surveys. That is, "is this zone induced?" is treated as an uncertain proposition, not a binary label.

**Grigoli et al. EGS guideline** — includes an explicit **b-value branch that equally weights the locally calibrated b-value against the global tectonic average of 1.0**, with a **sensitivity-analysis trigger when b-value estimation methods disagree by more than 0.2 units**. **[V]** That trigger is a nice, concrete, auditable rule.

**Bedretto** — branch weights from **expert elicitation in which scientists judged the similarity of analogue case studies to the target site** by rock type, depth, stress state, injected volume, and data reliability. **[V]** This is the standard answer to "no catalogue exists yet": borrow other sites, weighted by how much they resemble yours.

**Bommer & Verdon Mmax tree** — two nodes, as in §6.

### The Monte Carlo alternative

Increasingly the integration itself is Monte Carlo rather than numerical: **[V]**
- Generate many random ground-motion IM fields per simulated rupture; count exceedances empirically (Grigoli et al.)
- 10,000 one-year stochastic event-set simulations in OpenQuake (Oklahoma)
- 30,000 realisations for stochastic poromechanics (Pohang)
- 2,000 realisations per fault scenario for FSP

Monte Carlo also unlocks **spatially correlated** ground-motion fields, without which you cannot compute a portfolio loss exceedance curve. **[V]**

### Model-form uncertainty

The hardest kind — "what if my whole model class is wrong?" Two treatments in the corpus:
- **Weighted combination of multiple models** rather than a single one, inside the Adaptive Traffic Light architecture **[E]**
- **Ensembling across method families** (statistical + physics + ML) in ORION **[E]**

### Full Bayesian UQ

Bayesian 3D non-linear traveltime tomography (reversible-jump MCMC) jointly inverting velocity structure and hypocentres, yielding **full posterior distributions of local magnitude** instead of point estimates. **[E]**

The cost is brutal and worth stating plainly: **adding a single new event required 14,880 CPU-hours** on ARCHER. The authors state this is **computationally prohibitive for real-time monitoring**. **[E]**

---

## 9. Traffic light systems — where the number becomes a decision

A TLS is the operational endpoint of most of these workflows: green = continue, amber = reduce, red = stop.

### The evolution: heuristic → risk-normalised → probabilistic

**Generation 1 — heuristic magnitude thresholds.** Fixed numbers varying between roughly 0.5 and 4 across jurisdictions with **no link to any standard risk criterion**. **[E]**

**Generation 2 — risk-normalised thresholds.** Schultz et al. (2021, *JGR*; and *Science* 2021) chain probabilistic Mmax through formation depth → site amplification → GMPEs → felt/damage shaking tolerances → **population exposure**, simulating the spatial distribution of nuisance and damage impacts. This is a **forward risk calculation, not a pure hazard calculation**. **[E]**

Results: **[E]**
- Risk-normalised red-light thresholds span **Mw 2.0 to Mw 5.0** across a single play, rather than one play-wide cutoff (Eagle Ford: greater in the northeast, smaller in the southwest)
- **Population density is the dominant control** on spatial variation — more than geology or ground motion
- Nuisance and damage impacts have **different spatial correlation lengths** — hundreds of km for nuisance vs. tens of km for damage — so the two must be mapped and thresholded **separately**
- Back-analysis across eight North American shale plays (Horn River, Duvernay, Montney, Utica, Marcellus, SCOOP & STACK, Eagle Ford, Delaware Basin) shows regulators' de facto red-light magnitudes are **better explained by nuisance tolerance than by damage risk**

**Generation 3 — thresholds derived by inverting an exceedance criterion.** Mignan et al. (§3): solve for m_th such that P(m ≥ m_saf) meets an actuarial target, update live. **[E]**

**Generation 4 — probabilistic TLS with ambiguous zones.** Replace fixed thresholds with **confidence-level-dependent ambiguous bands** where an event cannot be assigned to a level with the required probability. At Preston New Road with α = 0.8: ambiguous ranges of **−0.036 < ML < 0.035** (zone A) and **0.46 < ML < 0.53** (zone B). **[E]** Applied retrospectively, this reclassifies the 19 October event from the operator's **green to amber under all three TLS variants tested** — implying operations should have been curtailed earlier. **[E]**

### The magnitude-vs-ground-motion argument

Verdon & Bommer (2020) argue **magnitude, not PGV, should be the control parameter**, and that TLS design rests on two testable assumptions: **[E]**
1. Gradual magnitude evolution through green → yellow → red
2. Limited post-injection (trailing) magnitude growth

Their empirical tests of both (magnitude jumps mostly < 2 units; largest events non-randomly late) are in §6. Their design recommendation: **set red-light thresholds up to two magnitude units below the magnitude you're trying to avoid.** **[E]**

Note that Bedretto and several CO₂ protocols use **PGV** thresholds instead (0.5 / 2.5 / 15 mm/s at Bedretto), so this is not settled practice. **[V]**

### Trailing seismicity — the risk that survives shut-in

Quantified by Verdon & Bommer: **[E]**
- Post-injection events increase the maximum observed magnitude in roughly **one quarter of hydraulic fracturing cases**
- Largest observed post-shut-in increase: **1.6 magnitude units**
- Mostly within a few days of the end of injection
- Their estimate uses G-R with b = 1 plus an assumption that **~20% of events occur after injection stops** — i.e. from expected extra event count, **not** from a physics-based fault model

Mignan's τ parameter (0.2–20 days) is the statistical-model equivalent. **[E]** Segall & Lu's shut-in rate spike is the physics-based mechanism. **[E]**

### The case against relying on TLS

Three independent sources, and this should be read carefully by anyone designing one:

- The IEAGHG report: TLS are **conditionally valid, not universally reliable**. They fail when a large earthquake is **not preceded by precursory smaller events**, or when **operations have already ceased**. They require both an escalating foreshock sequence *and* mitigation fast enough to act before escalation. **[E]**
- White & Foxall: potentially ineffective because **the largest event of a sequence may occur very early with little or no precursory microseismicity**; operators should build **damage mitigation into project design** rather than rely on reactive control. **[E]**
- The Test Bench result: on Basel and Soultz, **neither forecast model caught the largest event**, and both underpredicted rate around shut-in. **[E]**
- Van der Elst: the largest event occurs at a **random position in time** within a sequence. **[E]**

Against which: Watkins et al. (2023) and Verdon & Bommer's magnitude-jump statistics support gradual growth. **[E]** Unresolved (see §6).

---

## 10. CO₂ storage specifics

CO₂ storage sits awkwardly in this literature because it has the **longest** operational horizon and the **thinnest** seismic data.

**The characteristic pattern in the reviewed CO₂ case studies:** you get a *fault-slip* exceedance probability, not a ground-motion or magnitude one.

At the CO₂ site studied with FSP + CMG coupling, statistical seismicity characterisation was limited to a **regional** catalogue fit (b = 0.73, Mc ≈ 1.7, ~1000 events since 1980, largest M4.8, max-curvature fit in ZMAP, Gardner–Knopoff declustering leaving < 100 mainshocks). The authors state explicitly that **sparse station coverage prevented statistically significant G-R mapping near the plume, so no seismogenic-index or magnitude-exceedance forecast could be computed.** **[E]** They substitute **spatiotemporal seismicity density** (background ~0.1–0.2 events/month/km²) as the real-time metric, and state that **a dense pre-injection near-well network is a prerequisite** for the statistical route. **[E]**

**Practical TLS design at CO₂ sites** is correspondingly multi-metric rather than magnitude-only. The four-state protocol adapted from the **ADM Decatur, IL EPA Class VI application**: **[E]**

| State | Magnitude | Seismicity rate (events/month/km²) | Monitoring-well ΔP vs. simulated model |
|---|---|---|---|
| Green | < M1.5 | < 0.2 | — |
| Amber | M1.5–2.0 | 0.2–0.4 | 1% above model |
| Orange | > M2.0 (or > M1.5 on a mapped fault) | > 0.5 | 5–10% above model |
| Red | > M3.0 (or > M2.0 on a mapped fault) | > 1 | > 10% above model |

Plus a **PGA > 0.1 g** review trigger. Note the third column: pressure deviation from the history-matched simulation is itself a trip condition. That is a meaningful design idea — it catches "the reservoir is not behaving as modelled" *before* it becomes "the reservoir produced an earthquake."

**Where CO₂ sites have gone further:** Illinois Basin–Decatur got to Σ ≈ −4 and P(M < 2.27) > 50% (§5). ORION was purpose-built for CO₂ exceedance forecasting (§5). And the deterministic THM tradition (§B4) is strongest in this sector.

> 💡 **Relevant to this repo:** the FSP + CMG-coupled workflow in §B1 (3D CMG hydromechanical simulation → pressure at fault midpoints → Monte Carlo slip-probability CDF → time-dependent fault slip potential) is the closest published analogue to a post-CMG-simulation seismicity workflow. Two sources in the corpus describe it — `mdpi.com/2624-795X/4/4/24` and `doi.org/10.3390/geohazards4040024` — which appear to be **the same GeoHazards paper reached by two URLs**; treat as one source, not corroboration.

---

## 11. Software and tools mentioned

| Tool | Purpose | Notes |
|---|---|---|
| **OpenQuake** | PSHA/PSRA engine | Recommended by the EGS guideline; used for the Oklahoma Monte Carlo event sets **[V]** |
| **FSP 3.x** | Probabilistic fault slip potential | UT-Austin CISR / BEG TexNet; web-based; originally Stanford SCITS/ExxonMobil 2017 **[E]** |
| **ORION** | Operational exceedance forecasting for CO₂ | DOE/LLNL/LBNL, open source, three-family ensemble **[E]** |
| **MODFLOW** | Regional pore-pressure fields | Used in the Langenbruch Oklahoma hybrid **[E]** |
| **CMG** | 3D reservoir/hydromechanical simulation | Feeds FSP in the CO₂ case study **[E]** |
| **ZMAP** | G-R / Mc estimation | Max-curvature b-value fitting **[E]** |
| **Induced Seismicity Test Bench** | CSEP-style forecast validation | N/M/S-tests, information gain **[E]** |

---

## 12. Where the field actually disagrees

Worth stating baldly, because a report that smooths these over would be misleading:

1. **Is Mmax bounded by injected volume?** McGarr yes; van der Elst no; Pohang says "yes but with an extra stress parameter that makes the bound nearly useless in the primed-fault limit." **Unresolved.**
2. **Do induced sequences escalate gradually?** Verdon/Bommer/Watkins yes; van der Elst no (random timing). This determines whether TLS is a control or a placebo. **Unresolved.**
3. **Does shut-in raise or lower seismicity?** Segall & Lu predict a poroelastic *increase* on some fault orientations; Zhang et al. find poroelastic stress *stabilises* faults during shut-in. **Site- and orientation-specific; no general rule.**
4. **Is hazard sensitive to Mmax?** Yes at low exceedance rates and in risk calculations; no at high rates. **Reconcilable but routinely conflated.**
5. **Magnitude or ground motion as the TLS control variable?** Verdon & Bommer argue magnitude; Bedretto and several CO₂ protocols use PGV. **Unsettled practice.**
6. **Probabilistic hazard analysis or deterministic pressure management for CO₂?** Vilarrasa et al. take the second position explicitly. **Genuine methodological schism.**

---

## 13. If you had to build one workflow

Synthesising what the corpus converges on, for a site with no pre-injection induced catalogue:

1. **Pre-operational screening** — physics-based fault-slip probability (FSP-style Monte Carlo over stress, friction, fault geometry), fed by a reservoir pressure simulation. Accept that the answer is a *slip likelihood*, not a hazard curve. Expect stress magnitude and orientation to dominate the sensitivity.
2. **A priori hazard** — seismogenic-index formulation with parameters from **analogue sites weighted by similarity** (rock type, depth, stress state, volume, data quality). Expect **2–3 orders of magnitude** of spread. Do not present this as decision-grade; present it as a bounding range.
3. **Logic tree** for epistemic uncertainty, with branches on: source model, b-value (local vs. 1.0), **Mmax split into driven vs. triggered nodes with MmaxE distinguished from MmaxT**, GMPE, and — if relevant — whether the zone is induced at all. Weights by structured expert elicitation.
4. **Aleatory** inside the integral: GMPE sigma, focal-depth PMF, magnitude PDF.
5. **Compute by Monte Carlo**, not numerical integration, so you can get spatially correlated fields and loss exceedance curves.
6. **Short forecast window** — 6–24 months maximum, re-run on a fixed cadence (6-hourly for stimulation, 6-monthly for regional).
7. **Update live** — Bayesian change-point or moving-window re-estimation of rates; refit site-specific GMPEs as soon as you have recordings (worth a factor of ~2 in σ).
8. **Test the Poisson assumption** with K-S against Weibull/BPT/gamma per phase; convert to equivalent annual rate if it fails.
9. **Validate prospectively** with CSEP-style N/M/S tests and information gain, and report the failures.
10. **Derive TLS thresholds by inverting** the exceedance criterion against an explicit risk target (e.g. IR ≤ 10⁻⁶/yr), risk-normalised for local population density, with **ambiguous zones** reflecting magnitude-estimation uncertainty.
11. **Design for the TLS failing** — build damage mitigation into the project, budget for trailing seismicity (up to +1.6 magnitude units, mostly within days of shut-in), and don't assume forewarning.

---

## 14. Sources

**Verified sources (claims put through 3-vote adversarial refutation):**

| Source | Contribution |
|---|---|
| [USGS OFR 2016-1035](https://pubs.usgs.gov/publication/ofr20161035) — Petersen et al., 2016 one-year CEUS forecast | Canonical adapted-PSHA implementation; two-tree logic structure |
| [Grigoli et al. 2024, *Reviews of Geophysics*](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024RG000849) — EGS SHRA guideline | Six-component chain; FoP/Omori source model; b-value branch rule |
| [Bulletin of Earthquake Engineering 2019](https://link.springer.com/article/10.1007/s10518-019-00620-5) — Oklahoma time-dependent hazard/risk | Bayesian change-point rates; OpenQuake Monte Carlo; Mmax/mmin sensitivity |
| [Solid Earth 16:1153, 2025](https://se.copernicus.org/articles/16/1153/2025/) — Bedretto PISHA | Sequential updating; analogue-similarity elicitation; 2–3 order a priori spread |
| [Energies 14:2747, 2021](https://www.mdpi.com/1996-1073/14/10/2747) — St Gallen | 3D seismogenic volume; non-Poisson recurrence; closed-form Mmax exceedance; Bayesian ETAS |
| [Int. J. Greenhouse Gas Control 49:413, 2016](https://www.sciencedirect.com/science/article/pii/S175058361630127X) — White & Foxall | Six-step PSRA framing; the four induced modifications; GMPE weak link |

**Additional extracted sources (single-extraction, quote-supported, not refutation-tested):**

- [Shapiro 2018, *JGR*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JB015850) — seismogenic index theory and extension to production
- [Király-Proag et al. 2016, *JGR*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JB013236) — Induced Seismicity Test Bench, CSEP validation
- [Mignan et al. 2017, *Sci. Rep.*](https://www.nature.com/articles/s41598-017-13585-9) — closed-form exceedance criterion and inverted TLS threshold
- [Segall & Lu 2015, *JGR*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JB012060) — poroelastic rate-and-state; shut-in spike
- [McGarr 2014, *JGR*](http://www3.geosc.psu.edu/Courses/GeomechanicsSeminar/Fall2015/McGarr-2014_JGR.pdf) — M₀max = GΔV and its five assumptions
- [van der Elst et al. 2016, *JGR*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JB012818) — unbounded G-R sampling argument
- [Nature Communications 2021](https://www.nature.com/articles/s41467-020-20700-4) — Pohang; revised bound M₀max = ½(1−c)GΔV
- [Comms. Earth & Environment 2026](https://www.nature.com/articles/s43247-026-03268-7) — stochastic poromechanical analysis; 7–15% a priori Pohang exceedance
- [Bommer & Verdon 2024, *Geomech. Geophys. Geo-energy Geo-resour.*](https://link.springer.com/article/10.1007/s40948-024-00895-2) — MmaxE vs MmaxT; two-node Mmax tree
- [Bommer, van Elk & Zoback 2024, *BSSA* 114:2804](https://pubs.geoscienceworld.org/ssa/bssa/article/114/6/2804/646576/Estimating-the-Maximum-Magnitude-of-Induced) — Groningen Mmax elicitation critique
- [Verdon & Bommer 2020, *J. Seismology*](https://link.springer.com/article/10.1007/s10950-020-09966-9) — magnitude jumps; trailing seismicity statistics
- [Schultz et al. 2021, *JGR*](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB022340) and [*Science* 2021](https://www.science.org/doi/10.1126/science.abg5451) — risk-normalised red-light thresholds
- [Frontiers in Earth Science 9:634688, 2021](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.634688/full) — probabilistic TLS with ambiguous zones; Bayesian magnitude posteriors
- [Vilarrasa et al. 2019, *Solid Earth* 10:871](https://se.copernicus.org/articles/10/871/2019/se-10-871-2019.pdf) — deterministic THM counter-position for CO₂
- [Zhang et al. 2025, *Engineering Geology* 362:108536](https://www.sciencedirect.com/science/article/abs/pii/S0013795225006325) — hybrid physics+SI at Illinois Basin–Decatur
- [GeoHazards 4(4):24, 2023](https://doi.org/10.3390/geohazards4040024) — FSP + CMG coupled CO₂ workflow; Decatur-derived TLS *(duplicate URL: mdpi.com/2624-795X/4/4/24)*
- [IEAGHG 2022-02](https://ieaghg.org/publications/2022-02%20Current%20State%20of%20Knowledge%20Regarding%20the%20Risk%20of%20Induced%20Seismicity%20at%20CO2%20Storage%20Projects.pdf) — state of knowledge; Langenbruch hybrid; TLS conditional validity
- [White & Foxall 2016 (GWPC PDF)](https://www.gwpc.org/wp-content/uploads/2023/07/Assessing-induced-seismicity-risk-at-CO2-storage-projectsRecent-progress-and-remaining-challenges.pdf) — same paper as the ScienceDirect entry above, open-access
- [ORION (NETL EDX)](https://edx.netl.doe.gov/dataset/orion-operational-forecasting-of-induced-seismicity) — DOE ensemble forecasting toolkit
- [FSP (UT CISR)](https://cisr.beg.utexas.edu/fsp) — tool documentation

---

## 15. Known gaps in this review

Stated plainly so you can judge what to trust:

1. **The synthesis step of the automated pipeline failed** (session limit). This report was assembled by hand from the pipeline's verified-claims output and the full 140-claim extraction set. Semantic de-duplication across sources was done manually, not algorithmically.
2. **Three claims could not be verified** — the refutation agents errored out. All three are from White & Foxall 2016 (the four modifications, the two method families, the three Mmax estimators). They are marked **[?]** and are corroborated by other sources in the corpus, but were not independently attacked.
3. **Shapiro's original seismogenic index paper** ([*The Leading Edge* 29:304, 2010](https://pubs.geoscienceworld.org/seg/tle/article/29/3/304/135885/Seismogenic-index-and-magnitude-probability-of)) returned **zero extractable claims** — paywalled. The seismogenic index is characterised here through Shapiro (2018) and downstream implementations rather than the primary formulation.
4. **106 of 140 extracted claims were not verification-tested** — the pipeline verified the top 25 by relevance ranking. Claims marked **[E]** carry a supporting quote from the source but no adversarial check.
5. **Machine-learning forecasting** appears only as one branch of the ORION ensemble. No dedicated ML-forecasting source was retrieved; that is a real gap if you care about that sub-literature.
6. **Groningen** appears through the Mmax critique but the full Groningen hazard-and-risk model (the most operationally mature induced-seismicity PSHA in existence) was not retrieved as a primary source.
7. **No sources on regulatory frameworks per se** (EPA Class VI, Dutch SodM, UK OGA) beyond what the technical papers report.
