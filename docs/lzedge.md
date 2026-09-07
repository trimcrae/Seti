# LZEDGE — the LZ 248 keV nuclear-recoil candidate at the kinematic edge

*Status: package built and offline-tested (2026-09-07); the record is being
fetched by `lzlit.yml`; every number below marked PLACEHOLDER is replaced from
`results/lzlit/` before anything is quoted.*

## 1. The event and the question

LUX-ZEPLIN (arXiv:2609.02823, 1 September 2026) reports one event with the
characteristics of a nuclear recoil of 248 ± 23 (stat) ± 23 (sys) keV in a
2.84 tonne-year exposure (the 220 live-day WS2024 run, March 2023 – April
2024), in an analysis window extended to ≈ 270 keV to reach effective-field-
theory and inelastic dark-matter spectra; the event was recorded on 16 June
2023; the known-background expectation in that region is low and the global
significance is 2.6σ (3.4σ local, best model).  Ten theory papers appeared
within five days (Higgsino iDM, kinematic-edge model building, seasonal iDM,
dark-photon iDM, PQ-origin iDM, axion portal, fermionic absorption, iDM in
LZ+CRESST, the Higgsino high-energy sideband, atmospheric-neutrino
up-scattering).  Their common conclusion: an endothermic (inelastic) recoil
with mass splitting δ close to the *kinematic limit set by the halo escape
speed* is the natural reading, and such a signal is strongly seasonal.

That conclusion hands the question to astronomy.  A recoil at the kinematic
edge is a measurement of the **fastest dark matter in the solar
neighbourhood**: the rate at 248 keV for δ ≳ 350 keV is carried by particles
within a few tens of km/s of v_esc + v_lab, i.e. by the part of the velocity
distribution that (i) the sharp-cut Maxwellian misrepresents by construction,
(ii) Gaia has actually measured (escape-speed and tail-shape fits to halo
stars), and (iii) the Large Magellanic Cloud is predicted to dominate.  None of
the ten papers works from the measured tail.  This channel does.

## 2. What is computed

Everything is a pure function in `src/seti/lzedge/`, offline-tested:

* **Kinematics** (`kinematics.py`) — v_min(E; m_χ, δ), the endothermic
  threshold √(2δ/μ), the recoil-energy range at a given speed, the largest δ
  reachable at 248 keV for a given lab-frame maximum speed, the Helm form
  factor (zeros at 94 and 278 keV for Xe-131: 248 keV sits in the second lobe,
  just below its zero), and a √E resolution model anchored at the event.
* **The lab's velocity** (`earth.py`) — v_LSR + v_pec + v_Earth(t) with the
  McCabe (2014) ecliptic vectors; the lab speed peaks on 1 June and troughs on
  1 December; on 16 June 2023 it was 266.0 km/s (v_0 = 238), 15 days after
  the peak, 0.4 km/s below it.  `stream_peak_date` gives the date on which any
  bulk-velocity component is fastest in the lab.
* **Halo models** (`halo.py`) — components in the Galactic frame with density
  fractions: the truncated Maxwellian (closed-form lab-frame speed
  distribution, checked against the analytic g(v_min)); anisotropic Gaussians
  (the Gaia Sausage of SHM++, streams, unbound LMC-like debris) integrated with
  the escape-sphere cap handled exactly; and `Isotropic` components with an
  arbitrary radial profile for the *shape* of the tail — sharp cut, soft
  (King-like) cut, and the Gaia-style (v_esc − v)^k tail.  All give
  g(v_min, t) = ∫_{v>v_min} f_lab/v d³v, cached per date.
* **Rates** (`rate.py`) — SI-like xenon recoil spectra summed over isotopes,
  checked against N_T (ρ/m) σ_A ⟨v⟩; window rates through a tabulated
  efficiency; the observed-energy density after resolution.
* **Timing** (`timing.py`) — for one event, the timing Bayes factor
  Λ_t = R(t_obs)/⟨R⟩_livetime against a constant-rate background (1 for an
  unmodulated signal, → ∞ for a signal that exists only around the event date,
  0 for a model whose rate vanishes on 16 June), the modulation summary
  (peak date, fractional modulation, on-fraction of the year), and the
  energy-likelihood ratio between models at the observed energy.
* **Stages** (`run.py`) — `kinematics` (the (m_χ, δ) map, kinematic ceilings
  per halo, the *reachability calendar*: the dates on which a 248 keV recoil
  is possible at all), `scan` (timing Bayes factor, phase, on-fraction, energy
  density at 248 keV for every halo model in `config/lzedge.yaml`), `tails`
  (tail shape × v_esc × v_0 at the event).  `figures.py` draws the tail on the
  event date, the modulation curves, the calendar and the scan map.

## 3. First numbers (placeholders for the window and live time)

For m_χ = 1 TeV under LZ's halo (v_0 = 238, v_esc = 544 km/s):

| δ [keV] | v_min(248 keV) [km/s] | reachable in 2023 | timing Λ_t | on-fraction |
|---|---|---|---|---|
| 300 | 705 | all year | 1.4 | 1.00 |
| 360 | 777 | all year | 2.7 | 1.00 |
| 370 | 789 | 30 Jan – 1 Oct | — | — |
| 380 | 801 | 22 Mar – 12 Aug | — | — |
| 385 | 807 | 21 Apr – 12 Jul | — | — |
| ≥ 390 | ≥ 813 | never | 0 | 0 |

The energy likelihood at 248 keV rises by ≈ 800× from elastic scattering to
δ ≈ 360 keV, so the joint (E, t) likelihood pushes δ to the edge — where the
answer is set by the tail.  With the soft cut the density at
v_min(δ = 380 keV) is 3× below the sharp cut; with a (v_esc − v)^2.5 tail it
is 100× below.

## 4. Novelty status

PENDING the record (`results/lzlit/followups.json`, INSPIRE `refersto`).  The
specific claims to check against every follow-up: use of the event *date* as
a datum (a live-time-weighted likelihood, not a modulation curve); a measured
(Gaia) escape speed with its tail-shape degeneracy rather than an assumed
cut-off; the LMC-boosted tail at the edge and the phase it predicts; and the
seasonal exposure of the other experiments' high-energy windows.

## 5. Contamination model (what would make this analysis wrong)

* The event is a background: the analysis then constrains nothing about the
  halo, and every statement here is conditional on "if this is dark matter".
  Nothing is claimed unconditionally.
* Energy scale: ± 23 keV systematic shifts v_min(248 keV) by ± 28 km/s at
  fixed δ — comparable to the difference between escape-speed measurements.
  Treated as a nuisance shift.
* Live time: the timing Bayes factor depends on the WS2024 live-time calendar;
  a uniform placeholder is used until the paper's calendar is read.
* Muon seasonal modulation at SURF (per-cent level, July maximum) is far
  weaker than an edge signal's modulation and cannot mimic Λ_t ≫ 1.
