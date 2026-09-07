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

## 4. Novelty status — read from the record (2026-09-07)

`results/lzlit/` holds the LZ paper's LaTeX source (tables, supplement) and the
text of every paper INSPIRE lists as citing it (19 by 7 September) plus the
halo-kinematics literature. What each follow-up does with the *astrophysics*
and with the *date*:

| Paper | Model | Halo | Uses the event date? | v_esc treatment | LMC tail | Sideband |
|---|---|---|---|---|---|---|
| Freese & Theodosopoulos 2609.01583 | Higgsino iDM | SHM | no | fixed | no | no |
| Su, Yang & Yang 2609.01475 | iDM in LZ + CRESST | SHM | no | fixed | no | no |
| "Higgsino above the sea of fog" 2609.01504 | Higgsino | SHM (v0 220, vesc 540) + Smith-Orlik LMC halo integral, "rough" | notes June peak "not far from" the event | fixed | yes, rough (δ → 480–500 keV) | no |
| Wu, Zhang & Zhu 2609.01590 | Higgsino + Fermi-LAT | SHM | no | fixed | no | no |
| Lou & Lu 2609.01592 | fermionic absorption | — | no | — | no | no |
| Yin 2609.01892 | PQ high-scale SUSY | SHM | no | fixed | no | no |
| Nomura 2609.02505 | Z2 Higgs partner | SHM | no | fixed | no | no |
| Di Mauro 2609.02608 | iDM model building | SHM, one-harmonic v_E(t) | **explicitly not**: "we do not combine the event date… a rigorous timing analysis would have to convolve the predicted rate with the actual LZ live-time distribution" | fixed 544 | mentioned | mentioned |
| Pospelov & Ramani 2609.02775 | Higgsino solar capture | — | no | — | no | no (IceCube: δ > 566 keV excludes the thermal Higgsino) |
| Visinelli 2609.02807 | PQ-origin iDM | SHM | no | fixed | mentioned | mentioned |
| Yamashita 2609.02868 | inelastic dark photon | SHM | no | fixed | no | no |
| 2609.04144 | electroweak iDM signatures | SHM | no | fixed | no | no |
| **Higgsino sideband 2609.04175** | Higgsino, fixed Z-exchange σ | SHM, **v_esc = 544 / 567 / 610**, + **LMC boosted Gaussian** (\|v_b\| = 570, σ_b = 100, cut 200, w = 0.26 %, 0.6 %) and a digitised Smith-Orlik integral | no (year-averaged v_lab) | scanned upward; "as low as 480 km/s" noted in one sentence as relieving the tension | yes | **yes**: N_SB per window event 3.2–7.6 (SHM), 3.6–10.1 (LMC) |
| **McCabe 2609.04181** | iDM shape fit | SHM, **v_esc = 544 fixed**; "how deviations from it would change these results… we leave for future work" | **yes**: density evaluated on the date, ν integrated over 27 Mar 2023 – 1 Apr 2024 with uniform live time | fixed | no | implicit (window 125–400 keV) |
| Atmospheric-ν up-scattering 2609.04185 | background | — | no | — | no | no |
| Axion portal 2609.04186 | axion portal | SHM | no | fixed | no | no |
| Dent & Newstead 2609.04673 | exothermic iDM | SHM (v_lab 250, averaged) | no | fixed | no | yes (0 events in 350–680 keV) |
| Exothermic DM at LZ 2609.05204 | exothermic | SHM | no | fixed | mentioned | yes |
| Xenon excitation 2609.05291 | inelastic nuclear excitation | SHM + LMC "M1" + stream | no | fixed | yes (moderate enhancement) | mentioned |

So, as of 7 September: the event **date** enters exactly one likelihood
(McCabe's, uniform live time, SHM at 544 km/s); the **escape speed** is either
fixed at 544 or scanned *upward* (04175); the **LMC** admixture is used by
04175 (Higgsino only), 01504 (rough) and 05291; the **sideband** is used by
04175, 04673 and 05204. **Nobody integrates over the measured escape speeds**
— and the measurements sit *below* the conventional value: 484.6 +17.8/−7.4
(Necib & Lin 2022, Gaia eDR3), 497 ± 8 (Koppelman & Helmi 2021), 521 +46/−30
(Roche et al. 2024, Gaia DR3), 528 +24/−25 (Deason et al. 2019), against
580 ± 63 (Monari et al. 2018) and the RAVE-2007 544 that Baxter et al. (2021)
recommend and LZ, McCabe and everyone else adopt. Nobody compares halo models
by evidence given the event, and nobody inverts the event into a statement
about the local escape speed. Those three are this channel's claims:

1. **Joint energy × date × sideband likelihood over the measured halo tail.**
   The same three-factor likelihood for every halo model — sharp / soft /
   power-law tails, each Gaia escape-speed measurement integrated over its
   uncertainty, and the 04175 LMC admixture — with the cross section
   marginalised under a physical ceiling.
2. **Evidence per halo model.** A single event that must be (i) at the edge
   for its energy, (ii) at the peak of the year for its date, and (iii) not
   accompanied by sideband events prefers a *lower* escape speed: the sideband
   count per window event at the fitted splitting falls from ≈ 5 at 544 km/s
   to < 1 near 490 km/s (04175 noted the direction; here it is the quantity
   that ranks the Gaia measurements).
3. **The inverse: the escape speed from the event.** Under the inelastic
   reading, the event's own likelihood of v_esc (marginalised over m_χ, δ, σ)
   is a measurement — to be compared with the stellar ones.

Plus the pure-kinematics *reachability calendar* (§3), which no paper states,
and the isotope dependence of the edge (Xe-136's threshold is 20 km/s below
Xe-129's).

**Not novel, and not claimed:** the modulation amplitude at the edge (McCabe,
Di Mauro), the LMC extending δ_max (01504, 04175), the sideband as an upper
bound on δ for the Higgsino (04175), the exothermic alternative (04673, 05204).

**Data release:** the paper's "Data Release" is not yet public; the only
LZ HEPData record linked from lz.lbl.gov (155182) is the 2025 4.2 t yr SI
result (HEPData itself returns 403 to the runner). The efficiency curve used
here is the paper's own description (96 % plateau 14–250 keV, 50 % at 5.4 and
269.9 keV); 04175 digitised Fig. S2 and its sideband numbers agree with ours
to ~30 %.

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
