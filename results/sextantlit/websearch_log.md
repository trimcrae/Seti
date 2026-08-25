# SEXTANTLIT — in-sandbox WebSearch evidence log

Session of **2026-08-25** (all times US Eastern unless stated). Every query
issued from the dev sandbox is recorded here with its result count and the hits
that mattered. This is the *pass-1* evidence base; the *pass-2* base is the
runner sweep (`scripts/sextantlit_fetch.py` → `.github/workflows/sextantlit.yml`
→ `results/sextantlit/{queries,hits,zero_hits,ids_verified,term_occupancy}.*`).

## Egress: what is reachable from the sandbox

Both `curl` and `WebFetch` are refused for **every** host tried. This is a hard
constraint on the evidential standard reachable in-sandbox: **no full text was
retrievable here**, so no term-occupancy count was computed here, and every
claim below about a paper's *content* rests on a search-index summary, not on
the paper. The runner script exists to convert those into counted evidence.

| host | probe | result |
|---|---|---|
| `export.arxiv.org` | curl | CONNECT tunnel failed, 403 |
| `arxiv.org` | WebFetch | `EGRESS_BLOCKED` |
| `api.openalex.org` | curl + WebFetch | 403 / `EGRESS_BLOCKED` |
| `api.semanticscholar.org` | WebFetch | `EGRESS_BLOCKED` |
| `ui.adsabs.harvard.edu` | curl | CONNECT tunnel failed, 403 |
| `www.aanda.org` | curl + WebFetch | 403 / `EGRESS_BLOCKED` |
| `vizier.cds.unistra.fr` | curl | CONNECT tunnel failed, 403 |
| `ssd.jpl.nasa.gov` | WebFetch | `EGRESS_BLOCKED` |
| `en.wikipedia.org` | WebFetch | `EGRESS_BLOCKED` |

Proxy self-report (`$HTTPS_PROXY/__agentproxy/status`) confirms
`kind: connect_rejected`, `gateway answered 403 to CONNECT (policy denial)` for
each. **WebSearch works.** It is the only channel used below.

## Queries issued (34 WebSearch calls)

`n` = number of result links returned by the search tool.

| # | Q | query | n | what it settled |
|---|---|---|---|---|
| 1 | Q2 | `Liberato Tanga "Gaia astrometric asteroid binary candidates" A&A 688 A50 arXiv` | 9 | **Title correction**: the paper is *"Binary asteroid candidates in Gaia DR3 astrometry"*, not "Gaia astrometric asteroid binary candidates". arXiv:2406.07195. |
| 2 | Q1 | `Gaia asteroid astrometry residuals anomalous non-gravitational acceleration search` | 10 | No anomaly-search hit; all hits are DR2/DR3/FPR data papers, binaries, spin states, orbit determination. |
| 3 | Q2 | `"Binary asteroid candidates in Gaia DR3 astrometry" Liberato method residuals orbit fit chi-square candidates number` | 9 | Method = **period detection in orbit post-fit residuals**; 343 candidates over 410 windows; ~350+ announced. |
| 4 | Q2 | `Liberato 2024 Gaia DR3 binary asteroid candidates "156" OR "150" candidates list method sparse recovery periodogram photocentre wobble` | 10 | 352 candidates in ESA/press framing; DOI 10.1051/0004-6361/202349122; accepted 8 Aug 2024. |
| 5 | Q2 | `Liberato Tanga Gaia DR3 binary asteroid "post-fit residuals" orbit determination OrbFit independent orbit "along-scan" periodic signature detection method details` | 7 | Residual is the **along-scan (AL) projection**; sample = all ~30,000 DR3 asteroids with ≥1 window of ≥10 consecutive observations. |
| 6 | Q2 | `"Follow the wobble" Gaia FPR astrometric binary asteroids 2026 method false discovery rate abstract authors` | 6 | **arXiv:2605.22702**, Liberato + 9, 21 May 2026, on **Gaia FPR residuals**; 9 known binaries, 25 Pan-STARRS overlaps, 99 DR3 overlaps, **45 objects with trends**. |
| 7 | Q4 | `"Asteroid Orbit Determination Using Gaia FPR" statistical analysis AJ 2024 residuals authors abstract` | 10 | Fuentes-Muñoz, Farnocchia, Naidu, Park (JPL), AJ 167, 290 (2024); centre-of-light phase offsets must be modelled; reported uncertainties optimistic. |
| 8 | Q5 | `Gaia asteroid astrometry Yarkovsky detection non-gravitational parameter measure main belt milliarcsecond` | 10 | Surfaces arXiv:2411.09750 — Yarkovsky fitted on **Gaia DR3+FPR**; 446 NEAs, **54,094 IMBAs**, 43 robust NEA detections, **no main-belt detection**. |
| 9 | Q5 | `arXiv 2411.09750 "Assessing the detection of the Yarkovsky effect using the Gaia DR3 and FPR catalogues" authors abstract main belt 54094 spurious detections` | 6 | Authors: Dziadura, Bartczak, Oszkiewicz. Submitted 14 Nov 2024. |
| 10 | Q3 | `technosignature search asteroid astrometry artificial object orbit residual "Minor Planet Center" search for artificial objects among asteroids` | 9 | No paper doing it. Only generic statements that minor-planet surveys "are naturally also searches for such objects". |
| 11 | Q3 | `Lazio Mahabal "Anomalous Asteroid Accelerations" Acta Astronautica 2026 published technosignature` | 9 | Still only reachable as a citation inside Lazio's review; no independent record surfaced. |
| 12 | Q3 | `Lazio "Solar System Technosignatures" 2026 review Gaia astrometry asteroids anomalous acceleration section` | 9 | arXiv:2606.13797 confirmed to exist; **no evidence it uses Gaia or asteroid astrometry**. |
| 13 | Q4 | `Gaia DR3 asteroid observations "observed minus computed" residuals JPL orbits outlier list rejected observations statistics` | 10 | Gaia DR2 rejected ~1% (27,981 obs); DR3 0.58% outliers; **96% of AL residuals in [−5,+5] mas, 52% sub-mas**; across-scan strongly non-Gaussian; **110 orbits with \|Δa\|/a > 1e−7 vs independent orbits (~0.2%)**. |
| 14 | Q4 | `"Gaia-CRF3" planetary ephemerides asteroid observations comparison residuals INPOP DE440 systematic offsets` | 6 | A&A 2025 (aa52534-24): **1001 asteroids, orbits fitted on data independent of Gaia**, propagated to Gaia epochs, positional differences taken — ~10 mas frame offsets. |
| 15 | Q1 | `anomaly detection machine learning Gaia solar system object observations outlier search unsupervised asteroids astrometric` | 9 | Anomaly detection on Gaia = stellar/exoplanet. Asteroid outlier detection = SNAPS, **photometric features only**. |
| 16 | Q5 | `dark comets main belt non-gravitational acceleration population Seligman Farnocchia 2026 detection limits` | 8 | Dark comets are a **near-Earth** population (3–15 m radii), two sub-populations; surfaces arXiv:2606.01288. |
| 17 | Q3 | `"1998 KY26" Phobos 1 spacecraft dark comet 2026 paper authors arXiv 2606.01288` | 7 | Hibberd, Crowl, Gómez de Olea Ballester, Loeb — 31 May 2026 — **artificial-origin claim for a dark comet**, argued from mission history, not astrometric residuals. |
| 18 | Q1 | `search for unmodeled anomalous acceleration in asteroid orbits population survey "unexplained" residual trend systematic 2025 2026` | 9 | Surfaces the **1998 SH2** result (Nature Astronomy 2026) and arXiv:2508.00056 anomalous-asteroid ML work (physical/spin, not astrometric). |
| 19 | Q5 | `"1998 SH2" dark comet Nature Astronomy 2026 non-gravitational acceleration cometary activity dust release authors` | 8 | **LOOM's standing exceedance 875163 (1998 SH2) is now published as an outgassing dark comet** — 19σ position error at the Aug-2025 approach, coma+tail detected, continuous dust release. |
| 20 | Q5 | `Del Vigna Yarkovsky "detection of the Yarkovsky effect" near-Earth asteroids catalogue significance spurious S/N 2018 successors 2025` | 9 | arXiv:1805.05947 — 87 reliable, 24 marginal, plus an explicit **spurious list** ("unrealistic or not explicable with the Yarkovsky effect"). |
| 21 | Q3 | `"Gaia" solar system objects SETI technosignature search artificial satellites among asteroids Gaia catalogue` | 9 | Gaia in SETI = **stellar** only (SETI Ellipsoid, stellar engines, VLBI localisation). Nothing on SSOs. |
| 22 | Q3 | `technosignature "astrometry" minor planets search artificial probe trajectory deviation observable proposal review 2024 2025` | 9 | Technosignature astrometry = **interstellar objects**, 3I/ATLAS follow-up protocols. Not minor planets. |
| 23 | Q2 | `Liberato Gaia binary asteroid method "orbital fit" using "Gaia only" astrometry OrbFit or JPL orbits residual computation circularity DR3 30000 asteroids windows` | 9 | "residuals of the orbital fit **using only the astrometric measurements** for all 30,000 asteroids in Gaia DR3" — reads as a Gaia-only fit; **flagged UNRESOLVED pending full text**. |
| 24 | Q4 | `Gaia Focused Product Release asteroid orbital solution assessment 2310.14699 comparison JPL 110 orbits anomalous differences outliers` | 10 | arXiv:2310.14699 = A&A 680, A37: 66 months, ~157,000 asteroids, **assessed against best orbits from independent observations** + occultation timings. |
| 25 | Q5 | `Dziadura Bartczak Oszkiewicz Gaia FPR Yarkovsky main belt asteroids 54094 IMBA results false detections statistical distribution A2 negative` | 10 | A&A 693, A31 (Jan 2025). **OrbFit least squares, Gaia FPR + MPC + JPL radar**; 6 elements + A2. |
| 26 | Q2 | `"Follow the wobble" Liberato 2026 Gaia FPR "45 objects" trends wide binary linear detrending residuals method orbit fit JPL` | 8 | **"Linear trends in astrometric residuals may signal wide binaries even when the wobble period cannot be measured"** — trend detection is an explicit component of the method. |
| 27 | Q2 | `Liberato "Follow the wobble" Gaia FPR method section residuals computed orbit "OrbFit" or "JPL Horizons" ephemeris windows transits consecutive observations timescale hours` | 7 | Method components: FPR uncertainty model, MC statistical model, **trend detection**, period search, multi-window detection. |
| 28 | Q2 | `Gaia DR3 asteroid binary detection residuals "orbital fit" Gaia data alone circular "our own orbit" independent orbit Liberato 2024 section 2 data` | 8 | Confirms AL-only measurable residual, attitude-derived scan direction, ~1 mas precision. |
| 29 | contam | `Gaia asteroid astrometry photocentre offset irregular shape phase angle correction center of light model residual systematic mas` | 9 | **(21) Lutetia photocentre–barycentre offset 3.3–5.4 mas, varying with phase angle**; shape/phase offsets "comparable to or larger than the expected binary-induced signal". |
| 30 | Q5 | `Gaia DR4 asteroid astrometry expected Yarkovsky detections main belt sensitivity semimajor axis drift 10^-4 au/Myr prediction` | 8 | Several Gaia-era Yarkovsky drifts at S/N > 10; DR4 expected to raise the count. No published main-belt sensitivity floor found. |
| 31 | Q1 | `asteroid astrometry residuals constrain Planet Nine dark matter fifth force modified gravity anomalous acceleration solar system test` | 7 | Anomalous-acceleration searches in asteroid astrometry **do exist** — but as fifth-force/dark-sector tests (arXiv:2107.04038, 2309.13106), on 9 NEOs / Bennu, with a *physics* prior, not an artificiality prior. |
| 32 | contam | `"space debris" OR "artificial satellite" misidentified as asteroid orbit determination high area-to-mass residual detection catalogue contamination` | ~36 (4 rounds) | HAMR/GEO debris literature; "postfit residual statistics can be used to distinguish problematic candidates in NEO orbit derivation". |
| 33 | Q1 | `systematic search asteroid ephemeris offsets survey detections "observed minus predicted" population anomalies catalog-wide screen minor planets ZTF Rubin` | ~18 (2 rounds) | Ephemeris-offset machinery is standard survey plumbing (precovery, cross-matching). **No catalogue-wide anomaly screen surfaced.** |
| 34 | Q4 | `Veres Farnocchia Chesley statistical analysis astrometric errors asteroid surveys residuals distribution outliers 2017 population` | 9 | arXiv:1703.03479, Icarus 296, 139: population-scale residual statistics over the 13 most productive surveys → weighting scheme, **explicitly outlier-robust rather than outlier-hunting**. |

## Verified identifications carried forward

Every row below was returned by ≥2 independent queries with a consistent
title/author string. **They are still marked provisional**: this repository's
rule is that an ID is cited only after the arXiv API has echoed its real title,
which requires the runner (`VERIFY_IDS` in `scripts/sextantlit_fetch.py`).

| id / DOI | title as returned | authors as returned |
|---|---|---|
| arXiv:2406.07195 = A&A 688, A50, doi 10.1051/0004-6361/202349122 | Binary asteroid candidates in Gaia DR3 astrometry | L. Liberato, P. Tanga, D. Mary, K. Minker, B. Carry, F. Spoto, P. Bartczak, B. Sicardy, D. Oszkiewicz, J. Desmars |
| arXiv:2605.22702 | Follow the wobble: Statistical methods to detect astrometric binary asteroids in Gaia FPR | L. Liberato + 9 |
| arXiv:2411.09750 = A&A 693, A31 | Assessing the detection of the Yarkovsky effect using Gaia DR3 and FPR catalogues | K. Dziadura, P. Bartczak, D. Oszkiewicz |
| arXiv:2310.14699 = A&A 680, A37 | Gaia Focused Product Release: Asteroid orbital solution. Properties and assessment | Gaia Collaboration (P. Tanga et al.) |
| AJ 167, 290, doi 10.3847/1538-3881/ad4291 | Asteroid Orbit Determination Using Gaia FPR: Statistical Analysis | O. Fuentes-Muñoz, D. Farnocchia, S. P. Naidu, R. S. Park |
| arXiv:1805.05947 = A&A 617, A61 | Detecting the Yarkovsky effect among near-Earth asteroids from astrometric data | A. Del Vigna et al. |
| arXiv:1703.03479 = Icarus 296, 139 | Statistical Analysis of Astrometric Errors for the Most Productive Asteroid Surveys | P. Vereš, D. Farnocchia, S. R. Chesley, A. B. Chamberlin |
| arXiv:2606.01288 | Is the Dark Comet 1998 KY26 the Spacecraft Phobos 1? | A. Hibberd, A. Crowl, C. Gómez de Olea Ballester, A. Loeb |
| arXiv:2606.13797 | Solar System Technosignatures / Technosignatures in the Solar System | T. J. W. Lazio |
| doi 10.1038/s41550-026-02913-7 | Non-gravitational acceleration indicative of cometary activity of near-Earth object | Farnocchia et al. (full list not resolved) |
| A&A 2025, aa52534-24 | Comparison of the Gaia-CRF3 and planetary ephemerides via asteroid observations | not resolved |
| arXiv:2107.04038 | Novel constraints on fifth forces and ultralight dark sector with asteroidal data | not resolved |
| arXiv:2211.04498 = MNRAS 518, 3784 | Astrometric detection of binary asteroids | not resolved |

## Known gaps this log cannot close (runner required)

1. **Liberato 2024 / 2026: is the residual taken against a Gaia-only fit?** The
   circularity question that decides how much of SEXTANT is left. Search-index
   text is ambiguous.
2. **Term occupancy.** No full text was reachable, so no term was *counted*.
   Every "unoccupied" statement in `docs/sextant-priorart.md` is currently
   supported by query-level absence only.
3. **Dziadura et al. main-belt A2 distribution.** Whether the 54,094 IMBA fits
   produced a published per-object A2 table (a ready-made screen and
   contamination catalogue) or only a summary statement.
4. **VizieR `J/A+A/688/A50` table structure and row count** — needed before it
   can be used as a binarity veto.
5. **Lazio & Mahabal, *On Anomalous Asteroid Accelerations*** — still only
   visible as a citation. Publication status unknown.
