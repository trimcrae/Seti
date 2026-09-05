# What can feed TOCSIN and LOOM while Rubin is dark

*Measured on the runner 2026-08-25 (`scripts/survey_recon.py`, workflow
`survey-recon.yml`). Raw record: `results/survey_recon/recon.json`. This is a
probe, not a shortlist — every number below came from a live query.*

Rubin has been off sky since the night of 13/14 July 2026 (`docs/rubin-outage.md`,
verdict `SKY_STOPPED`). Both Rubin channels have consumed everything the brokers
hold. What follows is what else exists, and what each source actually serves.

## Ruled out by measurement: ZTF through the same broker — but NOT ZTF itself

*Correction, 2026-09-05.* What is measured below is that ALeRCE's TAP mirror
stopped carrying non-LSST data and that Fink's ZTF portal is unreachable. Those
are facts about two broker services. `rubin-outage` now asks ZTF's own nightly
alert archive, ALeRCE's separate ZTF API and ANTARES directly, and on
2026-09-05 all three held epochs from the night of 2026-09-04: **ZTF is
observing and its public alert stream is being served** (`docs/rubin-outage.md`,
"Is ZTF observing?"). The route through the TAP mirror is closed; the stream is
not.

The cheapest imaginable substitute — point TOCSIN at ZTF via the ALeRCE TAP
service it already uses, changing a survey id and nothing else — **does not
work**. `alerce_tap.object` grouped by `(sid, tid)`:

| sid, tid | objects | newest epoch |
|---|---|---|
| 0, 0 (non-LSST) | 9,093,519 | **2026-04-30** |
| 1, 1 (LSST diaObject) | 5,164,217 | 2026-07-14 |
| 2, 1 (LSST ssObject) | 130,909 | 2026-07-14 |

ALeRCE's non-LSST feed stopped two and a half months *before* the LSST one.
Fink's ZTF portal (`api.fink-portal.org`) did not answer at all (connect
timeout), while its LSST portal answered normally.

## For LOOM — Gaia's asteroid astrometry is the same observable, 1000× sharper

LOOM's observable is the **ephemeris residual** of a known minor planet,
decomposed along- and cross-track. Rubin serves it pre-computed at
**arcsecond** scale. Gaia serves the ingredients at **milliarcsecond** scale:

| table | observations | objects |
|---|---|---|
| `gaiafpr.sso_observation` | **46,264,083** | 156,823 |
| `gaiadr3.sso_observation` | 23,336,467 | 158,152 |

Every column the residual needs is present and public
(`gea.esac.esa.int/tap-server/tap`, already a repository dependency):

* `ra`, `dec` with **separate random and systematic** uncertainties, in mas
  (`ra_error_random`, `ra_error_systematic`, and the ra/dec correlations for
  each) — the error model is given, not assumed;
* `epoch`, `epoch_utc`, `epoch_err`;
* `x_gaia … vz_gaia`, barycentric **and** geocentric position and velocity of
  the observer in AU — so the light-time-corrected prediction needs no model of
  Gaia's own orbit;
* `position_angle_scan` — the scan direction. Gaia's precision is strongly
  anisotropic (along-scan is the good axis), so this is the axis every residual
  must be projected onto. It is the direct analogue of Rubin's
  along-track/cross-track split, and the systematic that would otherwise
  masquerade as structure;
* `astrometric_outcome_ccd`, `astrometric_outcome_transit`, `is_rejected` —
  per-observation quality;
* `number_mp`, `denomination` — the join to JPL/MPC orbit solutions.

**Why this is more than a stopgap.** LOOM's question — is there a *population*
of artificial objects already in the solar system — is a question about a static
population, so an archival dataset answers it as well as a live one. And the
sensitivity of the momentum-ceiling test scales with astrometric precision, so a
sample 7× smaller than Rubin's but ~10³× more precise is a *stronger* test of
the same hypothesis, not a weaker one.

**Two traps that have to be designed around from the start:**

1. **Circularity.** Gaia's SSO observations were themselves used in fitting
   Gaia's orbit solutions, so residuals taken against those orbits are minimised
   by construction and mean nothing. The comparison has to be against
   *independent* orbits — the JPL SBDB solutions LOOM already pulls.
2. **Binaries.** An unresolved binary asteroid's photocentre wobbles, producing
   exactly the anomalous astrometric residual being searched for. This is not
   hypothetical: VizieR carries `J/A+A/688/A50` (Liberato & Tanga), *Gaia
   astrometric asteroid binary candidates*, which is simultaneously the nearest
   prior art to position against and the contamination catalogue to subtract.

**Coverage limit, stated plainly:** Gaia SSO astrometry is archival (the mission's
2014–2020 window). It cannot detect a *new* event, and it cannot substitute for
Rubin's nightly cadence. It replaces the population question, not the alert.

## For TOCSIN — reachability and auth, measured

TOCSIN needs short exposures, a real revisit history (its ledger denominator is
forced photometry), and unattended bulk access.

| source | HTTP | auth | what it brings |
|---|---|---|---|
| **ASAS-SN Sky Patrol v2** | 200 | **none** | Full sky, ~1 d cadence, g≲18. Shallow — but TOCSIN's sample is *nearby* stars, which are bright. The only no-token option. |
| **ATLAS forced photometry** | 200 | free account + token | Full sky **including the south**, ~1 d, 30 s exposures, quads within the hour. The closest like-for-like to a Rubin visit. |
| **ZTF forced photometry (IPAC)** | 401 | account | Northern only. Whether ZTF is still *observing* is not settled by the broker facts above — those are facts about ALeRCE and Fink. Since 2026-09-05 `rubin-outage` asks ZTF's own nightly alert archive (`ztf.uw.edu/alerts/public/`), ALeRCE's ZTF API, ANTARES and Lasair-ZTF for their newest epoch every week and records `decision.ztf` in `results/rubin_outage/brokers.json`; see `docs/rubin-outage.md`. The IRSA light-curve archive (no login) is a data release whose newest epoch was 2025-10-20 on 2026-08-27. |
| **MPC** | 200 | none | Every reported asteroid observation since the 19th century — the longest possible baseline for a residual test. |
| **JPL SBDB** | 200 | none | A1/A2/A3 non-gravitational parameters. |
| Pan-STARRS via MAST | 404 | — | Endpoint path in this probe was wrong; deep northern per-epoch photometry is worth a corrected retry. |

## The zero-cost move, worth saying out loud

**LOOM's momentum-ceiling screen never depended on Rubin.** Its two standing
exceedances (`875163 (1998 SH2)`, `428209 (2006 VC)`) came from JPL SBDB
non-gravitational parameters, which are public, complete, and reachable today.
Re-running that screen at full catalogue scale needs no new survey and no new
acquisition code.
