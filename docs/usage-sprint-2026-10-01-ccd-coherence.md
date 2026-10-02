# Native-bound CCD coherence diagnostic — 2026-10-02

AI-assisted by OpenAI Codex; independently reviewed by the coordinator and a
separate source/science/native-input reviewer.

Completed one predeclared paired synthetic control. The existing native reader
records within-transit CCD scatter, but the photocentre fit consumes the collapsed
transit centroids and formal errors without a CCD-coherence test. This control
demonstrates the resulting information loss. It does not establish a production
confidence defect or justify a covariance repair.

The same 41 native Gaia-4 transit bindings and 365 eligible CCDs are used in
both arms. A coherent centroid reference assigns the prescribed synthetic AL
offset to each eligible slot; the discordant arm concentrates it into the
lowest original eligible array slot so the weighted mean stays the same.
Native geometry, errors, AGIS decisions and flags remain unchanged, with
original centroids separately recorded. An array slot is an ordinal, not a
documented physical CCD identifier.

| Actual paired synthetic result | Value |
|---|---|
| Maximum collapsed mean difference | 1.4210854715202004e-14 mas |
| Formal error difference | 0 |
| Reference maximum chi2_ccd | 4.145993789396943e-25 |
| Discordant sum / maximum chi2_ccd | 1109659770.821086 / 137145612.4562731 |
| Maximum concentrated synthetic centroid | 1105.6293803867318 mas |
| Both conditional fits | D=(300,-200) mas; BLEND_UNRESOLVED; sigma_D_max=0.0514591137190617 mas; jitter=0 |

For w_i=1/s_i^2 and W=sum(w_i), concentrating delta*W/w_j into slot j gives
the same transit mean delta and error 1/sqrt(W), while
chi2_ccd=delta^2*(W^2/w_j-W). The expected paired equivalence was declared
before execution, using the existing fixed golden-ratio flux modulation
(amplitude 0.2) and synthetic D=(300,-200) mas.

## Provenance and validation

Entry main b9b2f9ef55b4737396052007de58e89ea541af8d retained bot operational
changes after the completed spatial repair. Declaration/source checkpoint
08e8d31bb0d9921ac361e2a4cad3d7c845f18f33 preceded execution. Only a new
read-only generator, its tests, declaration README and focused CI wiring changed;
production reader/kernel/fit and protected scientific blobs remained unchanged.

Existing fixture source 1457486023639239296 is pinned by the unchanged fixture
README. ZIP blob b3d51a1506ca1d20f304a0895c294b4a9908039f, 625651 bytes,
SHA256 07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b.
The one XML member has SHA256
f81f4dc11064b72d99f536e3d34365694b839629b424d8247a4b5501016f3ce3.
Independent V8 decoding verified both hashes, ZIP CRCs and all 1008 BINARY2
rows/37 fields. Exact native centroid/error/AGIS-use/flag roles were confirmed;
the generic processing-flag description contains no bit definitions.

Actual focused [36992106733](https://github.com/trimcrae/Seti/actions/runs/36992106733),
job 110790454115, checked out e1f874cf01429c03cfe302c33f3deaff10e4b41c,
whose tree 6b4111e2b9133bcb396511e04fdc034bf8a82abb equals tested source
889076da888f7d90caf8ab036eff30ec977b322b. Lint, **321 tests in 61.08 s** and
all three generators passed. New regressions use independent analytical mean,
scatter and error equations, zero/unequal-weight offsets, excluded tiny-error
CCDs, shuffled rows, a single eligible CCD and native-bound fit equivalence.

The first source checkpoint failed actual run 36991935184/job 110789907302
at our B905 lint error: two zip calls omitted strict=True. Tests and generators
were skipped. The repaired source adds strict=True to those two calls only.
No environment failure or scientific output is inferred from the failed run.

Actual full [push run 36992102308](https://github.com/trimcrae/Seti/actions/runs/36992102308)
checked out exact 889076da. Offline job 110790600213 and paper job
110790600381/all steps completed SUCCESS: lint, full pytest through 100% with
retained skips, sample reproduction, paper build and artifact uploads. The
quiet full log prints no total; none is inferred.

[Canonical report](../research/parallax4/ccd-coherence-2026-10-02/report.json)
is the unique actual generator payload plus LF, 80876 UTF-8 bytes, SHA256
6f564c0ab76121861b73c2d60c0e1feabab5562b2d4896c155820b08189e0927.
The [receipt](usage-sprint-2026-10-01-ccd-coherence-receipt.json) pins original
logs, source/protected blobs and native provenance. Independent reviewer checks
matched all 410 slots and 41 native IDs/order, with 3936 comparisons and zero
mismatches beyond <=4.44e-16 rad angle-conversion roundoff. Separate V8 analytical
replay agreed with the actual Python output; it is not Python execution.

Reproduce from repository root with python -m seti.parallax4.ccd_control.
The generator writes no scientific outputs. No local Python execution is claimed.

Normally merged as b663c7df689fd96bd150621de5394fe34de12f74, preserving
late bot parent 0f2a951cc7e0fadc2b9227b908aa9aa4dbf540f6. The merge differs
from the tested tree only in four retained TOCSIN_ZTF result JSONs; all
implementation and protected scientific/configuration blobs match exactly.
The subsequent report/log/receipt/handoff checkpoint is documentation only;
final automatic main CI is separate from exact-source proof.

## Limits and stop

These are noise-free synthetic flux/centroids on native geometry, not measured
contamination, a detection, a sky false-positive rate or A4 calibration.
Coordinates reaching 1105.6 mas are a controlled concentration construction,
not a realistic contamination-frequency model. Fixed native AGIS eligibility
does not establish acceptance of the injected pattern. CCD flag bits are
uninterpreted diagnostics. Original native photometric phi and parallax factors
are pinned by the unchanged source helper; they are not emitted per CCD report row. Transit residual jitter cannot distinguish these
matched-mean arms; no common-mode covariance model or production uncertainty
repair follows.

Scientific thresholds, protected preregistrations, A4/release gates, native
fixtures and historical results are unchanged. The task stops after reviewed
normal integration and its evidence handoff. No new candidate, acquisition,
sky rescreen, continuing monitor or further control is started.

Next distinct candidate is conditional vetting of one native CCD quality, covariance
or eligibility criterion, only if authoritative metadata or a reviewed applicable
preregistered criterion already exists in permitted inputs. This fixture supplies
no processing-bit definitions or CCD covariance model; no such criterion is
established here. Otherwise stop with explicit missing-input/no-change. No next
worker/job, repeated control, acquisition or new coherence cut starts in this turn.
