# One native-bound matched-mean CCD discordance control

AI-assisted by OpenAI Codex; independent source/science/native-input review before execution.

Predeclared on 2026-10-02 before executing this control. Entry main was
b9b2f9ef55b4737396052007de58e89ea541af8d. No observed contamination or
confidence defect is asserted. Independent source review established that
epochs.collapse_transits records chi2_ccd but fit_photocentre does not consume
it; the existing tests do not exercise a matched-mean discordant CCD arm.
Independent decoding of the existing pinned ZIP confirmed the native roles,
source/transit IDs, aligned slot/error/use/flag arrays and usable joined rows.

## Fixed construction

Use only existing fixture source 1457486023639239296, its 41 retained native
joint transits and 365 eligible CCDs in 410 original array slots. Keep the
native times, angles, parallax factors, centroid errors, AGIS-use decisions and
unsigned processing flags. Original centroid values and masks remain separately
recorded. The array slot is an ordinal, not a documented physical sensor ID.

Set synthetic psi_input=0.2*sin(2*pi*j*(sqrt(5)-1)/2), j the chronological
transit index; phi=psi_input/(1-psi_input). Apply the production median
re-reference exactly: psi_final=(1+median(phi))*psi_input-median(phi).
Set the prescribed transit centroid delta=psi_final*(300*sin(theta)-200*cos(theta))
mas. These are the preexisting deterministic nondegenerate flux pattern and
D=(300,-200) mas, used here solely for the new CCD-collapse diagnostic.

Reference arm: replace every eligible centroid in a transit with delta.
Discordant arm: replace only the lowest original eligible slot's centroid with
delta*W/w_j; replace all other eligible centroids with zero. Here w_i=1/s_i^2,
W=sum(w_i). Keep ineligible centroids unchanged. No added noise.

Both means equal delta and their formal error equals 1/sqrt(W). The reference
chi2_ccd is zero up to rounding; the discordant chi2_ccd is
delta^2*(W^2/w_j-W). Selected weight fraction and concentrated centroid magnitude
are recorded. The downstream fits should be numerically equivalent because
they see the same means, errors, design and residuals. This is one paired
conditional control, with no tuning after results.

## Verification and limits

Meaningful Python regressions compare actual collapse output against the
independent analytic mean/scatter/error equations for unequal weights,
delta=0 and positive/negative delta, an excluded tiny-error CCD, row shuffling
and a single eligible CCD. Native assertions pin fixture provenance, 41 exact
source/transit bindings, original slots/errors/use/flags, analytic scatter and
numerically equivalent downstream fits. The generator prints canonical JSON
plus LF and SHA256; it writes no scientific outputs.

The fixed native AGIS mask conditions on a published eligibility decision; it
does not show AGIS would accept injected corruption. Generic CCD processing
flags have no bit definitions in this fixture and supply no invented veto.
Large concentrated coordinates are a controlled algebraic construction, not
a realistic contamination-frequency model. Matched means make transit residual
jitter unable to distinguish the arms. This does not estimate contamination
rates, common-mode covariance, a detection or a sky false-positive rate. It
does not calibrate A4, repair production uncertainty, change cuts or alter
historical scientific results. Reader/kernel/fit, fixtures/configuration,
release gates and thresholds remain untouched.

Run from repository root:
python -m seti.parallax4.ccd_control
