# PARALLAX4 VSX target-association guard — 2026-10-02

AI-authored implementation with independent code/evidence and scientific review
coordinated by the usage sprint. Base: `65fad7e900faa1a88204ae87948c8a8ec69e94d5`.
Branch: `codex/usage-sprint-2026-10-01-vsx-association`.

## Demonstrated unsafe association

The old VSX query selected the first unordered row in a 10-arcsec cone and its
type became a mechanism for the Gaia target. Actual baseline
[CI 36956258823](https://github.com/trimcrae/Seti/actions/runs/36956258823),
job `110679757271`, source `9a5e9c1dff2f358597f4baee0a68b0224b77ce64`,
executed the production query/classifier path on a synthetic 9-arcsec EA
neighbor. Its original transcript reports
`sep_arcsec=9.000000000 fate=KNOWN_ECLIPSING_BINARY(VSX:EA)`.
Lint and 174 cases passed. This is a reproducible code defect, not an observed
sky misclassification or measured false-positive rate.

## Bounded source/association contract

Native VSX cone rows remain candidates. The query now retains up to 101 rows,
marks a response over 100 as truncated, and preserves native object identifiers,
names, types, periods and coordinate columns. It never assigns the first row's
type to the target. An empty cone at an unverified position epoch is recorded as
`NO_ROWS_AT_QUERY_POSITION`, not catalogue absence.

`RAJ2000/DEJ2000` report J2000 equinox coordinates: that label does not establish
a position epoch. VSX's photometric `Epoch` is never used as an astrometric epoch.
The default live path has no externally reviewed position-epoch/cross-ID receipt
and remains explicitly incomplete. Available Gaia PM fields are passed through
from the existing vetted row without additional catalogue requests; a missing
reference epoch is preserved, not defaulted to 2016.

The optional receipt is a **caller trust boundary**. It must explicitly bind
`B/vsx/vsx` native OID to the exact Gaia DR3 source ID, declare an ICRS position
epoch and an evidence-sourced angular tolerance, provide an HTTPS source URL and
SHA256, and carry `CALLER_REVIEWED` status. Program checks validate syntax,
namespace/identifier bindings and geometry; they **do not authenticate the
receipt or acquire its source**. Synthetic positive receipts in tests are not
evidence that live VSX provides these fields.

Gaia pmra is mu_alpha*cos(dec), in mas/year. The spherical approximation is
`normalize(u_ref + dt * (pmra*east + pmdec*north) * mas_to_radians)`.
It handles RA wrapping/poles and propagates both the query center and comparison
direction to the explicitly evidenced position epoch. This constant tangent
model does not establish perspective acceleration, position/PM covariance or
long-baseline accuracy. It is a consistency check on the caller's identity
evidence, not an identity proof from angular proximity.

A nontruncated response must have valid exact IDs/positions and exactly one
position-compatible candidate whose native OID matches the receipt. Missing,
malformed, duplicated, offset or multiple candidates remain named unresolved
states. No ad-hoc identity radius or scientific threshold is introduced.
Positive output is named `CROSS_ID_RECEIPT_POSITION_CONSISTENT`, and reports
`vsx_receipt_authenticated_by_program = false`.

The classifier recomputes this association contract. Legacy `vsx_type` or a
serialized success flag alone cannot supply a known mechanism, and its type is
read back from the native row. Missing native type after a coherent association
is a separate classification gap. Other existing mechanism paths remain in
place, with the VSX gap flagged; their own counterpart-association/calibration
limits are unchanged. Historical science outputs, preregistrations, queues and
bot work are untouched.

## Validation and stop condition

Actual final scoped/full CI, exact source/blob bindings, independent review and
normal main integration receipts will be recorded after they settle. Controls
cover the original unsafe neighbor, explicit caller-reviewed positive bindings,
epoch shifts/RA wrap/poles, missing metadata, identifier namespace/precision,
ambiguity/duplicates, offsets, truncation boundaries, status-forgery attempts and
stage propagation without additional acquisition. Existing ZTF identity/alias
guards are retained; legacy VSX test fixtures now require explicit synthetic
association evidence.

Stop after one reviewed implementation, actual scoped/full success and normal
main integration plus a small receipt handoff. No science acquisition, paid job,
local Python execution, empirical calibration or astronomical detection is
claimed.

## Next bounded task

Obtain independently acquired native positional-epoch/cross-ID evidence for a
named VSX/Gaia survivor before supplying a caller-reviewed association receipt.
Stop with a missing-input result if no authoritative source binds those facts.
Separately, SIMBAD counterpart identity and real Gaia-sampling calibration remain
unresolved; this repair does not claim to validate them.
