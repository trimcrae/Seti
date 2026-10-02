# PARALLAX4 native VSX evidence: one recorded input gap — 2026-10-02

AI-authored input audit with independent raw-input/scientific/code review.
Base: `72c8a3277f7016ef3a635d13c282d3c3a24a385d`.
Branch: `codex/usage-sprint-2026-10-01-vsx-native-evidence`.

## One target and the finite result

The selected historical target is **Gaia DR3 370895959591587712**.
It is the first clean DIP in the pinned `results/parallax4/vet.json` report
and occurs once at physical line 3 of the pinned `vetted.csv`.
Its original row says `SURVIVES`, `DIP`, score `280.358`.
Those labels belong to run `36028172559`, reported at
`2026-09-24T20:55:52Z` with producer
`d9426f62a91bd5106631baab761f6b06250b81b6`.
This is the historical over-vetoed DR3 output described in STATUS.
It does **not** establish a current survivor, astronomical detection or health.

The selected inputs do not record a native VSX OID/name, native Gaia–VSX
cross-ID, VSX positional epoch or Gaia reference epoch. The current pinned
tree has no `results/parallax4/deepvet.csv` with a counterpart binding.
The recorded RA/Dec `16.0037 / 39.6213` and PM
`6.40039 / -4.99519` are rounded historical output strings, not newly
acquired astrometry. No epoch or counterpart is inferred from them.

The result is `MISSING_NATIVE_CROSS_ID_AND_POSITION_EPOCH_INPUT`.
Native source acquisition is `SOURCE_NOT_ACQUIRED`; source availability is
`NOT_ASSESSED`. There was no native catalogue query, guessed counterpart,
pipeline rerun or live association receipt. The missing target/source binding
is the stop reason. Direct tools being GitHub-only is **not** catalogue
absence or a claim that managed CPU acquisition is impossible in principle.
Existing cone proximity cannot establish the required identity or epoch.

## Preserved raw evidence and reproducibility

[Evidence record](../research/parallax4/vsx-native-evidence-2026-10-02/evidence.json)
binds the exact input revision, Git blobs, SHA256s, historical producer and
separate unknown states. [CSV excerpt](../research/parallax4/vsx-native-evidence-2026-10-02/vetted-target.raw.csv)
is the verbatim header plus the noncontiguous selected physical row.
[Report excerpt](../research/parallax4/vsx-native-evidence-2026-10-02/vet-target.raw.json)
is the verbatim native first survivor object plus a terminal newline.
Neither excerpt is an acquired native VSX record.

The validator reads the original repository files, checks their Git-blob
and SHA256 byte fingerprints, matches both raw excerpts, and preserves the
Gaia ID as an opaque decimal string **before** JSON parsing.
It refuses rounded/adjacent identifiers, coherently relabeled source changes,
wrong historical provenance, invented epochs/identity/availability states,
nonverbatim excerpts and a newly supplied deep-vet input. It never writes
scientific outputs or creates an association receipt.

Exact reviewed source `b2c17488bbd24278eb271e601fde20a8b18d3abb`
passed actual [CI 36961573577](https://github.com/trimcrae/Seti/actions/runs/36961573577),
job `110696199524`: Node **22.23.3**, syntax, **35/35 cases**, original
repository Git/SHA256 byte/raw-excerpt checks and preserved science-file check.
The original checkout is that same source, tree
`057103397c296b2e95699ac5b371dde61bb4a046`.
Both independent reviewers matched the raw excerpts to original Git blobs
and reviewed source, unknown states and actual CI.

[PR27](https://github.com/trimcrae/Seti/pull/27) normally merged at
`7f24082abe360b2441482709aa3964331b9ab688`, preserving main
`72c8a3277f7016ef3a635d13c282d3c3a24a385d` and tested source as parents.
The merged tree is identical to the tested source. This subsequent checkpoint
adds only this handoff, [exact validation receipt](usage-sprint-2026-10-01-vsx-native-evidence-receipt.json)
and [original focused transcript](usage-sprint-2026-10-01-vsx-native-evidence-focused.log),
SHA256 `3369315fae7705e4c372d076614899580d7cb34df7fa65bd0fedbf9556cddf4a`.
Automatic CI on the later documentation/main head is observed separately once.

Executed commands: `node --test tests/parallax4-vsx-input-gap.test.mjs`
and `node tools/parallax4-vsx-input-gap.mjs --check`, plus syntax and the
preserved-file guard listed in the receipt.
No local Node/Python test execution is claimed.

The preceding VSX repair remains complete: tested source `3165be48`,
262 scoped cases and full CI `36957880959`. Its later final-main CI
`36960008519` was observed once in progress on `72c8a327` before this
input audit; that automatic state is separate from the completed source proof.
No unchanged repair/test was repeated.

## Next bounded action and stop

Only after a real native VSX target binding and an authoritative positional
epoch source are independently supplied, acquire one target-specific source
receipt and check native identities, edition and provenance. If unavailable,
retain this gap. Never substitute equinox, photometric epoch, proximity or a
synthetic receipt. This one-target missing-input result ends the science task. Do not repeat the
unchanged audit without new native target/source evidence; a later session may
select a distinct detection-forward task under its own bounded instruction.
Historical outputs, thresholds, shared queues and bots remain preserved.
