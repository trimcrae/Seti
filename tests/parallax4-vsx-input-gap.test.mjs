// AI-authored precision/provenance controls for one historical missing-input audit.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { loadEvidence, parseReportWithoutRoundingIds, validateEvidence } from '../tools/parallax4-vsx-input-gap.mjs';

const base = loadEvidence();
const target = '370895959591587712';
const clone = () => JSON.parse(JSON.stringify(base.record));

test('actual pinned input and raw excerpts produce only a missing-input result', () => {
  const result = validateEvidence(base.record, base.inputs);
  assert.equal(result.target_sid, target);
  assert.equal(result.matching_rows, 1);
  assert.equal(result.physical_csv_line, 3);
  assert.equal(result.status, 'MISSING_NATIVE_CROSS_ID_AND_POSITION_EPOCH_INPUT');
  assert.equal(result.native_source_acquisition, 'SOURCE_NOT_ACQUIRED');
  assert.equal(result.source_availability_status, 'NOT_ASSESSED');
  assert.equal(result.current_survivor_status, 'UNVERIFIED');
});

test('native JSON identifiers are preserved before Number parsing', () => {
  const raw = base.inputs.report.toString('utf8');
  assert.notEqual(String(JSON.parse(raw).survivors[0].source_id), target);
  assert.equal(parseReportWithoutRoundingIds(raw).survivors[0].source_id, target);
  const adjacent = '370895959591587713';
  assert.equal(Number(target), Number(adjacent));
  const probe = '{"source_id":' + target + ',"other":{"source_id":' + adjacent + '}}';
  const safe = parseReportWithoutRoundingIds(probe);
  assert.equal(safe.source_id, target);
  assert.equal(safe.other.source_id, adjacent);
});

const invalidRecords = [
  ['numeric Gaia ID', r => { r.selected_target.gaia_source_id = Number(target); }],
  ['rounded Gaia ID string', r => { r.selected_target.gaia_source_id = String(Number(target)); }],
  ['adjacent native Gaia ID', r => { r.selected_target.gaia_source_id = '370895959591587713'; }],
  ['wrong release', r => { r.selected_target.gaia_release = 'DR2'; }],
  ['unpinned input revision', r => { r.input_commit = '3165be48ea08eacf52a1c6f81c060e5a486df658'; }],
  ['swapped source paths', r => { r.inputs[0].path = r.inputs[1].path; }],
  ['wrong Git source hash', r => { r.inputs[0].git_blob_sha = '0'.repeat(40); }],
  ['wrong SHA256 source hash', r => { r.inputs[0].sha256 = '0'.repeat(64); }],
  ['wrong source size', r => { r.inputs[0].utf8_bytes++; }],
  ['wrong physical row', r => { r.inputs[0].selected_line = 2; }],
  ['wrong excerpt hash', r => { r.inputs[0].raw_excerpt_sha256 = '0'.repeat(64); }],
  ['wrong excerpt path', r => { r.inputs[0].raw_excerpt = 'results/parallax4/vetted.csv'; }],
  ['changed rounded recorded RA', r => { r.selected_target.recorded_fields.ra = '16.0037001'; }],
  ['recorded RA converted to Number', r => { r.selected_target.recorded_fields.ra = 16.0037; }],
  ['wrong historical run', r => { r.historical_report.run_id = '36960008519'; }],
  ['wrong historical producer', r => { r.historical_report.producer_git_sha = r.input_commit; }],
  ['invented current survivor', r => { r.selected_target.current_survivor_status = 'SURVIVES'; }],
  ['guessed Gaia reference epoch', r => { r.missing_inputs.gaia_reference_epoch.value = 2016; }],
  ['guessed J2000 position epoch', r => { r.missing_inputs.vsx_position_epoch.value = 2000; }],
  ['invented native VSX identity', r => { r.missing_inputs.vsx_native_oid.value = '12345'; }],
  ['catalogue absence from tool scope', r => { r.native_source_acquisition.source_availability_status = 'ABSENT'; }],
  ['unexecuted source acquisition', r => { r.native_source_acquisition.status = 'ACQUIRED'; }],
  ['false managed acquisition impossibility', r => { r.native_source_acquisition.managed_cpu_acquisition_possible_in_principle = false; }],
  ['synthetic receipt masquerading as acquired', r => { r.association_receipt = { review_status: 'CALLER_REVIEWED' }; }],
  ['invented identity result', r => { r.result.identity_established = true; }],
  ['invented receipt readiness', r => { r.result.receipt_ready = true; }],
  ['invented detection', r => { r.result.detected = true; }],
  ['missing preservation state', r => { delete r.preservation.science_acquisition_dispatched; }],
];
for (const [label, mutate] of invalidRecords) {
  test('refuses ' + label, () => {
    const record = clone();
    mutate(record);
    assert.throws(() => validateEvidence(record, base.inputs));
  });
}

test('refuses a coherently relabeled changed source fingerprint', () => {
  const record = clone();
  const report = Buffer.from(base.inputs.report.toString('utf8').replace(target, '370895959591587713'));
  record.inputs[1].git_blob_sha = createHash('sha1').update('blob ' + report.length + '\0').update(report).digest('hex');
  record.inputs[1].sha256 = createHash('sha256').update(report).digest('hex');
  assert.throws(() => validateEvidence(record, { ...base.inputs, report }));
});

test('refuses modified native CSV bytes', () => {
  const csv = Buffer.from(base.inputs.csv.toString('utf8').replace(target, '370895959591587713'));
  assert.throws(() => validateEvidence(base.record, { ...base.inputs, csv }));
});

test('refuses nonverbatim excerpts even with a matching supplied digest', () => {
  const record = clone();
  const csvExcerpt = Buffer.from(base.inputs.csvExcerpt.toString('utf8').replace('16.0037', '16.003700'));
  record.inputs[0].raw_excerpt_sha256 = createHash('sha256').update(csvExcerpt).digest('hex');
  assert.throws(() => validateEvidence(record, { ...base.inputs, csvExcerpt }));
});

test('refuses a different report excerpt even with a matching supplied digest', () => {
  const record = clone();
  const reportExcerpt = Buffer.from(base.inputs.reportExcerpt.toString('utf8').replace(target, '370895959591587713'));
  record.inputs[1].raw_excerpt_sha256 = createHash('sha256').update(reportExcerpt).digest('hex');
  assert.throws(() => validateEvidence(record, { ...base.inputs, reportExcerpt }));
});

test('new deep-vet data invalidates the missing-input snapshot', () => {
  assert.throws(() => validateEvidence(base.record, { ...base.inputs, deepvetCsvPresent: true }));
});
