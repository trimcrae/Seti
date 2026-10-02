// AI-authored bounded audit of one historical input; never a VSX association receipt.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const RECORD = 'research/parallax4/vsx-native-evidence-2026-10-02/evidence.json';
const TARGET = '370895959591587712';
const COMMIT = '72c8a3277f7016ef3a635d13c282d3c3a24a385d';
const PINS = [
  ['results/parallax4/vetted.csv', '19a0e4280f11ee8261a9d508a1ea7807a174990a',
   '60d8e7aee48f558c87a5973e6d88196387d9484d67d89aac7c20b7e399f05995', 5360230],
  ['results/parallax4/vet.json', '7bc277eb2e63a73fb51347089aa811d01098a38d',
   '0c77f50790b01d9475bade10c822c98932a1e08cd9d6ca61a5db31ea9780e4b0', 33600],
];
const EXCERPTS = [
  'research/parallax4/vsx-native-evidence-2026-10-02/vetted-target.raw.csv',
  'research/parallax4/vsx-native-evidence-2026-10-02/vet-target.raw.json',
];
function requireThat(condition, message) {
  if (!condition) throw new Error(message);
}
const digest = (bytes, algorithm = 'sha256') => createHash(algorithm).update(bytes).digest('hex');
const gitBlob = (bytes) => createHash('sha1').update('blob ' + bytes.length + '\0').update(bytes).digest('hex');

export function parseReportWithoutRoundingIds(text) {
  // The source is numeric JSON, but Gaia identifiers are opaque decimal tokens.
  // Quote the native tokens BEFORE JSON.parse; never recover them from Number.
  return JSON.parse(text.replace(/("source_id"\s*:\s*)([1-9][0-9]*)(?=\s*[,}])/g, '$1"$2"'));
}
function csvCells(line) {
  const cells = [];
  let value = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') { value += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === ',' && !quoted) { cells.push(value); value = ''; }
    else value += ch;
  }
  requireThat(!quoted, 'Malformed quoted CSV row');
  cells.push(value);
  return cells;
}
function firstNativeObject(text) {
  const marker = text.indexOf('"survivors": [');
  requireThat(marker >= 0, 'Missing native survivor list');
  const start = text.indexOf('{', marker);
  let depth = 0, quoted = false, escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth++;
    else if (ch === '}' && --depth === 0) return text.slice(start, i + 1) + '\n';
  }
  throw new Error('Incomplete native survivor object');
}

export function loadEvidence(root = '.') {
  const read = (path) => readFileSync(resolve(root, path));
  return {
    record: JSON.parse(read(RECORD).toString('utf8')),
    inputs: {
      csv: read(PINS[0][0]), report: read(PINS[1][0]),
      csvExcerpt: read(EXCERPTS[0]), reportExcerpt: read(EXCERPTS[1]),
      deepvetCsvPresent: existsSync(resolve(root, 'results/parallax4/deepvet.csv')),
    },
  };
}

export function validateEvidence(record, inputs) {
  requireThat(record?.schema === 'seti-parallax4-native-evidence-gap-v1', 'Wrong gap schema');
  requireThat(record.ai_authored === true && record.repository === 'trimcrae/Seti', 'Wrong authorship/repository');
  requireThat(record.input_commit === COMMIT, 'Wrong pinned input commit');
  requireThat(record.selected_target?.gaia_release === 'DR3' &&
    typeof record.selected_target.gaia_source_id === 'string' &&
    record.selected_target.gaia_source_id === TARGET, 'Native Gaia identity must be the exact opaque decimal string');
  requireThat(record.selected_target.selection === 'first clean DIP entry in the pinned historical vet report', 'Wrong bounded selection');
  requireThat(record.selected_target.current_survivor_status === 'UNVERIFIED', 'Historical output cannot establish current survivor status');
  requireThat(Array.isArray(record.inputs) && record.inputs.length === 2, 'Exactly two pinned repository inputs required');
  for (const [i, bytes] of [inputs.csv, inputs.report].entries()) {
    requireThat(Buffer.isBuffer(bytes), 'Input must preserve original bytes');
    const pin = PINS[i], source = record.inputs[i];
    requireThat(source.path === pin[0] && source.git_blob_sha === pin[1] &&
      source.sha256 === pin[2] && source.utf8_bytes === pin[3], 'Wrong source provenance binding');
    requireThat(bytes.length === pin[3] && gitBlob(bytes) === pin[1] &&
      digest(bytes) === pin[2], 'Original repository source fingerprint mismatch');
    const excerpt = i === 0 ? inputs.csvExcerpt : inputs.reportExcerpt;
    requireThat(Buffer.isBuffer(excerpt) && source.raw_excerpt === EXCERPTS[i] &&
      digest(excerpt) === source.raw_excerpt_sha256, 'Raw excerpt fingerprint/path mismatch');
  }
  requireThat(record.inputs[0].selected_line === 3, 'Wrong physical CSV line');
  const lines = inputs.csv.toString('utf8').split('\n');
  const matches = lines.map((line, index) => ({ line, index }))
    .filter(({ line }) => line.startsWith(TARGET + ','));
  requireThat(matches.length === 1 && matches[0].index === 2, 'Target row is not uniquely bound at the recorded line');
  const header = csvCells(lines[0]), cells = csvCells(matches[0].line);
  requireThat(header.length === 74 && cells.length === header.length, 'Wrong native CSV shape');
  const row = Object.fromEntries(header.map((key, i) => [key, cells[i]]));
  requireThat(inputs.csvExcerpt.toString('utf8') === lines[0] + '\n' + matches[0].line + '\n', 'CSV excerpt is not verbatim header plus selected row');
  const rawReport = inputs.report.toString('utf8');
  requireThat(inputs.reportExcerpt.toString('utf8') === firstNativeObject(rawReport), 'Report excerpt is not the verbatim native first object');
  const report = parseReportWithoutRoundingIds(rawReport);
  const native = report.survivors[0];
  requireThat(native.source_id === TARGET && row.source_id === TARGET, 'Native report/CSV identity mismatch');
  requireThat(native.vet_class === 'SURVIVES' && native.kind === 'DIP' &&
    row.vet_class === native.vet_class && row.kind === native.kind &&
    row.score === String(native.score) && row.t_peak === String(native.t_peak), 'Historical target fields disagree');
  requireThat(record.selected_target.historical_vet_class === row.vet_class &&
    record.selected_target.historical_kind === row.kind, 'Recorded historical class/kind mismatch');
  for (const field of ['ra', 'dec', 'pmra', 'pmdec', 'score', 't_peak']) {
    requireThat(record.selected_target.recorded_fields?.[field] === row[field], 'Recorded native field mismatch: ' + field);
  }
  const history = record.historical_report;
  requireThat(history?.generated_utc === report.generated_utc && history.run_id === report.run_id &&
    history.producer_git_sha === report.git_sha, 'Historical report provenance mismatch');
  requireThat(report.run_id === '36028172559' &&
    report.git_sha === 'd9426f62a91bd5106631baab761f6b06250b81b6', 'Wrong historical producer');
  requireThat(inputs.deepvetCsvPresent === false, 'A new deepvet input requires fresh review');
  requireThat(!header.some((field) => /^(ref_epoch|oid|name)$|^vsx_|cross.?id|position_epoch/i.test(field)),
    'New positional/cross-ID input requires fresh review');
  for (const field of ['gaia_reference_epoch', 'vsx_native_oid', 'vsx_native_name', 'native_gaia_vsx_cross_id', 'vsx_position_epoch']) {
    requireThat(record.missing_inputs?.[field]?.state === 'NOT_RECORDED_IN_SELECTED_INPUT' &&
      record.missing_inputs[field].value === null, 'Unknown input cannot be invented: ' + field);
  }
  requireThat(record.native_source_acquisition?.status === 'SOURCE_NOT_ACQUIRED' &&
    record.native_source_acquisition.source_availability_status === 'NOT_ASSESSED' &&
    record.native_source_acquisition.managed_cpu_acquisition_possible_in_principle === true,
    'No native source acquisition/availability result was obtained');
  requireThat(record.result?.status === 'MISSING_NATIVE_CROSS_ID_AND_POSITION_EPOCH_INPUT' &&
    record.result.identity_established === false && record.result.receipt_ready === false &&
    record.result.classification_updated === false && record.result.detected === false,
    'Input gap cannot become an association, receipt or detection');
  requireThat(!Object.hasOwn(record, 'association_receipt'), 'A missing-input audit is not a live association receipt');
  for (const field of ['historical_outputs_modified', 'science_acquisition_dispatched', 'shared_queues_modified', 'paid_compute_started']) {
    requireThat(record.preservation?.[field] === false, 'Preservation claim missing or changed: ' + field);
  }
  return { target_sid: TARGET, physical_csv_line: 3, matching_rows: 1,
    status: record.result.status, native_source_acquisition: 'SOURCE_NOT_ACQUIRED',
    source_availability_status: 'NOT_ASSESSED', current_survivor_status: 'UNVERIFIED',
    recorded_fields: record.selected_target.recorded_fields };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { record, inputs } = loadEvidence();
  console.log(JSON.stringify(validateEvidence(record, inputs), null, 2));
}
