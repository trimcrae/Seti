# results/romanlit — ROMAN prior-art sweep

Written by `scripts/romanlit_fetch.py` (workflow `roman-lit.yml`).  Every file is
either a VERBATIM copy of what an archive returned or a machine record of what
was asked and what came back.  Nothing is paraphrased; nothing is asserted from
memory.  The novelty position in `docs/roman.md` §4 is to be read against
`concept_scan.json`, and only when `n_abstracts_scanned > 0`.

## Files

| File | What it is |
|---|---|
| `README.md` | this file |
| `summary.json` | run metadata: offline flag, counts of URLs ok / failed / skipped, whether OpenAlex answered, the five questions, the non-arXiv references the brief names (flagged as assertions), and the full status list |
| `fetch_log.json` | every URL requested with every attempt's HTTP status (or the exception), bytes, entry count; `skipped` records a call that was never made (offline, or OpenAlex unreachable). Rewritten after every fetch so a killed run still leaves a record |
| `id_title_check.json` | for every arXiv id the script asserted: did the fetched entry carry the expected title fragment (`match`)? `null` = unverified. For every named title: did the title search return it (`found`), and which ids does the record itself give (`matched_ids`)? |
| `concept_scan.json` | the decoy-aware scan: every group's `target` regexes over every fetched abstract; per hit the phrase that fired, decoy and booster tags, a verbatim `evidence` window, and the verbatim abstract; per group `closest_neighbours` (decoy-free first, then booster-rich), `decoy_free_hits`, `all_hits`, and the reading `interpretation` |
| `arxiv_id_<group>__<name>.atom` | arXiv API response for one asserted id (verbatim Atom) |
| `arxiv_q_title_<group>__<name>.atom` | arXiv API title search for one named paper (verbatim Atom) |
| `arxiv_q_<group>__<name>.atom` | arXiv API keyword sweep; `decoy_*` names are searches run ON PURPOSE for the confounding literature so a null is interpretable (verbatim Atom) |
| `oa_probe.json` | the single OpenAlex reachability probe (verbatim JSON) |
| `oa_q_<group>__<name>.json` | OpenAlex `works?search=` response (verbatim JSON; abstracts are stored inverted and are re-ordered, not rewritten, for the scan) |
| `txt_<group>__<name>.txt` | `pdftotext` full text of an anchor paper, pulled ONLY after its id/title check passed |

## Groups

* **g1_opaque_lens** — Has anyone searched microlensing light curves for a lens that is OPAQUE over a fraction of its Einstein radius -- an occulter that removes the minor image in the wings (symmetric steps at +/- u_c) -- or asked what density an observed lens's occulting radius implies?  Lensing BY Dyson spheres / megastructures and finite-lens-size / black-hole-shadow microlensing are the neighbours.
* **g2_nir_laser** — Has anyone searched, or proposed to search, for laser lines in the NEAR-INFRARED (1.0-1.9 um; 1064 nm Nd:YAG, 1550 nm Er-fibre) on stellar point sources -- and in particular with wide-field SLITLESS spectroscopy (Euclid NISP, Roman G150/P127)?
* **g3_ramp_flash** — Has anyone used the up-the-ramp jump (cosmic-ray) flags of a non-destructive-read detector as a detector of ASTROPHYSICAL sub-exposure flashes -- or asked the pulsed-optical-SETI question of a wide-field NIR survey?  The engineering neighbours are jump detection, snowballs and cosmic-ray rejection in H4RG/H2RG ramps; the science neighbours are nanosecond pulsed SETI and wide-field sub-second transients.
* **g4_statite** — Has anyone proposed NON-KEPLERIAN ASTROMETRY of a direct-imaging point source -- a reflector that does not move on an orbit -- as a technosignature test, or searched coronagraphic images for statites / station-kept structures / artificial satellites?  Forward 1993 (statite), Socas-Navarro 2018 (Clarke exobelt) and Roman CGI technosignature proposals are the neighbours.
* **g5_roman_seti** — Has ANY technosignature / SETI search or proposal been made for the Nancy Grace Roman Space Telescope / WFIRST -- its GBTDS, HLTDS, HLWAS, grism / prism or CGI products -- of any kind?

## How to read a null

A group with `n_target_regex_hits: 0` after a run with `n_abstracts_scanned > 0`
means the phrasings tried returned no abstract that matches the concept; it is
evidence of novelty on THIS record, not proof.  A run with `offline: true` or
`n_abstracts_scanned: 0` says nothing about novelty at all.
