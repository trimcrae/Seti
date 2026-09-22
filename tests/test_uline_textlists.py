"""Offline suite for the non-VizieR U-line route (AAS MRT / CDS ReadMe + .dat).

Every fetch is an injected callable; no socket is opened.  The formats are
exercised on realistic fragments — a machine-readable table with its
byte-by-byte header, a CDS ReadMe describing two files, a login page served
where a table was expected — because the failure this module exists to survive
is *being answered with the wrong thing*, not being refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from seti.uline.textlists import (
    canonical_line_table,
    fetch_first,
    fetch_text_line_table,
    find_links,
    mrt_table,
    parse_byte_by_byte,
    read_fixed_width,
)

MRT = """Title: The Herschel/HIFI spectral survey of Orion KL
Authors: Crockett N.R. et al.
Table: Unidentified lines
================================================================================
Byte-by-byte Description of file: apj493713t5_mrt.txt
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 10 F10.3  MHz     Freq      Rest frequency of the feature
  12- 17 A6     ---     Species   Molecular species or U for unidentified
  19- 26 F8.3   K       Tpeak     Peak main-beam brightness temperature
  28- 35 A8     ---     Trans     Transition quantum numbers
--------------------------------------------------------------------------------
Note (1): frequencies are rest frequencies at v_LSR = 9 km/s.
--------------------------------------------------------------------------------
 555000.100 CH3OH     1.250 5(1)-4(1)
 555111.200 U         0.310 --------
 556333.400 U         0.880 --------
 557444.500 SO2       2.010 4(2)-3(1)
"""

READ_ME = """J/ApJ/787/112   Orion KL HIFI survey   (Crockett+, 2014)
================================================================================
Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  9 F9.3   GHz     Freq      Rest frequency
  11- 16 A6     ---     Mol       Species ("U" = unidentified)
  18- 24 F7.3   K       TA        Antenna temperature
--------------------------------------------------------------------------------
Byte-by-byte Description of file: table4.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  4 I4     ---     Nlines    Number of lines
--------------------------------------------------------------------------------
"""

DAT = """--------------------------------------------------------------------------------
  555.000 CH3OH   1.250
  555.111 U       0.310
  556.333 U       0.880
"""


def test_parse_byte_by_byte_reads_ranges_units_and_labels():
    spec = parse_byte_by_byte(MRT)
    assert [c["label"] for c in spec] == ["Freq", "Species", "Tpeak", "Trans"]
    assert spec[0] == {"start": 0, "stop": 10, "format": "F10.3", "unit": "MHz",
                       "label": "Freq", "description": "Rest frequency of the feature"}
    assert spec[1]["unit"] == ""                     # "---" is not a unit


def test_parse_byte_by_byte_picks_the_named_file_from_a_multi_file_readme():
    t3 = parse_byte_by_byte(READ_ME, "table3.dat")
    t4 = parse_byte_by_byte(READ_ME, "table4.dat")
    assert [c["label"] for c in t3] == ["Freq", "Mol", "TA"]
    assert [c["label"] for c in t4] == ["Nlines"]
    assert parse_byte_by_byte(READ_ME)[0]["label"] == "Freq"      # first block by default


def test_mrt_table_reads_the_data_after_the_last_rule():
    df = mrt_table(MRT)
    assert len(df) == 4
    assert list(df["Species"]) == ["CH3OH", "U", "U", "SO2"]
    assert df["Freq"].iloc[0] == pytest.approx(555000.100)
    assert df["Tpeak"].iloc[2] == pytest.approx(0.880)
    assert df.attrs["units"]["Freq"] == "MHz"


def test_read_fixed_width_applies_a_readme_spec_to_a_separate_dat_file():
    spec = parse_byte_by_byte(READ_ME, "table3.dat")
    df = read_fixed_width(spec, DAT)
    assert list(df["Mol"]) == ["CH3OH", "U", "U"]
    assert df["Freq"].iloc[1] == pytest.approx(555.111)


def test_canonical_line_table_resolves_roles_and_the_frequency_unit():
    df = canonical_line_table(mrt_table(MRT))
    assert list(df.columns) >= ["row", "freq_mhz", "ident", "intensity", "unidentified"]
    assert df.attrs["freq_scale"] == "unit:MHz"
    assert df["freq_mhz"].iloc[0] == pytest.approx(555000.100)
    assert list(df["unidentified"]) == [False, True, True, False]
    assert np.isfinite(df["intensity"]).all()        # Tpeak resolved as the intensity


def test_canonical_line_table_converts_ghz_from_the_declared_unit():
    spec = parse_byte_by_byte(READ_ME, "table3.dat")
    df = canonical_line_table(read_fixed_width(spec, DAT))
    assert df.attrs["freq_scale"] == "unit:GHz"
    assert df["freq_mhz"].iloc[0] == pytest.approx(555000.0)


def test_all_unidentified_overrides_the_label_parse():
    df = canonical_line_table(mrt_table(MRT), all_unidentified=True)
    assert bool(df["unidentified"].all())


def test_find_links_picks_the_mrt_and_absolutises_it():
    html = ('<a href="/journals/0004-637X/787/2/112/apj493713t5_mrt.txt">Table 5</a>'
            '<a href="https://other/apj493713t1_mrt.txt">Table 1</a>'
            '<a href="/article/10.1088/0004-637X/787/2/112/pdf">PDF</a>')
    got = find_links(html, r"_mrt\.txt$", base="https://iopscience.iop.org/article/x")
    assert got == ["https://iopscience.iop.org/journals/0004-637X/787/2/112/apj493713t5_mrt.txt",
                   "https://other/apj493713t1_mrt.txt"]


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def test_fetch_first_records_every_attempt_and_stops_at_the_first_answer():
    def fake(url, **kw):
        if url == "https://b/ok":
            return MRT
        raise OSError(f"404 {url}")
    body, url, att = fetch_first(["https://a/dead", "https://b/ok", "https://c/never"],
                                 fetch_fn=fake)
    assert body == MRT and url == "https://b/ok"
    assert [a["url"] for a in att] == ["https://a/dead", "https://b/ok"]
    assert att[0]["status"] == "QUERY_FAILED" and "404" in att[0]["error"]


def test_fetch_first_rejects_a_login_page_served_where_a_table_was_expected():
    banner = "<html><body>Please sign in to continue</body></html>" + "x" * 500

    def fake(url, **kw):
        return banner
    body, url, att = fetch_first(["https://paywall/table"], fetch_fn=fake,
                                 must_match="Byte-by-byte Description")
    assert body is None and url is None
    assert att[0]["status"] == "QUERY_RETURNED_ZERO_ROWS"
    assert "does not match" in att[0]["error"]
    assert att[0]["head"].startswith("<html>")


def test_fetch_text_line_table_end_to_end_through_a_scraped_index():
    pages = {"https://iop/article": '<a href="/d/apj1t5_mrt.txt">Table 5</a>',
             "https://iop/d/apj1t5_mrt.txt": MRT}

    def fake(url, **kw):
        if url in pages:
            return pages[url]
        raise OSError("404")
    rec = fetch_text_line_table({"index_urls": ["https://iop/article"],
                                 "link_pattern": r"_mrt\.txt$"}, fetch_fn=fake)
    assert rec["status"] == "OK"
    assert rec["url"] == "https://iop/d/apj1t5_mrt.txt"
    assert rec["n_rows"] == 4 and rec["n_unidentified"] == 2
    assert rec["has_intensity"] is True
    assert rec["roles"]["freq"] == "Freq" and rec["roles"]["ident"] == "Species"
    assert len(rec["table"]) == 4


def test_fetch_text_line_table_uses_a_readme_when_the_data_file_has_no_header():
    def fake(url, **kw):
        if url.endswith("table3.dat"):
            return DAT + "y" * 300
        if url.endswith("ReadMe"):
            return READ_ME
        raise OSError("404")
    rec = fetch_text_line_table({"urls": ["https://cds/table3.dat"],
                                 "readme_urls": ["https://cds/ReadMe"],
                                 "readme_file": "table3.dat"}, fetch_fn=fake)
    assert rec["status"] == "OK" and rec["readme_url"] == "https://cds/ReadMe"
    assert rec["n_rows"] == 3 and rec["freq_scale"] == "unit:GHz"


def test_fetch_text_line_table_reports_every_door_when_all_are_shut():
    def fake(url, **kw):
        raise OSError("403 Forbidden")
    rec = fetch_text_line_table({"index_urls": ["https://iop/article"],
                                 "urls": ["https://cds/a.dat", "https://cds/b.dat"]},
                                fetch_fn=fake)
    assert rec["status"] == "QUERY_FAILED"
    assert rec["error"] == "every data URL failed"
    assert {a["url"] for a in rec["attempts"]} == {"https://iop/article", "https://cds/a.dat",
                                                   "https://cds/b.dat"}
    assert all("403" in a.get("error", "") for a in rec["attempts"])
    assert "table" not in rec                        # nothing is invented


def test_fetch_text_line_table_says_so_when_there_is_no_candidate_url():
    rec = fetch_text_line_table({}, fetch_fn=lambda url, **kw: MRT)
    assert rec["status"] == "QUERY_FAILED" and "no candidate data URL" in rec["error"]


def test_fetch_text_line_table_rejects_a_body_with_no_byte_by_byte_block():
    def fake(url, **kw):
        return "just some prose about Orion KL, at length, " * 40
    rec = fetch_text_line_table({"urls": ["https://x/y.txt"]}, fetch_fn=fake)
    assert rec["status"] == "QUERY_RETURNED_ZERO_ROWS"
    assert "no byte-by-byte description" in rec["error"]
