"""Offline test suite for ISOTOPE --- isotopic purity without a package (S46).

No network anywhere (``conftest.py`` raises on any socket).  Per
``docs/channel-brief.md`` §5 the suite:

* recovers an injected 28Si-pure grain (-995 permil) with solar partners as a
  PURITY_CANDIDATE, and an injected 13C-free grain with solar N and Si as a
  CARBON_PURITY_CANDIDATE;
* classifies a synthetic supernova X grain (-700/-900 with 26Al and 44Ti) as
  NATURAL_PACKAGE, whether or not it is labelled X;
* yields zero candidates on a synthetic mainstream population (with the
  mass-dependent grains tagged FRACTIONATION);
* returns INSUFFICIENT_PANEL (never a candidate) for a grain with only Si
  measured, and raises CONTAMINATION_SUSPECT on a wafer-like grain;
* resolves PGD-like headers (delta, err[...], unit multipliers) at runtime and
  records a missing role as a test NOT RUN;
* parses xlsx with the stdlib reader (no openpyxl in the venv), csv and zip;
* walks the acquisition routes in order with a scripted fetcher, and reports
  NO_DATA_REACHED with the routes tried when every route fails.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seti.isotope import acquire as A
from seti.isotope import purity as P
from seti.isotope.run import (
    isotope_run,
    load_isotope_config,
    main,
    stage_assess,
    stage_screen,
)

CONF = load_isotope_config()

# PGD-like headers: Greek delta, err[...] error columns, a unit multiplier.
HEADERS = ["PGD ID", "Meteorite", "PGD Type", "δ(29Si/28Si) [‰]", "err[δ(29Si/28Si)]",
           "δ(30Si/28Si) [‰]", "err[δ(30Si/28Si)]", "12C/13C", "err[12C/13C]", "14N/15N",
           "err[14N/15N]", "26Al/27Al", "err[26Al/27Al]", "δ(44Ca/40Ca) [‰]", "err[δ(44Ca/40Ca)]"]


def grain(gid, typ, d29, e29, d30, e30, c=None, ec=None, n=None, en=None, al=None, eal=None,
          ca=None, eca=None, met="Murchison"):
    return [gid, met, typ, d29, e29, d30, e30, c, ec, n, en, al, eal, ca, eca]


def mainstream_rows(n=200, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d30 = rng.uniform(-50, 200)
        d29 = 1.37 * d30 + rng.normal(0, 12)
        rows.append(grain(f"M{i}", "M", round(d29, 1), round(rng.uniform(8, 25), 1), round(d30, 1),
                          round(rng.uniform(8, 25), 1), round(rng.uniform(40, 100), 1), 3.0,
                          round(float(np.exp(rng.uniform(np.log(300), np.log(5000)))), 0), 60.0))
    return rows


def x_rows(n=20, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(grain(f"X{i}", "X", round(rng.uniform(-550, -250), 1), 12.0,
                          round(rng.uniform(-700, -350), 1), 12.0, round(rng.uniform(100, 1000), 1),
                          20.0, round(rng.uniform(20, 100), 1), 5.0, round(rng.uniform(0.1, 0.5), 3),
                          0.05, round(rng.uniform(100, 2000), 0), 50.0))
    return rows


def fractionation_rows():
    return [grain("F1", "M", -10.0, 3.0, -20.0, 3.0, 60.0, 3.0, 800.0, 60.0),
            grain("F2", "M", 12.0, 4.0, 24.0, 4.0, 70.0, 3.0, 900.0, 60.0)]


PURE = grain("PURE1", "U", -995.0, 10.0, -996.0, 10.0, 89.0, 4.0, 272.0, 15.0, "<0.0005", None, 5.0, 30.0)
XLIKE_UNLABELLED = grain("SN1", "U", -700.0, 10.0, -900.0, 10.0, 150.0, 5.0, 50.0, 5.0, 0.3, 0.05, 500.0, 100.0)
XLIKE_LABELLED = grain("XEXT", "X", -700.0, 10.0, -900.0, 10.0, 150.0, 5.0, 50.0, 5.0, 0.3, 0.05, 500.0, 100.0)
SI_ONLY = grain("SIONLY", "U", -995.0, 10.0, -996.0, 10.0)
SI_AND_AL_ONLY = grain("SIAL", "U", -995.0, 10.0, -996.0, 10.0, al="<0.0005")
WEAK_PARTNERS = grain("WEAK", "U", -995.0, 10.0, -996.0, 10.0, 89.0, 200.0, 272.0, 500.0)
CARBON_PURE = grain("C12PURE", "U", 5.0, 10.0, 3.0, 10.0, 50000.0, 3000.0, 272.0, 20.0)
CARBON_NATURAL = grain("C12SN", "U", 5.0, 10.0, 3.0, 10.0, 50000.0, 3000.0, 40.0, 5.0)


def raw_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=HEADERS)


def classify(rows, conf=CONF):
    raw = raw_frame(rows)
    res = A.resolve_roles(raw.columns, conf)
    canon = A.canonicalise(raw, res)
    df, env, counts = P.classify_table(canon, conf, roles=res.roles)
    return df.set_index("grain_id"), env, counts


# ---------------------------------------------------------------------------
# schema: headers, roles, numbers
# ---------------------------------------------------------------------------
def test_normalise_header_examples():
    assert A.normalise_header("err[δ(29Si/28Si)] (‰)") == "errd29si28sipermil"
    assert A.normalise_header("δ³⁰Si/²⁸Si") == "d30si28si"
    assert A.normalise_header("Delta 29Si ± 1σ") == "d29sipm1sig"
    assert A.normalise_header("26Al/27Al (×10⁻³)") == "26al27al103"


def test_resolve_roles_on_pgd_like_headers():
    res = A.resolve_roles(HEADERS, CONF)
    r = res.roles
    assert r["grain_id"] == "PGD ID" and r["meteorite"] == "Meteorite" and r["grain_type"] == "PGD Type"
    assert r["d29si"] == "δ(29Si/28Si) [‰]" and r["d29si_err"] == "err[δ(29Si/28Si)]"
    assert r["d30si"] == "δ(30Si/28Si) [‰]" and r["d30si_err"] == "err[δ(30Si/28Si)]"
    assert r["c12c13"] == "12C/13C" and r["c12c13_err"] == "err[12C/13C]"
    assert r["n14n15"] == "14N/15N" and r["n14n15_err"] == "err[14N/15N]"
    assert r["al26al27"] == "26Al/27Al" and r["al26al27_err"] == "err[26Al/27Al]"
    assert r["d44ca"] == "δ(44Ca/40Ca) [‰]" and r["d44ca_err"] == "err[δ(44Ca/40Ca)]"
    assert r["ti44ti48"] is None and r["ti44ti48_err"] is None
    d = res.as_dict()
    assert "ti44ti48" in d["missing"] and "d29si" in d["found"]


def test_resolve_roles_alternative_spellings_and_unit_scale():
    cols = ["Grain Label", "Type", "d29Si", "d29Si err", "d30Si", "d30Si err", "12C/13C", "12C/13C 1σ",
            "26Al/27Al (×10⁻³)", "26Al/27Al (×10⁻³) err", "δ44Ca", "δ44Ca ±", "44Ti/48Ti", "44Ti/48Ti err"]
    res = A.resolve_roles(cols, CONF)
    r = res.roles
    assert r["grain_id"] == "Grain Label" and r["grain_type"] == "Type"
    assert r["d29si"] == "d29Si" and r["d29si_err"] == "d29Si err"
    assert r["c12c13_err"] == "12C/13C 1σ"
    assert r["al26al27"] == "26Al/27Al (×10⁻³)" and res.scales["al26al27"] == 0.001
    assert res.scales["al26al27_err"] == 0.001
    assert r["d44ca"] == "δ44Ca" and r["d44ca_err"] == "δ44Ca ±"
    assert r["ti44ti48"] == "44Ti/48Ti" and r["ti44ti48_err"] == "44Ti/48Ti err"
    raw = pd.DataFrame([["g", "M", 1, 2, 3, 4, 50, 2, 250.0, 20.0, 1, 2, 0.03, 0.01]], columns=cols)
    canon = A.canonicalise(raw, res)
    assert np.isclose(canon["al26al27"][0], 0.25) and np.isclose(canon["al26al27_err"][0], 0.02)


def test_missing_role_is_none_and_its_test_is_recorded_not_run():
    cols = ["ID", "Type", "δ29Si", "err δ29Si", "δ30Si", "err δ30Si"]
    res = A.resolve_roles(cols, CONF)
    assert all(res.roles[p] is None for p in A.PARTNER_ROLES)
    raw = pd.DataFrame([["a", "X", -400.0, 10.0, -500.0, 10.0], ["b", "U", -995.0, 10.0, -995.0, 10.0]],
                       columns=cols)
    canon = A.canonicalise(raw, res)
    df, env, counts = P.classify_table(canon, CONF, roles=res.roles)
    assert "partner:c12c13:column_not_found" in counts["tests_not_run"]
    assert "partner:d44ca:column_not_found" in counts["tests_not_run"]
    b = df.set_index("grain_id").loc["b"]
    assert b["class"] == P.CLASS_INSUFFICIENT and b["beyond_envelope"]
    assert counts[P.CLASS_CANDIDATE] == 0


def test_parse_numeric_handles_limits_commas_and_blanks():
    assert A.parse_numeric("<0.001") == (pytest.approx(np.nan, nan_ok=True), 0.001)
    assert A.parse_numeric("1,234.5")[0] == 1234.5
    assert A.parse_numeric("−999.2 ‰")[0] == -999.2
    assert np.isnan(A.parse_numeric("n.d.")[0]) and np.isnan(A.parse_numeric("")[0])
    assert np.isnan(A.parse_numeric(None)[0]) and A.parse_numeric(3)[0] == 3.0
    v, lim = A.parse_numeric("≤ 2e-3")
    assert np.isnan(v) and lim == 0.002


# ---------------------------------------------------------------------------
# detector: injection, confounders, rejection rules
# ---------------------------------------------------------------------------
def test_injected_pure_grain_with_solar_partners_is_recovered():
    df, env, counts = classify(mainstream_rows() + x_rows() + [PURE])
    assert env["source"] == "empirical_x_grains" and env["n_x_grains"] == 20
    assert env["d29si_min"] >= -560 and env["d30si_min"] >= -710
    g = df.loc["PURE1"]
    assert g["beyond_envelope"] and g["class"] == P.CLASS_CANDIDATE and g["grade"] == "A"
    assert g["n_partners_measured"] == 4 and g["partners_anomalous"] == []
    assert set(g["partners_package_excluded"]) == {"c12c13", "n14n15", "al26al27", "d44ca"}
    assert P.FLAG_CONTAMINATION not in g["flags"]
    assert counts[P.CLASS_CANDIDATE] == 1 and counts["candidates_by_grade"]["A"] == 1
    # the X package was taken from the database's own X grains
    assert env["x_package_source"]["al26al27"].startswith("median_of_")


def test_synthetic_x_grain_is_natural_package_labelled_or_not():
    df, env, _ = classify(mainstream_rows() + x_rows() + [XLIKE_UNLABELLED, PURE])
    sn = df.loc["SN1"]
    assert sn["beyond_envelope"] and sn["class"] == P.CLASS_NATURAL
    assert set(sn["partners_anomalous"]) >= {"al26al27", "d44ca", "n14n15"}
    # labelled X it becomes the envelope; the pure grain is still beyond it
    df2, env2, _ = classify(mainstream_rows() + x_rows() + [XLIKE_LABELLED, PURE])
    assert env2["d29si_min"] == -700.0 and env2["d30si_min"] == -900.0
    assert env2["d29si_min_grain"] == "XEXT"
    assert df2.loc["XEXT"]["class"] == P.CLASS_ORDINARY and not df2.loc["XEXT"]["beyond_envelope"]
    assert df2.loc["PURE1"]["class"] == P.CLASS_CANDIDATE


def test_mainstream_population_yields_zero_candidates_and_tags_fractionation():
    df, env, counts = classify(mainstream_rows(300) + x_rows() + fractionation_rows())
    assert counts[P.CLASS_CANDIDATE] == 0 and counts[P.CLASS_CARBON] == 0
    assert counts["n_beyond_envelope"] == 0
    assert df.loc["F1"]["class"] == P.CLASS_FRACTIONATION
    assert df.loc["F2"]["class"] == P.CLASS_FRACTIONATION
    # every grain is ORDINARY or FRACTIONATION (small-amplitude mainstream grains
    # legitimately sit on the mass-dependent line within their errors); none is
    # NATURAL_PACKAGE / INSUFFICIENT_PANEL / a candidate
    assert counts[P.CLASS_ORDINARY] + counts[P.CLASS_FRACTIONATION] == 322
    assert counts[P.CLASS_FRACTIONATION] >= 2 and counts[P.CLASS_ORDINARY] >= 250
    assert counts[P.CLASS_NATURAL] == 0 and counts[P.CLASS_INSUFFICIENT] == 0


def test_si_only_grain_is_insufficient_panel_and_contamination_suspect():
    df, _, counts = classify(mainstream_rows() + x_rows() + [SI_ONLY, SI_AND_AL_ONLY])
    g = df.loc["SIONLY"]
    assert g["beyond_envelope"] and g["class"] == P.CLASS_INSUFFICIENT
    assert g["reason"] == "no_partner_measured" and P.FLAG_CONTAMINATION in g["flags"]
    g2 = df.loc["SIAL"]
    assert g2["class"] == P.CLASS_INSUFFICIENT and g2["reason"] == "only_1_partner_measured"
    assert P.FLAG_CONTAMINATION in g2["flags"]          # still no C or N
    assert counts[P.CLASS_CANDIDATE] == 0 and counts[P.FLAG_CONTAMINATION] == 2


def test_uninformative_partners_give_a_grade_c_candidate_not_grade_a():
    df, _, _ = classify(mainstream_rows() + x_rows() + [WEAK_PARTNERS])
    g = df.loc["WEAK"]
    assert g["class"] == P.CLASS_CANDIDATE and g["grade"] == "C"
    assert g["partners_package_excluded"] == []


def test_carbon_analogue_recovered_and_rejected_with_anomalous_nitrogen():
    df, env, counts = classify(mainstream_rows() + x_rows() + [CARBON_PURE, CARBON_NATURAL])
    assert env["carbon"]["source"] == "empirical_classified_grains"
    assert env["carbon"]["c12c13_max_natural"] <= 1000.0
    assert df.loc["C12PURE"]["class"] == P.CLASS_CARBON
    assert df.loc["C12SN"]["class"] == P.CLASS_NATURAL
    assert df.loc["C12SN"]["reason"] == "carbon_beyond_nitrogen_anomalous"
    assert counts[P.CLASS_CARBON] == 1


def test_envelope_falls_back_to_config_without_classified_x_grains():
    rows = [grain(f"U{i}", "", -100.0 - i, 10.0, -150.0 - i, 10.0, 60.0, 3.0, 500.0, 40.0) for i in range(5)]
    df, env, counts = classify(rows + [PURE])
    assert env["source"].startswith("config_fallback")
    assert env["d29si_min"] == -650.0 and env["d30si_min"] == -1000.0
    # nothing can be beyond a -1000 floor in d30Si: the fallback is conservative by design
    assert not df.loc["PURE1"]["beyond_envelope"]
    assert env["carbon"]["source"].startswith("not_run")
    assert counts[P.CLASS_CANDIDATE] == 0


def test_missing_silicon_value_or_error_is_no_silicon():
    rows = [grain("NOSI", "M", None, None, None, None, 60.0, 3.0, 500.0, 40.0),
            grain("NOERR", "M", -995.0, None, -995.0, None, 89.0, 4.0, 272.0, 15.0)]
    df, _, counts = classify(x_rows() + rows)
    assert df.loc["NOSI"]["class"] == P.CLASS_NO_SILICON and df.loc["NOSI"]["reason"] == "si_value_missing"
    assert df.loc["NOERR"]["class"] == P.CLASS_NO_SILICON and "si_err_missing" in df.loc["NOERR"]["flags"]
    assert counts[P.CLASS_CANDIDATE] == 0


def test_partner_tests_one_sided_aluminium_and_ti_slot():
    env = {"x_package": dict(P.DEFAULT_PACKAGE)}
    row = {"al26al27": 0.002, "al26al27_err": 0.001, "ti44ti48": 0.022, "ti44ti48_err": 0.001,
           "d44ca": 900.0, "d44ca_err": 50.0}
    t = P.partner_tests(row, env, CONF)
    assert t["al26al27"]["measured"] and t["al26al27"]["solar_consistent"] and t["al26al27"]["package_excluded"]
    assert t["ti44ti48"]["measured"] and t["ti44ti48"]["solar_consistent"]
    assert not t["d44ca"]["measured"]                    # superseded by the Ti slot
    row2 = {"al26al27": 0.02, "al26al27_err": 0.002}
    assert P.partner_tests(row2, env, CONF)["al26al27"]["solar_consistent"] is False
    row3 = {"al26al27_limit": 0.05}
    t3 = P.partner_tests(row3, env, CONF)["al26al27"]
    assert t3["measured"] and t3["solar_consistent"] and not t3["package_excluded"]


def test_mixing_line_distance_and_hull():
    end = (-600.0, -800.0)
    d = P.mixing_line_distance([0.0, -300.0, -300.0], [0.0, -400.0, 0.0], end)
    assert np.isclose(d[0], 0.0) and np.isclose(d[1], 0.0) and d[2] > 200
    hull = P.convex_hull_vertices(np.array([[0, 0], [-600, -800], [-100, -700], [-500, -100]]))
    assert len(hull) == 4
    inside = P.inside_hull(np.array([[-300.0, -400.0], [-900.0, -900.0], [np.nan, 1.0]]), hull)
    assert inside.tolist() == [True, False, False]
    assert P.convex_hull_vertices(np.array([[0, 0], [1, 1]])) == []


def test_frontier_lists_the_purest_grains_regardless_of_class():
    df, _, _ = classify(mainstream_rows() + x_rows() + [SI_ONLY, XLIKE_UNLABELLED, PURE])
    fr = P.frontier(df.reset_index(), 3)
    assert list(fr["grain_id"]) == ["PURE1", "SIONLY", "SN1"]
    assert set(fr["class"]) == {P.CLASS_CANDIDATE, P.CLASS_INSUFFICIENT, P.CLASS_NATURAL}


# ---------------------------------------------------------------------------
# table readers
# ---------------------------------------------------------------------------
def _col(j: int) -> str:
    s = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        s = chr(65 + r) + s
    return s


def make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    """A minimal workbook: shared strings for text, inline strings for the odd cell."""
    from xml.sax.saxutils import escape

    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    shared: list[str] = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        sheet_xml = []
        for si, (name, rows) in enumerate(sheets.items(), start=1):
            out = [f'<worksheet xmlns="{ns}"><sheetData>']
            for i, row in enumerate(rows, start=1):
                cells = []
                for j, v in enumerate(row):
                    if v is None:
                        continue
                    ref = f"{_col(j)}{i}"
                    if isinstance(v, str):
                        if (i + j) % 5 == 0:
                            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(v)}</t></is></c>')
                        else:
                            if v not in shared:
                                shared.append(v)
                            cells.append(f'<c r="{ref}" t="s"><v>{shared.index(v)}</v></c>')
                    else:
                        cells.append(f'<c r="{ref}"><v>{v}</v></c>')
                out.append(f'<row r="{i}">' + "".join(cells) + "</row>")
            out.append("</sheetData></worksheet>")
            z.writestr(f"xl/worksheets/sheet{si}.xml", "".join(out))
            sheet_xml.append(f'<sheet name="{name}" sheetId="{si}" r:id="rId{si}"/>')
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="{rns}"><sheets>'
                                       + "".join(sheet_xml) + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i}" Type="x" Target="worksheets/sheet{i}.xml"/>'
                             for i in range(1, len(sheets) + 1)) + "</Relationships>")
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">'
                   + "".join(f"<si><t>{escape(s)}</t></si>" for s in shared) + "</sst>")
        z.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
    return buf.getvalue()


def _preamble_sheet(rows):
    return [["Presolar Grain Database", None, None], ["SiC release 2024", None, None], [],
            HEADERS] + rows


def test_xlsx_stdlib_reader_finds_the_header_below_a_preamble():
    rows = x_rows(5) + [PURE]
    content = make_xlsx({"Read me": [["This workbook lists grains"], ["nothing here"]],
                         "SiC": _preamble_sheet(rows)})
    sheets = A.read_xlsx_rows_stdlib(content)
    assert [s for s, _ in sheets] == ["Read me", "SiC"]
    df, meta = A.best_frame(sheets, CONF)
    assert meta["sheet"] == "SiC" and meta["header_row_index"] == 3
    assert list(df.columns)[:3] == HEADERS[:3] and len(df) == 6
    res = A.resolve_roles(df.columns, CONF)
    canon = A.canonicalise(df, res)
    assert canon.loc[5, "grain_id"] == "PURE1" and np.isclose(canon.loc[5, "d29si"], -995.0)
    assert np.isclose(canon.loc[5, "al26al27_limit"], 0.0005) and np.isnan(canon.loc[5, "al26al27"])
    # the generic dispatcher recognises the workbook by its zip content
    assert len(A.rows_from_bytes(content, "table.xlsx")) == 2


def test_csv_and_zip_containers():
    csv_bytes = raw_frame(x_rows(3) + [PURE]).to_csv(index=False).encode("utf-8")
    sheets = A.rows_from_bytes(csv_bytes, "pgd.csv")
    df, meta = A.best_frame(sheets, CONF)
    assert meta["header_row_index"] == 0 and len(df) == 4
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("readme.txt", "notes")
        z.writestr("PGD_SiC_2024/sic.csv", csv_bytes)
    sheets = A.rows_from_bytes(zbuf.getvalue(), "release.zip")
    df2, meta2 = A.best_frame(sheets, CONF)
    assert len(df2) == 4 and meta2["sheet"].startswith("PGD_SiC_2024/sic.csv")
    assert A.rows_from_bytes(b"", "x.csv") == []
    assert A.rows_from_bytes(b"PK\x03\x04garbage", "x.zip") == []


def test_link_and_form_extraction():
    html = ('<html><body><form action="search.php" method="get"><input type="text" name="q">'
            '<input type="submit"></form><a href="/files/PGD.xlsx">Download <b>SiC</b> table</a>'
            '<a href="view.php?id=9">Presolar Grain Database</a></body>')
    links = A.extract_links(html, "https://ecl.earthchem.org/home.php")
    assert links[0] == ("https://ecl.earthchem.org/files/PGD.xlsx", "Download SiC table")
    assert links[1][0] == "https://ecl.earthchem.org/view.php?id=9"
    forms = A.extract_search_forms(html, "https://ecl.earthchem.org/home.php")
    assert forms == [{"action": "https://ecl.earthchem.org/search.php", "method": "get", "field": "q"}]


# ---------------------------------------------------------------------------
# acquisition routes with a scripted fetcher
# ---------------------------------------------------------------------------
class Fetch:
    """``pages``: url → (status, bytes, content_type); anything else times out."""

    def __init__(self, pages: dict):
        self.pages, self.calls = pages, []

    def __call__(self, url, *, timeout=120.0, retries=2, max_bytes=None, method="GET"):
        self.calls.append(url)
        if url in self.pages:
            st, body, ct = self.pages[url]
            return {"status": st, "url": url, "content_type": ct, "bytes": len(body),
                    "content": body if st < 400 else b"", "error": None}
        return {"status": None, "url": url, "content_type": None, "bytes": 0, "content": b"",
                "error": "ConnectTimeout: connect timeout=120.0"}


TABLE_XLSX = make_xlsx({"SiC": _preamble_sheet(x_rows(6) + [PURE])})
TABLE_CSV = raw_frame(x_rows(6) + mainstream_rows(20) + [PURE]).to_csv(index=False).encode()
WUSTL = "https://presolar.physics.wustl.edu/presolar-grain-database/"


def test_route_order_human_url_then_crawl():
    pages = {"https://example.org/wrong.xlsx": (404, b"", "text/html"),
             WUSTL: (200, b'<a href="files/PGD_SiC_2024.xlsx">SiC (xlsx)</a>', "text/html"),
             WUSTL + "files/PGD_SiC_2024.xlsx": (200, TABLE_XLSX, "application/octet-stream")}
    f = Fetch(pages)
    acq = A.acquire_table(CONF, table_url="https://example.org/wrong.xlsx", fetch_fn=f)
    assert acq.status == "OK" and acq.route_used == "wustl_crawl_link" and acq.n_rows == 7
    routes = [r["route"] for r in acq.routes]
    assert routes[0] == "human_url" and acq.routes[0]["outcome"] == "FETCH_FAILED"
    assert routes[1] == "local_path" and acq.routes[1]["outcome"] == "SKIPPED_NO_PATH"
    assert "wustl_crawl" in routes and acq.routes[-1]["outcome"] == "OK"
    assert f.calls[0] == "https://example.org/wrong.xlsx" and f.calls[1] == WUSTL
    assert acq.table_meta["header_row_index"] == 3


def test_table_reached_over_a_route_is_cached_and_screened_from_the_cache(tmp_path):
    pages = {WUSTL: (200, b'<a href="files/PGD_SiC_2024.xlsx">SiC (xlsx)</a>', "text/html"),
             WUSTL + "files/PGD_SiC_2024.xlsx": (200, TABLE_XLSX, "application/octet-stream")}
    out = tmp_path / "isotope"
    rep = isotope_run(CONF, stage="acquire,screen,assess", out_dir=out, fetch_fn=Fetch(pages))
    assert (out / "data" / "pgd_raw.csv").exists()
    assert rep["verdict"] == "PURITY_CANDIDATES" and rep["n_grains"] == 7
    assert rep["acquisition"]["route_used"] == "wustl_crawl_link"
    assert rep["candidates"][0]["grain_id"] == "PURE1"
    # a second acquire with every route dead falls back to the cache, not to NO_DATA
    rep2 = isotope_run(CONF, stage="acquire,screen,assess", out_dir=out, fetch_fn=Fetch({}))
    assert rep2["verdict"] == "PURITY_CANDIDATES" and rep2["acquisition"]["route_used"] == "cache"


def test_route_ecl_search_follows_record_and_download():
    home = "https://ecl.earthchem.org/home.php"
    pages = {
        home: (200, b'<form action="search.php" method="get"><input type="text" name="q"></form>', "text/html"),
        "https://ecl.earthchem.org/search.php?q=Presolar+Grain+Database":
            (200, b'<a href="view.php?id=1234">Presolar Grain Database: SiC</a>', "text/html"),
        "https://ecl.earthchem.org/view.php?id=1234":
            (200, b'<a href="download.php?id=1234&f=PGD_SiC.csv">Download data file</a>', "text/html"),
        "https://ecl.earthchem.org/download.php?id=1234&f=PGD_SiC.csv": (200, TABLE_CSV, "text/csv"),
    }
    f = Fetch(pages)
    acq = A.acquire_table(CONF, fetch_fn=f)
    assert acq.status == "OK" and acq.route_used == "ecl_download" and acq.n_rows == 27
    outcomes = {(r["route"], r["outcome"]) for r in acq.routes}
    assert ("wustl_crawl", "FETCH_FAILED") in outcomes
    assert ("ecl_home", "REACHED") in outcomes and ("ecl_search", "REACHED") in outcomes
    assert ("ecl_record", "REACHED") in outcomes
    # the site's own form was used before the guessed search URLs
    assert f.calls.index("https://ecl.earthchem.org/search.php?q=Presolar+Grain+Database") < \
        min((i for i, u in enumerate(f.calls) if "title=Presolar" in u), default=10 ** 6)


def test_route_datacite_lookup_resolves_the_doi():
    api = "https://api.datacite.org/dois"
    q = 'titles.title:"Presolar Grain Database" AND titles.title:SiC'
    from urllib.parse import quote
    js = {"data": [{"id": "10.60520/ieda/999999", "attributes": {
        "doi": "10.60520/ieda/999999", "url": "https://ecl.earthchem.org/view.php?id=999999",
        "titles": [{"title": "Presolar Grain Database: Silicon Carbide (2024)"}]}}]}
    pages = {
        f"{api}?query={quote(q)}&page[size]=25": (200, json.dumps(js).encode(), "application/json"),
        "https://ecl.earthchem.org/view.php?id=999999":
            (200, b'<a href="https://ecl.earthchem.org/data/PGD_SiC.zip">Download</a>', "text/html"),
        "https://ecl.earthchem.org/data/PGD_SiC.zip": (200, _zip_of(TABLE_CSV), "application/zip"),
    }
    f = Fetch(pages)
    acq = A.acquire_table(CONF, fetch_fn=f)
    assert acq.status == "OK" and acq.route_used == "datacite_download"
    ds = [r for r in acq.routes if r["route"] == "datacite_search" and r["outcome"] == "REACHED"]
    assert ds and ds[0]["n_presolar"] == 1 and ds[0]["presolar"][0]["doi"] == "10.60520/ieda/999999"


def _zip_of(csv_bytes: bytes) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("PGD_SiC.csv", csv_bytes)
    return b.getvalue()


def test_every_route_failing_is_no_data_reached_with_the_routes_named(tmp_path):
    f = Fetch({})
    acq = A.acquire_table(CONF, table_url="https://example.org/pgd.xlsx", fetch_fn=f)
    assert acq.status == "NO_DATA_REACHED" and acq.route_used is None and acq.n_rows == 0
    tried = set(acq.as_dict()["routes_tried"])
    assert {"human_url", "local_path", "wustl_crawl", "ecl_home", "ecl_search", "datacite_search"} <= tried
    assert acq.log[-1].startswith("NO_DATA_REACHED")
    # a page that answers but carries no table is REACHED_NO_TABLE, not OK
    f2 = Fetch({"https://example.org/pgd.xlsx": (200, b"<html>login required</html>", "text/html")})
    acq2 = A.acquire_table(CONF, table_url="https://example.org/pgd.xlsx", fetch_fn=f2, routes=("url",))
    assert acq2.status == "NO_DATA_REACHED" and acq2.routes[0]["outcome"] == "REACHED_NO_TABLE"


def test_run_with_unreachable_archive_is_no_data_reached(tmp_path):
    out = tmp_path / "isotope"
    rep = isotope_run(CONF, stage="all", out_dir=out, fetch_fn=Fetch({}))
    assert rep["verdict"] == "NO_DATA_REACHED" and rep["n_grains"] == 0 and rep["candidates"] == []
    s = json.loads((out / "summary.json").read_text())
    assert s["verdict"] == "NO_DATA_REACHED" and "NOT a null result" in s["note"]
    assert "wustl_crawl" in s["acquisition"]["routes_tried"]
    assert all(v == 0 for v in s["counts"].values())
    probe = json.loads((out / "probe.json").read_text())
    assert probe["n_reached"] == 0 and probe["n_probed"] >= 6
    acq = json.loads((out / "acquire.json").read_text())
    assert acq["status"] == "NO_DATA_REACHED" and acq["role_resolution"] is None
    assert (out / "candidates.csv").exists()
    assert len(pd.read_csv(out / "candidates.csv")) == 0


# ---------------------------------------------------------------------------
# end to end from a local table
# ---------------------------------------------------------------------------
def test_end_to_end_from_a_local_csv_reaches_purity_candidates(tmp_path):
    rows = mainstream_rows(120) + x_rows(15) + fractionation_rows() + \
        [PURE, XLIKE_UNLABELLED, SI_ONLY, CARBON_PURE, CARBON_NATURAL, WEAK_PARTNERS]
    path = tmp_path / "PGD_SiC_2024.csv"
    raw_frame(rows).to_csv(path, index=False)
    out = tmp_path / "isotope"
    rep = isotope_run(CONF, stage="acquire,screen,assess", table_path=str(path), out_dir=out,
                      fetch_fn=Fetch({}))
    assert rep["verdict"] == "PURITY_CANDIDATES"
    s = json.loads((out / "summary.json").read_text())
    assert s["n_grains"] == len(rows) and s["n_candidates"] == 3
    assert s["counts"]["PURITY_CANDIDATE"] == 2 and s["counts"]["CARBON_PURITY_CANDIDATE"] == 1
    assert s["counts"]["NATURAL_PACKAGE"] == 2 and s["counts"]["INSUFFICIENT_PANEL"] == 1
    assert s["counts"]["FRACTIONATION"] >= 2 and s["n_contamination_suspect"] == 1
    assert s["candidates_by_grade"] == {"A": 2, "B": 0, "C": 1}
    assert s["role_resolution"]["roles"]["d29si"] == "δ(29Si/28Si) [‰]"
    assert "ti44ti48" in s["role_resolution"]["missing"]
    assert s["envelope"]["source"] == "empirical_x_grains" and s["envelope"]["n_x_grains"] == 15
    assert s["degraded"] == ["partner:ti44ti48:column_not_found"]
    assert s["acquisition"]["route_used"] == "local_path"
    cands = pd.read_csv(out / "candidates.csv")
    assert set(cands["grain_id"]) == {"PURE1", "C12PURE", "WEAK"}
    assert list(cands.sort_values("grade")["grade"]) == ["A", "A", "C"]
    fr = pd.read_csv(out / "frontier.csv")
    assert fr["grain_id"].iloc[0] in ("PURE1", "SIONLY", "WEAK")
    grains = pd.read_csv(out / "grains.csv")
    assert len(grains) == len(rows) and "c12c13_z_solar" in grains.columns
    # re-running screen+assess from the files on disk gives the same verdict
    stage_screen(CONF, out)
    assert stage_assess(CONF, out)["verdict"] == "PURITY_CANDIDATES"


def test_run_with_a_clean_population_is_no_purity_candidate_not_a_null(tmp_path):
    path = tmp_path / "pgd.csv"
    raw_frame(mainstream_rows(80) + x_rows(12)).to_csv(path, index=False)
    out = tmp_path / "isotope"
    rc = main(["--stage", "acquire,screen,assess", "--table-path", str(path), "--out-dir", str(out)])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["verdict"] == "NO_PURITY_CANDIDATE" and s["n_candidates"] == 0
    assert "not an occurrence limit" in s["note"]
    assert s["n_grains"] == 92 and s["grain_types"] == {"M": 80, "X": 12}


def test_xlsx_local_path_end_to_end(tmp_path):
    path = tmp_path / "PGD_SiC_2024.xlsx"
    path.write_bytes(TABLE_XLSX)
    out = tmp_path / "isotope"
    rep = isotope_run(CONF, stage="acquire,screen,assess", table_path=str(path), out_dir=out)
    assert rep["verdict"] == "PURITY_CANDIDATES" and rep["n_grains"] == 7
    assert rep["candidates"][0]["grain_id"] == "PURE1"


# ---------------------------------------------------------------------------
# config, workflow, docs
# ---------------------------------------------------------------------------
def test_config_file_parses_and_carries_every_threshold():
    assert Path("config/isotope.yaml").exists()
    for k in ("acquire", "roles", "error_marker", "solar", "x_package", "envelope_fallback",
              "thresholds", "x_type_regex", "natural_type_regex"):
        assert k in CONF, k
    for role in A.ROLES:
        assert CONF["roles"].get(role), role
    for k in P.DEFAULT_THRESHOLDS:
        assert k in CONF["thresholds"], k
    assert CONF["solar"]["c12c13"] == 89.0 and CONF["solar"]["n14n15"] == 272.0
    assert CONF["acquire"]["timeout_s"] >= 120
    assert any("wustl" in u for u in CONF["acquire"]["crawl_urls"])


def test_workflow_and_doc_exist():
    assert Path(".github/workflows/isotope.yml").exists()
    wf = Path(".github/workflows/isotope.yml").read_text()
    assert "workflow_dispatch" in wf and "table_url" in wf and "commit_results.sh" in wf
    assert 'pip install -e ".[dev]"' in wf
    assert Path("docs/isotope.md").exists()
