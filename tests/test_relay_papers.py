"""RELAY: the e-print route to the published narrowband event tables.

The VizieR route returns the surveys' TARGET lists and none of their EVENT
lists, so the hit tables are parsed out of the papers' arXiv source.  Nothing
here opens a socket: every transport is injected and every byte is a fixture.

What must hold:

* a deluxetable with a frequency and a drift-rate column becomes hit rows, with
  the units the header declares and the provenance of the line it came from;
* a table without a drift rate is NOT a hit table, however SETI-shaped it looks
  (the dominant confounder: target lists outnumber event lists);
* an AAS machine-readable table shipped in the same tarball parses too;
* an arXiv id that resolves to the wrong title is REFUSED, and the refusal
  carries both titles;
* a failed fetch is a recorded failure, never zero hits about the sky.
"""

from __future__ import annotations

import gzip
import io
import tarfile

import numpy as np
import pytest

from seti.metronome.acquire import STATUS_FAILED, STATUS_OK, AcquisitionLog
from seti.relay import papers as pap

# --------------------------------------------------------------------------
# fixtures: what an AAS e-print actually looks like
# --------------------------------------------------------------------------
EVENTS_TEX = r"""
\begin{deluxetable*}{lccccc}
\tablecaption{Events that passed the on--off filter\label{tab:events}}
\tablehead{\colhead{Source} & \colhead{MJD} & \colhead{Frequency (MHz)} &
           \colhead{Drift Rate (Hz s$^{-1}$)} & \colhead{SNR} & \colhead{Classification}}
\startdata
HIP 17147   & 57523.34 & 1420.312 & $-$0.113 & 25.4 & unresolved \\
GJ 251      & 57600.11 & 1539.900 & 0.284\tablenotemark{a} & 14.1 & RFI \\
HD 190360   & 57601.88 & 8412.004 & $+$1.902 & 31.0 & unresolved \\
\enddata
\tablenotetext{a}{Detected in a single on-scan.}
\end{deluxetable*}
"""

TARGETS_TEX = r"""
\begin{deluxetable}{lccc}
\tablecaption{The observed sample}
\tablehead{\colhead{Star} & \colhead{R.A. (deg)} & \colhead{Decl. (deg)} &
           \colhead{Distance (pc)}}
\startdata
HIP 17147 & 55.10 & 18.20 & 25.6 \\
GJ 251    & 101.2 & 33.29 &  5.6 \\
\enddata
\end{deluxetable}
"""

GHZ_TABULAR_TEX = r"""
\begin{table}
\caption{Signals of interest}
\begin{tabular}{lccc}
\hline
Target & Freq. (GHz) & Drift (Hz/s) & S/N \\
\hline
Proxima Cen & 0.982002 & 0.038 & 60.0 \\
\hline
\end{tabular}
\end{table}
"""

MRT_TXT = """Title: A deep-learning search for technosignatures
Table 3. Signals of interest
================================================================================
Byte-by-byte Description of file: table3.mrt
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  9 A9     ---     Target    Star name
  11- 19 F9.3   MHz     Freq      Centre frequency
  21- 26 F6.3   Hz/s    Drift     Drift rate
  28- 32 F5.1   ---     SNR       Signal to noise
--------------------------------------------------------------------------------
HIP 13402 1379.271  -0.311  20.1
HIP 62207 1621.100   0.552  15.9
"""

ATOM_OK = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>http://arxiv.org/abs/1709.03491v1</id>
  <published>2017-09-11T18:00:00Z</published>
  <title>The Breakthrough Listen Search for Intelligent Life: 1.1-1.9 GHz
  Observations of 692 Nearby Stars</title>
 </entry>
</feed>
"""

ATOM_WRONG = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>http://arxiv.org/abs/1709.03491v1</id>
  <published>2017-09-11T18:00:00Z</published>
  <title>A totally unrelated paper about dust in the Large Magellanic Cloud</title>
 </entry>
</feed>
"""


def _tarball(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def log():
    return AcquisitionLog(prefix="test/relay/papers")


# --------------------------------------------------------------------------
# LaTeX parsing
# --------------------------------------------------------------------------
def test_deluxetable_header_and_rows_survive_the_macros():
    tables = pap.latex_tables(EVENTS_TEX, "ms.tex")
    assert len(tables) == 1
    t = tables[0]
    assert t.header[:3] == ["Source", "MJD", "Frequency (MHz)"]
    assert t.header[0 + 3].startswith("Drift Rate (Hz s")
    assert t.header[4:] == ["SNR", "Classification"]
    assert len(t.rows) == 3
    assert t.rows[0][0] == "HIP 17147"
    # $-$ is a minus sign, \tablenotemark{a} is not part of the number
    assert t.rows[0][3] == "-0.113"
    assert t.rows[1][3] == "0.284"
    assert "filter" in t.caption and "tab:events" not in t.caption


def test_event_table_becomes_hit_rows_with_provenance():
    t = pap.latex_tables(EVENTS_TEX, "ms.tex")[0]
    df, rep = pap.table_hits(t, {"seed": "enriquez2017", "arxiv_id": "1709.03491",
                                 "title": "BL 692", "telescope": "GBT", "band": "L"})
    assert rep["kind"] == "hits"
    assert rep["freq_unit"] == "mhz"
    assert rep["freq_unit_assumed"] is False
    assert len(df) == 3
    assert np.isclose(df["freq_mhz"].iloc[0], 1420.312)
    assert np.isclose(df["drift_hz_s"].iloc[0], -0.113)
    # both the raw-role and the converted column, so either reader agrees
    assert np.allclose(df["drift"].to_numpy(float), df["drift_hz_s"].to_numpy(float))
    assert df["target"].tolist() == ["HIP 17147", "GJ 251", "HD 190360"]
    assert np.isclose(df["snr"].iloc[2], 31.0)
    assert df["table"].iloc[0] == "arXiv:1709.03491:ms.tex#deluxetable*"
    assert df["catalogue_telescope"].iloc[0] == "GBT"


def test_a_target_list_is_not_a_hit_table():
    t = pap.latex_tables(TARGETS_TEX, "ms.tex")[0]
    df, rep = pap.table_hits(t, {"seed": "x", "arxiv_id": "0000.00000"})
    assert len(df) == 0
    assert rep["kind"] == "targets"
    assert "freq_mhz" not in rep["roles"]


def test_ghz_header_is_converted_and_plain_tabular_is_read():
    t = pap.latex_tables(GHZ_TABULAR_TEX, "soi.tex")[0]
    df, rep = pap.table_hits(t, {"seed": "blc1", "arxiv_id": "2111.06350"})
    assert rep["kind"] == "hits"
    assert rep["freq_unit"] == "ghz"
    assert np.isclose(df["freq_mhz"].iloc[0], 982.002)
    assert np.isclose(df["drift_hz_s"].iloc[0], 0.038)


def test_unitless_frequency_is_scaled_by_magnitude_and_flagged():
    assert np.isclose(pap.freq_to_mhz(np.array([1.42]), None)[0], 1420.0)
    assert np.isclose(pap.freq_to_mhz(np.array([1420.0]), None)[0], 1420.0)
    assert np.isclose(pap.freq_to_mhz(np.array([1.42e9]), None)[0], 1420.0)
    assert np.isclose(pap.freq_to_mhz(np.array([1.42]), "GHz")[0], 1420.0)
    assert np.isclose(pap.drift_to_hz_s(np.array([3.0]), "nHz/s")[0], 3e-9)


# --------------------------------------------------------------------------
# machine-readable tables
# --------------------------------------------------------------------------
def test_machine_readable_table_parses_by_byte_range():
    t = pap.mrt_table(MRT_TXT, "table3.mrt")
    assert t is not None
    assert t.header == ["Target", "Freq (MHz)", "Drift (Hz/s)", "SNR"]
    df, rep = pap.table_hits(t, {"seed": "ma2023", "arxiv_id": "2301.12670"})
    assert rep["kind"] == "hits"
    assert df["target"].tolist() == ["HIP 13402", "HIP 62207"]
    assert np.isclose(df["freq_mhz"].iloc[1], 1621.100)
    assert np.isclose(df["drift_hz_s"].iloc[0], -0.311)


def test_sources_unpack_from_tar_gz_and_from_a_bare_gzipped_tex():
    blob = _tarball({"ms.tex": EVENTS_TEX, "table3.mrt": MRT_TXT, "fig1.pdf": "not text"})
    files = pap.unpack_sources(blob)
    assert set(files) == {"ms.tex", "table3.mrt"}
    tables = pap.tables_from_sources(files)
    assert {t.env for t in tables} == {"deluxetable*", "mrt"}
    single = pap.unpack_sources(gzip.compress(EVENTS_TEX.encode()))
    assert "main" in single


# --------------------------------------------------------------------------
# paper resolution: the id is a hint, the title is the check
# --------------------------------------------------------------------------
def test_an_id_hint_that_resolves_to_another_paper_is_refused(log):
    calls = []

    def get_text(url, timeout=0):
        calls.append(url)
        return ATOM_WRONG if "id_list" in url else "<feed/>"

    rep = pap.resolve_paper("enriquez2017",
                            {"arxiv_id": "1709.03491",
                             "title_contains": ["Breakthrough Listen", "692"],
                             "query": 'all:"692 Nearby Stars"'},
                            get_text_fn=get_text, log=log)
    assert rep["status"] == pap.STATUS_UNRESOLVED
    assert "Magellanic" in rep["title"]          # what it actually got, verbatim
    assert rep["title_contains"] == ["Breakthrough Listen", "692"]
    assert any("search_query" in u for u in calls)   # it fell back to the search


def test_a_verified_id_is_accepted_without_a_search(log):
    calls = []

    def get_text(url, timeout=0):
        calls.append(url)
        return ATOM_OK

    rep = pap.resolve_paper("enriquez2017",
                            {"arxiv_id": "1709.03491",
                             "title_contains": ["Breakthrough Listen", "692"]},
                            get_text_fn=get_text, log=log)
    assert rep["status"] == STATUS_OK
    assert rep["arxiv_id"] == "1709.03491"
    assert rep["route"] == "id_hint"
    assert len(calls) == 1


def test_search_url_encodes_the_query_once():
    u = pap.search_url('all:"Breakthrough Listen" AND ti:"692"', 5)
    assert "%22Breakthrough%20Listen%22" in u or "%22Breakthrough+Listen%22" in u
    assert "%2522" not in u
    assert u.endswith("max_results=5")


# --------------------------------------------------------------------------
# the whole harvest
# --------------------------------------------------------------------------
def test_harvest_reads_the_events_and_records_every_paper(log):
    blob = _tarball({"ms.tex": EVENTS_TEX + TARGETS_TEX, "table3.mrt": MRT_TXT})

    def get_text(url, timeout=0):
        return ATOM_OK

    def get_bytes(url, timeout=0):
        assert "1709.03491" in url
        return blob

    conf = {"papers": {"enriquez2017": {"arxiv_id": "1709.03491",
                                        "title_contains": ["Breakthrough Listen", "692"],
                                        "telescope": "GBT", "band": "L"}}}
    h = pap.harvest_papers(conf, get_text_fn=get_text, get_fn=get_bytes, log=log)
    assert len(h.hits) == 5                       # 3 deluxetable events + 2 MRT rows
    assert h.n_hit_tables == 2
    assert {t["kind"] for t in h.tables} == {"hits", "targets"}
    p = h.papers[0]
    assert p["status"] == STATUS_OK
    assert p["n_hit_rows"] == 5
    assert p["source"]["status"] == STATUS_OK
    assert p["source"]["urls"][0]["url"].endswith("1709.03491")


def test_a_failed_eprint_fetch_is_recorded_not_reported_as_zero_hits(log):
    def get_text(url, timeout=0):
        return ATOM_OK

    def get_bytes(url, timeout=0):
        raise OSError("503 from the mirror")

    conf = {"papers": {"enriquez2017": {"arxiv_id": "1709.03491",
                                        "title_contains": ["Breakthrough Listen", "692"]}}}
    h = pap.harvest_papers(conf, get_text_fn=get_text, get_fn=get_bytes, log=log)
    assert len(h.hits) == 0
    p = h.papers[0]
    assert p["status"] == pap.STATUS_NO_SOURCE
    assert all(u["status"] == STATUS_FAILED for u in p["source"]["urls"])
    assert "503" in p["source"]["urls"][0]["error"]
    assert len(p["source"]["urls"]) == len(pap.EPRINT_URLS)   # both mirrors were tried


def test_the_sweep_skips_papers_already_seeded(log):
    blob = _tarball({"ms.tex": EVENTS_TEX})
    seen = []

    def get_text(url, timeout=0):
        return ATOM_OK                      # every query answers with the same id

    def get_bytes(url, timeout=0):
        seen.append(url)
        return blob

    conf = {"papers": {"enriquez2017": {"arxiv_id": "1709.03491",
                                        "title_contains": ["Breakthrough Listen", "692"]}},
            "sweep_queries": ['abs:"drift rate"'], "sweep_max_papers": 5}
    h = pap.harvest_papers(conf, get_text_fn=get_text, get_fn=get_bytes, log=log)
    assert len(seen) == 1                   # not fetched twice
    assert len(h.hits) == 3
