"""OpenUniverse 2024 adapters (seti.roman.openuniverse) and their ingest dispatch.

Every test runs on small synthetic files built here.  The ``real_*`` tests
re-check the conventions against the public sample files when they are
present locally (they are not on CI) and skip otherwise.  No network.

Conventions the synthetic files reproduce (verified on the real files,
2026-09-09): HEAD ``PTROBS_MIN/MAX`` are 1-based inclusive; PHOT carries one
``MJD == -777`` separator row per SN placed *after* ``PTROBS_MAX`` (outside the
pointer range); truth-index ``x, y`` are 1-based (galsim ``GS_XMIN = 1``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from seti.roman import openuniverse as OU
from seti.roman.schema import DQCutout, LightCurve, load_roman_config

SAMPLE_DIR = Path("/tmp/claude-0/-home-user-Seti/e40f945d-1430-55b0-b071-1f72b15adf18/scratchpad")
REAL_HEAD = SAMPLE_DIR / "snana_0001_HEAD.FITS.gz"
REAL_PHOT = SAMPLE_DIR / "snana_0001_PHOT.FITS.gz"
REAL_IMAGE = SAMPLE_DIR / "tds_img_F184_10307_17.fits.gz"
REAL_TRUTH = SAMPLE_DIR / "truth_index_F184_10307_17.txt"
REAL_OBSEQ = SAMPLE_DIR / "tds_obseq.fits"
REAL_POINTSOURCE = SAMPLE_DIR / "pointsource_10307.parquet"

LETTERS = "ugrizyRZYJWHFK"


@pytest.fixture(scope="module")
def conf():
    return load_roman_config()


# --------------------------------------------------------------------------------------
# Synthetic builders
# --------------------------------------------------------------------------------------

def make_snana_pair(tmp_path: Path, n_sn: int = 3, n_epochs: int = 6, faint_band: str = "R",
                    n_faint: int = 2) -> tuple[Path, Path]:
    """A HEAD/PHOT FITS pair in the real layout: 14 bands per epoch, 1-based
    inclusive PTROBS pointers, one -777 separator row after each SN, SIM_MAGOBS only.
    ``n_faint`` epochs of ``faint_band`` sit at mag 35 (below any detection)."""
    from astropy.io import fits

    mjd_rows, band_rows, mag_rows = [], [], []
    snid, ra, dec, nobs, pmin, pmax, z, peak, model = [], [], [], [], [], [], [], [], []
    row = 0
    for k in range(n_sn):
        start = row + 1                      # 1-based
        for e in range(n_epochs):
            for j, b in enumerate(LETTERS):
                mjd_rows.append(62000.0 + 5.0 * e)
                band_rows.append(b)
                mag = 22.0 + 0.1 * j + 0.05 * e + k
                if b == faint_band and e < n_faint:
                    mag = 35.0
                mag_rows.append(mag)
                row += 1
        stop = row                           # 1-based inclusive
        # separator row, outside the pointer range
        mjd_rows.append(-777.0)
        band_rows.append("-")
        mag_rows.append(99.0)
        row += 1
        snid.append(f"{20000001 + k}")
        ra.append(10.0 + k)
        dec.append(-44.0 - k)
        nobs.append(stop - start + 1)
        pmin.append(start)
        pmax.append(stop)
        z.append(0.5 + 0.1 * k)
        peak.append(62010.0 + k)
        model.append("SALT3.NIR_WAVEEXT")
    head_cols = [
        fits.Column(name="SNID", format="16A", array=np.array(snid)),
        fits.Column(name="RA", format="D", array=np.array(ra)),
        fits.Column(name="DEC", format="D", array=np.array(dec)),
        fits.Column(name="NOBS", format="J", array=np.array(nobs)),
        fits.Column(name="PTROBS_MIN", format="J", array=np.array(pmin)),
        fits.Column(name="PTROBS_MAX", format="J", array=np.array(pmax)),
        fits.Column(name="REDSHIFT_FINAL", format="E", array=np.array(z)),
        fits.Column(name="SIM_PEAKMJD", format="E", array=np.array(peak)),
        fits.Column(name="SIM_MODEL_NAME", format="32A", array=np.array(model)),
        fits.Column(name="SIM_PEAKMAG_F", format="E", array=np.full(n_sn, 23.0)),
    ]
    phot_cols = [
        fits.Column(name="MJD", format="D", array=np.array(mjd_rows)),
        fits.Column(name="BAND", format="2A", array=np.array(band_rows)),
        fits.Column(name="SIM_MAGOBS", format="E", array=np.array(mag_rows, dtype=np.float32)),
    ]
    head = tmp_path / "SYN_0001_HEAD.FITS"
    phot = tmp_path / "SYN_0001_PHOT.FITS"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(head_cols, name="Header")]).writeto(head)
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(phot_cols, name="Photometry")]).writeto(phot)
    return head, phot


def make_tan_wcs(n: int = 64):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [n / 2.0, n / 2.0]
    w.wcs.crval = [9.6, -44.0]
    w.wcs.cd = np.array([[-0.11 / 3600.0, 0.0], [0.0, 0.11 / 3600.0]])
    w.wcs.cunit = ["deg", "deg"]
    return w


def make_image(tmp_path: Path, n: int = 64, filter_token: str = "H158", exptime: float = 302.275,
               mjd: float = 62131.754, zptmag: float = 17.5, sca: int = 3,
               dq_bits: dict | None = None, name: str = "Roman_TDS_simple_model_H158_10307_3.fits") -> Path:
    """A galsim-like image: PRIMARY header (EXPTIME, MJD-OBS, FILTER, ZPTMAG, TAN WCS,
    SCA_NUM, GS_XMIN=1) + SCI float64, ERR float32, DQ uint32 with ``dq_bits``
    ({bit: [(y, x), ...]}) set."""
    from astropy.io import fits
    w = make_tan_wcs(n)
    hdr = w.to_header()
    hdr["EXPTIME"] = exptime
    hdr["MJD-OBS"] = mjd
    hdr["FILTER"] = filter_token
    hdr["ZPTMAG"] = zptmag
    hdr["SCA_NUM"] = sca
    hdr["GS_XMIN"] = 1
    hdr["GS_YMIN"] = 1
    sci = np.full((n, n), 5.0)
    err = np.ones((n, n), dtype=np.float32)
    dq = np.zeros((n, n), dtype=np.uint32)
    for bit, pix in (dq_bits or {}).items():
        for (y, x) in pix:
            dq[y, x] |= np.uint32(bit)
    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr), fits.ImageHDU(sci, name="SCI"),
                         fits.ImageHDU(err, name="ERR"), fits.ImageHDU(dq, name="DQ")])
    p = tmp_path / name
    hdul.writeto(p)
    return p


def make_truth_index(tmp_path: Path, wcs, stars_xy_1based: list[tuple[float, float]], mags: list[float],
                     name: str = "Roman_TDS_index_H158_10307_3.txt", n_galaxies: int = 2) -> Path:
    """The truth-index layout with 1-based x/y (as galsim writes them) and ra/dec from the WCS."""
    lines = ["#    object_id            ra              dec               x                y          "
             "realized_flux         flux             mag            obj_type     "]
    oid = 40000000000
    for (x, y), m in zip(stars_xy_1based, mags, strict=True):
        ra, dec = wcs.all_pix2world([[x, y]], 1)[0]
        flux = 10.0 ** (-0.4 * m) * 1.0e6
        lines.append(f"{oid} {ra:.14e} {dec:.14e} {x:.8e} {y:.8e} {flux:.6e} {flux:.6e} {m:.8e} star")
        oid += 1
    for g in range(n_galaxies):
        x, y = 20.0 + g, 30.0 + g
        ra, dec = wcs.all_pix2world([[x, y]], 1)[0]
        lines.append(f"{oid} {ra:.14e} {dec:.14e} {x:.8e} {y:.8e} 100.0 100.0 5.0 galaxy")
        oid += 1
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def make_obseq(tmp_path: Path, n_visits: int = 5, cadence: float = 5.0, per_visit: int = 2,
               filters=("R062", "H158")) -> Path:
    """20 rows: two filters x 5 visits x 2 exposures, exposures 0.003 d apart within a visit."""
    from astropy.io import fits
    ra, dec, filt, expt, date, pa = [], [], [], [], [], []
    for f in filters:
        for v in range(n_visits):
            for e in range(per_visit):
                ra.append(9.6 + 0.3 * e)
                dec.append(-44.0)
                filt.append(f)
                expt.append(161.025 if f == "R062" else 302.275)
                date.append(62000.0 + cadence * v + 0.003 * e + (0.1 if f == "H158" else 0.0))
                pa.append(0.0)
    cols = [fits.Column(name="ra", format="D", array=np.array(ra)),
            fits.Column(name="dec", format="D", array=np.array(dec)),
            fits.Column(name="filter", format="4A", array=np.array(filt)),
            fits.Column(name="exptime", format="D", array=np.array(expt)),
            fits.Column(name="date", format="D", array=np.array(date)),
            fits.Column(name="pa", format="D", array=np.array(pa))]
    p = tmp_path / "Roman_TDS_obseq_syn.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(p)
    return p


def make_pointsource(tmp_path: Path, wcs=None, n: int = 4) -> Path:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    xs = [10.0, 40.0, 50.0, 500.0][:n]
    ys = [12.0, 40.0, 20.0, 500.0][:n]
    w = wcs or make_tan_wcs(64)
    sky = w.all_pix2world(np.c_[xs, ys], 0)
    df = pd.DataFrame({"object_type": ["star"] * (n - 1) + ["galaxy"], "id": [f"4097{i}" for i in range(n)],
                       "ra": sky[:, 0], "dec": sky[:, 1], "magnorm": [20.0, 22.0, 24.0, 26.0][:n],
                       "mura": [1.0, 2.0, 3.0, 4.0][:n], "mudec": [0.0, 1.0, 0.0, 1.0][:n],
                       "parallax": [0.1, 0.2, 0.3, 0.4][:n], "variability_model": ["", "", "", ""][:n],
                       "sed_filepath": ["a", "b", "c", "d"][:n]})
    p = tmp_path / "pointsource_10307.parquet"
    df.to_parquet(p)
    return p


# --------------------------------------------------------------------------------------
# Band maps and filename classification
# --------------------------------------------------------------------------------------

def test_band_maps_from_config(conf):
    assert OU.wfi_band_from_letter("F", conf) == "F184"
    assert OU.wfi_band_from_letter("W", conf) == "F146"
    assert OU.wfi_band_from_letter("u", conf) == "lsst_u"
    assert OU.wfi_band_from_letter("-", conf) is None
    assert OU.wfi_band_from_token("R062", conf) == "F062"
    assert OU.wfi_band_from_token("F184", conf) == "F184"
    assert OU.wfi_band_from_token("K213", conf) == "F213"
    assert OU.wfi_band_from_token("XYZ", conf) is None
    assert OU.is_roman_band("F158") and not OU.is_roman_band("lsst_g")
    # the config carries both maps
    assert conf["openuniverse"]["band_letters"]["K"] == "F213"
    assert conf["openuniverse"]["band_tokens"]["Y106"] == "F106"


def test_classify_openuniverse_uri_names_siblings():
    root = "s3://nasa-irsa-simulations/openuniverse2024/roman/full"
    img = OU.classify_openuniverse_uri(f"{root}/RomanTDS/images/simple_model/F184/10307/"
                                       "Roman_TDS_simple_model_F184_10307_17.fits.gz")
    assert img["format"] == "openuniverse_image" and img["band"] == "F184"
    assert img["pointing"] == 10307 and img["sca"] == 17 and img["simulated"] is True
    assert img["truth_index"] == f"{root}/RomanTDS/truth/F184/10307/Roman_TDS_index_F184_10307_17.txt"
    head = OU.classify_openuniverse_uri(f"{root}/ROMAN+LSST_LARGE_SNIa-normal/X_NONIaMODEL0-0001_HEAD.FITS.gz")
    assert head["format"] == "snana_head"
    assert head["phot_sibling"].endswith("X_NONIaMODEL0-0001_PHOT.FITS.gz")
    assert OU.classify_openuniverse_uri("a/b/X_PHOT.FITS.gz")["format"] == "snana_phot"
    assert OU.classify_openuniverse_uri("Roman_TDS_obseq_11_6_23.fits")["format"] == "obseq"
    assert OU.classify_openuniverse_uri("x/pointsource_10307.parquet")["format"] == "pointsource"
    assert OU.classify_openuniverse_uri("x/galaxy_flux_10307.parquet")["format"] == "galaxy_catalog"
    assert OU.classify_openuniverse_uri("x/Roman_TDS_index_H158_1_3.txt")["format"] == "openuniverse_truth_index"
    assert OU.classify_openuniverse_uri("r0000101001001001001_0001_wfi01_cal.asdf") is None


# --------------------------------------------------------------------------------------
# SNANA light curves
# --------------------------------------------------------------------------------------

def test_snana_head_reader(tmp_path):
    head, _ = make_snana_pair(tmp_path)
    df = OU.read_snana_head(head)
    assert list(df["SNID"]) == ["20000001", "20000002", "20000003"]
    assert all(isinstance(v, str) for v in df["SNID"])
    assert (df["NOBS"] == df["PTROBS_MAX"] - df["PTROBS_MIN"] + 1).all()
    assert int(df["PTROBS_MIN"].iloc[0]) == 1
    # the separator row sits between consecutive SNe: pointer gap of 2
    assert (df["PTROBS_MIN"].values[1:] - df["PTROBS_MAX"].values[:-1] == 2).all()
    assert "SIM_PEAKMJD" in df and "REDSHIFT_FINAL" in df


def test_snana_lightcurves_roman_bands_fluxcal_and_assumed_errors(tmp_path, conf):
    head, phot = make_snana_pair(tmp_path, n_sn=3, n_epochs=6)
    lcs = list(OU.iter_snana_lightcurves(head, phot, conf))
    assert all(isinstance(lc, LightCurve) for lc in lcs)
    # 3 SNe x 8 Roman bands, nothing from the LSST letters
    assert len(lcs) == 24
    assert {lc.band for lc in lcs} == {"F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213"}
    assert {lc.star_id for lc in lcs} == {"snana_20000001", "snana_20000002", "snana_20000003"}
    lc = next(lc for lc in lcs if lc.star_id == "snana_20000001" and lc.band == "F184")
    assert lc.flux_unit == "fluxcal" and lc.flux_zp_ab == 27.5
    assert lc.time_system == "MJD_sim" and lc.survey == "HLTDS_sim" and lc.dq is None
    assert lc.meta["simulated"] is True and lc.meta["object_class"] == "SNIa_model"
    assert lc.meta["errors_assumed"] is True and "10**(0.4*(mag-24))" in lc.meta["error_model"]
    assert lc.meta["redshift"] == pytest.approx(0.5, abs=1e-6)
    assert lc.meta["peak_mjd"] == pytest.approx(62010.0)
    assert lc.meta["band_letter"] == "F" and lc.meta["roman_band"] is True
    assert lc.meta["peak_mag"] == pytest.approx(23.0)
    # FLUXCAL convention: flux = 10**(-0.4 (mag - 27.5)); letter F is index 12 -> mag 23.2 at epoch 0
    mag0 = 22.0 + 0.1 * LETTERS.index("F")
    assert lc.flux[0] == pytest.approx(10 ** (-0.4 * (mag0 - 27.5)), rel=1e-5)
    # assumed error model: at mag 23.2 the floor (0.01 mag) applies
    assert lc.flux_err[0] / lc.flux[0] == pytest.approx(0.01 * np.log(10) / 2.5, rel=1e-5)
    assert lc.n == 6 and lc.meta["n_dropped_faint"] == 0 and lc.meta["n_epochs_raw"] == 6
    # the -777 separator never leaks in
    assert np.all(lc.mjd > 60000)
    # magnitudes round-trip through the schema
    _, mag, _ = lc.to_mag()
    assert mag[0] == pytest.approx(mag0, abs=1e-4)


def test_snana_faint_epochs_dropped_and_counted(tmp_path, conf):
    head, phot = make_snana_pair(tmp_path, n_sn=1, n_epochs=6, faint_band="R", n_faint=2)
    lcs = {lc.band: lc for lc in OU.iter_snana_lightcurves(head, phot, conf)}
    assert lcs["F062"].n == 4 and lcs["F062"].meta["n_dropped_faint"] == 2
    assert lcs["F062"].meta["n_epochs_raw"] == 6
    assert lcs["F087"].n == 6 and lcs["F087"].meta["n_dropped_faint"] == 0
    # a lower faint limit drops more
    c2 = dict(conf)
    c2["openuniverse"] = dict(conf["openuniverse"], snana_mag_faint_limit=22.5)
    lcs2 = {lc.band: lc for lc in OU.iter_snana_lightcurves(head, phot, c2)}
    assert "F184" not in lcs2 or lcs2["F184"].n < 6


def test_snana_roman_only_bands_and_max_objects(tmp_path, conf):
    head, phot = make_snana_pair(tmp_path, n_sn=3, n_epochs=4)
    allb = list(OU.iter_snana_lightcurves(head, phot, conf, roman_only=False, max_objects=1))
    assert len(allb) == 14
    lsst = [lc for lc in allb if lc.band.startswith("lsst_")]
    assert len(lsst) == 6 and all(lc.meta["roman_band"] is False and lc.survey == "LSST_sim" for lc in lsst)
    sub = list(OU.iter_snana_lightcurves(head, phot, conf, bands=["F146", "F184"], max_objects=2))
    assert len(sub) == 4 and {lc.band for lc in sub} == {"F146", "F184"}
    assert {lc.star_id for lc in sub} == {"snana_20000001", "snana_20000002"}


def test_snana_gzipped_phot_is_decompressed_once_and_sliced(tmp_path, conf):
    import gzip
    import shutil
    head, phot = make_snana_pair(tmp_path, n_sn=2, n_epochs=3)
    gz = tmp_path / "SYN_0001_PHOT.FITS.gz"
    with open(phot, "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    phot.unlink()
    lcs = list(OU.iter_snana_lightcurves(head, gz, conf))
    assert len(lcs) == 16
    assert (tmp_path / "SYN_0001_PHOT.FITS").exists()       # the memmap-able copy
    # a second pass reuses it
    lcs2 = list(OU.iter_snana_lightcurves(head, gz, conf))
    assert len(lcs2) == 16


def test_snana_error_model_is_the_configured_curve(conf):
    err, desc = OU.snana_error_model(np.array([20.0, 24.0, 26.5, 29.0]), conf)
    assert err[0] == pytest.approx(0.01) and err[1] == pytest.approx(0.01)
    assert err[2] == pytest.approx(0.1, rel=1e-6) and err[3] == pytest.approx(1.0, rel=1e-6)
    assert "0.01" in desc


# --------------------------------------------------------------------------------------
# Images, truth index, dq census
# --------------------------------------------------------------------------------------

def test_truth_index_reader(tmp_path):
    w = make_tan_wcs()
    p = make_truth_index(tmp_path, w, [(10.0, 12.0), (40.0, 40.0)], [4.0, 5.0])
    df = OU.read_truth_index(p)
    assert list(df.columns) == OU.TRUTH_COLUMNS
    assert df["obj_type"].tolist() == ["star", "star", "galaxy", "galaxy"]
    assert all(isinstance(v, str) for v in df["object_id"]) and df["x"].iloc[0] == 10.0


def test_image_cutouts_on_truth_stars_with_measured_pixel_origin(tmp_path, conf):
    w = make_tan_wcs(64)
    jump = int(conf["dq_flags"]["JUMP_DET"])
    sat = int(conf["dq_flags"]["SATURATED"])
    img = make_image(tmp_path, dq_bits={jump: [(5, 5), (5, 6), (40, 41)], sat: [(30, 30)]})
    # five stars, 1-based positions: three inside, two outside (x < 0 and x > 64)
    stars = [(11.0, 13.0), (41.0, 41.0), (51.0, 21.0), (-30.0, 10.0), (200.0, 10.0)]
    mags = [4.0, 5.0, 3.0, 2.0, 2.0]          # AB = mag + ZPTMAG(17.5) = 21.5 / 22.5 / 20.5
    truth = make_truth_index(tmp_path, w, stars, mags)
    rec: dict = {}
    cuts = OU.read_openuniverse_image(img, truth, conf, box=64, image_record=rec)
    assert isinstance(cuts, list) and all(isinstance(c, DQCutout) for c in cuts)
    got = {s["star_id"]: s for c in cuts for s in c.stars}
    assert len(got) == 3
    # 1-based truth -> 0-based detector pixels
    xy = sorted((round(s["x_det"], 6), round(s["y_det"], 6)) for s in got.values())
    assert xy == [(10.0, 12.0), (40.0, 40.0), (50.0, 20.0)]
    c = cuts[0]
    assert c.band == "F158" and c.detector == "SCA3" and c.mjd == pytest.approx(62131.754)
    assert c.exposure_s == pytest.approx(302.275)
    assert c.psf_fwhm_px == pytest.approx(conf["instruments"]["WFI"]["psf_fwhm_px"]["F158"])
    assert c.meta["simulated"] is True and c.meta["origin"] == "openuniverse"
    assert c.meta["zptmag"] == pytest.approx(17.5)
    assert c.meta["truth_xy_origin"] == 1
    assert c.meta["truth_xy_offset"] == pytest.approx([-1.0, -1.0], abs=1e-5)
    assert c.meta["n_stars_in_image"] == 3 and c.meta["n_stars_in_index"] == 5
    assert c.image_id == "Roman_TDS_simple_model_H158_10307_3"
    # AB magnitudes carry the ZPTMAG offset
    assert got["40000000000"]["mag"] == pytest.approx(21.5)
    # cutout dq keeps the flag bits where they were set
    full = next(cc for cc in cuts if cc.x0 == 0 and cc.y0 == 0)
    assert full.dq[5, 5] & jump and full.dq[30, 30] & sat and full.dq[0, 0] == 0
    # the image record carries the whole-plane census and header facts
    assert rec["dq_flag_census"]["per_flag"]["JUMP_DET"] == 3
    assert rec["dq_flag_census"]["per_flag"]["SATURATED"] == 1
    assert rec["dq_flag_census"]["n_nonzero"] == 4 and rec["band"] == "F158"
    assert rec["exptime"] == pytest.approx(302.275) and rec["truth_xy_origin"] == 1


def test_image_mag_limit_and_max_stars(tmp_path, conf):
    w = make_tan_wcs(64)
    img = make_image(tmp_path)
    stars = [(11.0, 13.0), (41.0, 41.0), (51.0, 21.0)]
    truth = make_truth_index(tmp_path, w, stars, [4.0, 5.0, 8.0])     # AB 21.5, 22.5, 25.5
    cuts = OU.read_openuniverse_image(img, truth, conf, box=64, mag_limit=24.0)
    ids = {s["star_id"] for c in cuts for s in c.stars}
    assert ids == {"40000000000", "40000000001"}
    cuts = OU.read_openuniverse_image(img, truth, conf, box=64, mag_limit=30.0, max_stars=1)
    ids = {s["star_id"] for c in cuts for s in c.stars}
    assert ids == {"40000000000"}          # brightest first


def test_image_stars_from_catalogue_through_the_wcs(tmp_path, conf):
    w = make_tan_wcs(64)
    img = make_image(tmp_path)
    ps = OU.read_pointsource_catalog(make_pointsource(tmp_path, w), conf)
    assert ps["object_type"].eq("star").all() and len(ps) == 3
    assert {"pm_ra", "pm_dec", "pm", "parallax", "magnorm", "variability_model"} <= set(ps.columns)
    cuts = OU.read_openuniverse_image(img, None, conf, stars_df=ps, box=64)
    got = {s["star_id"]: s for c in cuts for s in c.stars}
    assert set(got) == {"40970", "40971", "40972"}
    assert got["40970"]["x_det"] == pytest.approx(10.0, abs=1e-6)
    assert got["40970"]["y_det"] == pytest.approx(12.0, abs=1e-6)
    assert got["40970"]["mag"] is None and got["40970"]["magnorm"] == pytest.approx(20.0)
    assert cuts[0].meta["star_source"] == "catalogue_wcs" and cuts[0].meta["truth_xy_origin"] is None


def test_image_without_dq_hdu_is_reader_unavailable(tmp_path, conf):
    from astropy.io import fits

    from seti.roman.products import ReaderUnavailable
    p = tmp_path / "Roman_TDS_simple_model_H158_1_1.fits"
    hdr = make_tan_wcs(16).to_header()
    hdr["FILTER"] = "H158"
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.ImageHDU(np.zeros((16, 16)), name="SCI")]).writeto(p)
    out = OU.read_openuniverse_image(p, None, conf)
    assert isinstance(out, ReaderUnavailable) and "DQ" in out.reason


def test_science_reader_returns_planes(tmp_path):
    img = make_image(tmp_path, n=16)
    sci = OU.read_openuniverse_science(img)
    assert sci["sci"].shape == (16, 16) and sci["err"].dtype == np.float32 and sci["dq"].shape == (16, 16)
    assert sci["header"]["FILTER"] == "H158"


def test_dq_flag_census_counts_bits(conf):
    flags = {"DO_NOT_USE": 1, "SATURATED": 2, "JUMP_DET": 4}
    dq = np.zeros((8, 8), dtype=np.uint32)
    dq[0, 0] = 4
    dq[1, 1] = 6            # SATURATED | JUMP_DET
    dq[2, 2] = 1
    c = OU.dq_flag_census(dq, flags)
    assert c["n_pixels"] == 64 and c["n_nonzero"] == 3
    assert c["per_flag"] == {"DO_NOT_USE": 1, "SATURATED": 1, "JUMP_DET": 2}
    assert c["distinct_values"]["0"] == 61 and c["distinct_values"]["6"] == 1
    z = OU.dq_flag_census(np.zeros((4, 4), dtype=np.uint32), flags)
    assert z["n_nonzero"] == 0 and z["per_flag"]["JUMP_DET"] == 0


# --------------------------------------------------------------------------------------
# Pointing sequence
# --------------------------------------------------------------------------------------

def test_obseq_cadence_record(tmp_path, conf):
    p = make_obseq(tmp_path)
    rec = OU.read_obseq(p, conf)
    assert rec["n_exposures"] == 20 and rec["simulated"] is True
    assert set(rec["filters"]) == {"R062", "H158"}
    assert rec["filters"]["R062"]["wfi_band"] == "F062" and rec["filters"]["R062"]["n_exposures"] == 10
    assert rec["filters"]["R062"]["exptimes_s"] == [161.025]
    assert rec["filters"]["R062"]["n_visits"] == 5
    assert rec["per_filter_median_revisit_days"]["H158"] == pytest.approx(5.0)
    assert rec["hltds_like_cadence_days"] == pytest.approx(5.0)
    assert rec["exptimes_s"] == {"161.025": 10, "302.275": 10}
    assert rec["n_distinct_epochs"] == 20
    assert rec["median_gap_between_distinct_epochs_days"] == pytest.approx(0.003, abs=1e-6)
    assert rec["mjd_min"] == pytest.approx(62000.0) and rec["span_days"] == pytest.approx(20.103, abs=1e-6)
    assert rec["footprint"]["ra_min"] == pytest.approx(9.6) and rec["footprint"]["n_unique_pointings"] == 2


# --------------------------------------------------------------------------------------
# Ingest dispatch through run.py
# --------------------------------------------------------------------------------------

def test_ingest_dispatches_openuniverse_products_and_writes_calibration(tmp_path, monkeypatch, conf):
    from seti.roman import archive, run
    from seti.roman.schema import read_json

    head, phot = make_snana_pair(tmp_path, n_sn=3, n_epochs=4)
    w = make_tan_wcs(64)
    jump = int(conf["dq_flags"]["JUMP_DET"])
    img = make_image(tmp_path, dq_bits={jump: [(5, 5), (6, 6)]})
    truth = make_truth_index(tmp_path, w, [(11.0, 13.0), (41.0, 41.0), (-5.0, 3.0)], [4.0, 5.0, 4.0])
    obseq = make_obseq(tmp_path)
    fetched: list[str] = []

    def stub_fetch(uri, cache_dir, session=None, **kw):
        fetched.append(str(uri))
        p = Path(str(uri))
        return p if p.exists() else None

    monkeypatch.setattr(archive, "fetch_to_cache", stub_fetch)
    inv = {"products": [
        {"uri": str(head), "kind": "unknown", "simulated": True, "size_bytes": head.stat().st_size,
         "meta": {"format": "snana_head", "phot_sibling": str(phot)}},
        # no meta.format: the image and its truth index are inferred from the filename
        {"uri": str(img), "kind": "unknown", "simulated": True, "size_bytes": img.stat().st_size, "meta": {}},
        {"uri": str(obseq), "kind": "obseq", "simulated": True, "size_bytes": 100, "meta": {"format": "obseq"}},
        {"uri": str(tmp_path / "pointsource_10307.parquet"), "kind": "catalog", "simulated": True,
         "size_bytes": 1, "meta": {"format": "pointsource"}},
        {"uri": str(tmp_path / "SYN_0001_PHOT.FITS"), "kind": "unknown", "simulated": True,
         "size_bytes": 1, "meta": {"format": "snana_phot"}},
    ]}
    out = tmp_path / "results"
    cache = tmp_path / "cache"
    rec = run.ingest(out, conf, cache_dir=cache, inventory_record=inv, max_objects_per_product=2)
    assert rec["n_objects_by_kind"]["lightcurve"] == 16         # 2 SNe (max_objects) x 8 Roman bands
    assert rec["n_objects_by_kind"]["dq"] >= 1
    assert rec["per_kind"] == {"lightcurve": 1, "dq": 1, "obseq": 1}
    assert rec["simulated_inputs"] is True and rec["max_objects_per_product"] == 2
    assert rec["catalog_only"][0]["format"] == "pointsource"
    assert rec["funnel"]["counts"]["catalog_only"] == 1
    assert rec["n_unreadable"] == 0, rec["unreadable"]
    assert set(rec["calibration_records"]) == {"survey_cadence_sim", "dq_flag_census_sim", "zero_points_sim"}
    # the PHOT sibling and the truth index were fetched through the same fetcher
    assert str(phot) in fetched and str(truth) in fetched
    assert str(tmp_path / "pointsource_10307.parquet") not in fetched
    manifest = read_json(cache / "manifest.json")["objects"]
    assert manifest and all(m["simulated"] is True for m in manifest)
    assert {m["kind"] for m in manifest} == {"lightcurve", "dq"}
    lc_row = next(m for m in manifest if m["kind"] == "lightcurve")
    lc = run.load_object(lc_row, cache / "normalized")
    assert isinstance(lc, LightCurve) and lc.flux_zp_ab == 27.5 and lc.meta["errors_assumed"] is True
    dq_row = next(m for m in manifest if m["kind"] == "dq")
    cut = run.load_object(dq_row, cache / "normalized")
    assert isinstance(cut, DQCutout) and cut.band == "F158" and len(cut.stars) >= 1
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["survey_cadence_sim"]["hltds_like_cadence_days"] == pytest.approx(5.0)
    assert cal["survey_cadence_sim"]["uri"] == str(obseq)
    assert cal["dq_flag_census_sim"]["n_images"] == 1
    assert cal["dq_flag_census_sim"]["per_flag"]["JUMP_DET"] == 2
    assert cal["dq_flag_census_sim"]["images"][0]["truth_xy_origin"] == 1
    assert cal["zero_points_sim"]["F158"]["n_images"] == 1
    assert cal["written_utc"]
    # a second ingest merges into the file rather than replacing it
    (out / "calibration.json").write_text(json.dumps({**cal, "psf_core_fraction_measured": {"F146": 0.2}}))
    run.ingest(out, conf, cache_dir=cache, inventory_record={"products": inv["products"][2:3]},
               max_objects_per_product=2)
    cal2 = json.loads((out / "calibration.json").read_text())
    assert cal2["psf_core_fraction_measured"] == {"F146": 0.2} and "dq_flag_census_sim" in cal2
    assert cal2["survey_cadence_sim"]["n_exposures"] == 20


def test_ingest_records_a_missing_sibling_as_unreadable(tmp_path, monkeypatch, conf):
    from seti.roman import archive, run
    head, phot = make_snana_pair(tmp_path, n_sn=1, n_epochs=3)
    phot.unlink()
    monkeypatch.setattr(archive, "fetch_to_cache",
                        lambda uri, cache_dir, session=None, **kw: Path(uri) if Path(uri).exists() else None)
    inv = {"products": [{"uri": str(head), "kind": "unknown", "simulated": True, "size_bytes": 10,
                         "meta": {"format": "snana_head", "phot_sibling": str(phot)}}]}
    rec = run.ingest(tmp_path / "r", conf, cache_dir=tmp_path / "c", inventory_record=inv)
    assert rec["n_objects"] == 0 and rec["n_unreadable"] == 1
    assert "PHOT sibling" in rec["unreadable"][0]["reason"]
    assert rec["funnel"]["rejections"]["reader_unavailable"] == 1


def test_kind_of_reads_format_then_falls_back_to_filename():
    from seti.roman.run import _kind_of
    assert _kind_of({"uri": "x/A_HEAD.FITS.gz", "kind": "unknown", "meta": {}}) == "lightcurve"
    assert _kind_of({"uri": "x/A_PHOT.FITS.gz", "kind": "unknown", "meta": {}}) is None
    assert _kind_of({"uri": "x/Roman_TDS_simple_model_F184_1_2.fits.gz", "kind": "unknown"}) == "dq"
    assert _kind_of({"uri": "x/Roman_TDS_index_F184_1_2.txt", "kind": "unknown"}) is None
    assert _kind_of({"uri": "x/Roman_TDS_obseq_11_6_23.fits", "kind": "unknown"}) == "obseq"
    assert _kind_of({"uri": "x/pointsource_1.parquet", "kind": "catalog"}) == "catalog"
    assert _kind_of({"uri": "x/r_cal.asdf", "kind": "cal"}) == "dq"
    assert _kind_of({"uri": "x/lc.parquet", "kind": "lightcurve"}) == "lightcurve"
    assert _kind_of({"uri": "x/thing.asdf", "kind": "unknown"}) is None


def test_cli_accepts_max_objects_flag():
    import argparse

    from seti.roman.run import _add_args
    p = argparse.ArgumentParser()
    _add_args(p)
    a = p.parse_args(["ingest", "--max-objects-per-product", "7"])
    assert a.max_objects_per_product == 7
    assert "obseq" in p.parse_args(["ingest"]).kinds


# --------------------------------------------------------------------------------------
# Real sample files (skipped when absent; never on CI)
# --------------------------------------------------------------------------------------

@pytest.mark.skipif(not (REAL_HEAD.exists() and REAL_PHOT.exists()), reason="real SNANA sample absent")
def test_real_snana_first_sn_has_eight_roman_bands_of_295_epochs(conf):
    head = OU.read_snana_head(REAL_HEAD)
    assert int(head["PTROBS_MIN"].iloc[0]) == 1
    assert (head["NOBS"] == head["PTROBS_MAX"] - head["PTROBS_MIN"] + 1).all()
    lcs = list(OU.iter_snana_lightcurves(REAL_HEAD, REAL_PHOT, conf, max_objects=1))
    assert len(lcs) == 8
    assert {lc.band for lc in lcs} == {"F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213"}
    for lc in lcs:
        assert lc.meta["n_epochs_raw"] == 295
        assert lc.n + lc.meta["n_dropped_faint"] == 295
        assert lc.meta["errors_assumed"] is True and lc.flux_zp_ab == 27.5
        assert np.all(lc.mjd > 61000)
    # the separator convention: one -777 row per SN, just past PTROBS_MAX
    from astropy.io import fits
    with fits.open(OU._materialise_phot(REAL_PHOT), memmap=True) as h:
        d = h[1].data
        pmax = int(head["PTROBS_MAX"].iloc[0])
        assert float(d["MJD"][pmax]) == -777.0 and float(d["MJD"][pmax - 1]) > 0


@pytest.mark.skipif(not (REAL_IMAGE.exists() and REAL_TRUTH.exists()), reason="real image sample absent")
def test_real_image_truth_positions_agree_with_the_sip_wcs(conf):
    from seti.roman.products import stars_to_pixels
    rec: dict = {}
    cuts = OU.read_openuniverse_image(REAL_IMAGE, REAL_TRUTH, conf, image_record=rec, max_stars=50)
    assert isinstance(cuts, list) and cuts
    assert rec["band"] == "F184" and rec["detector"] == "SCA17" and rec["has_sip"] is True
    assert rec["truth_xy_origin"] == 1
    assert rec["truth_xy_offset"] == pytest.approx([-1.0, -1.0], abs=1e-3)
    hdr = OU.read_openuniverse_header(REAL_IMAGE)
    stars = [s for c in cuts for s in c.stars]
    x, y = stars_to_pixels(hdr["wcs"], [s["ra"] for s in stars], [s["dec"] for s in stars])
    resid = np.hypot(x - np.array([s["x_det"] for s in stars]), y - np.array([s["y_det"] for s in stars]))
    assert resid.max() < 1.0
    # the implied per-e-/s zero point agrees with the config value to 0.2 mag
    assert abs(rec["zp_ab_implied_per_e_s"] - conf["instruments"]["WFI"]["filters"]["F184"]["zp_ab"]) < 0.2
    # the sim carries no jump flags at all -- recorded, not assumed
    assert rec["dq_flag_census"]["per_flag"]["JUMP_DET"] == 0 and rec["dq_all_zero"] is True
    assert cuts[0].psf_fwhm_px == pytest.approx(1.3) and cuts[0].exposure_s == pytest.approx(901.175)


@pytest.mark.skipif(not REAL_OBSEQ.exists(), reason="real obseq sample absent")
def test_real_obseq_has_seven_filters_at_a_five_day_cadence(conf):
    rec = OU.read_obseq(REAL_OBSEQ, conf)
    assert rec["n_exposures"] == 57365 and len(rec["filters"]) == 7
    assert {v["wfi_band"] for v in rec["filters"].values()} == {"F062", "F087", "F106", "F129", "F158",
                                                                 "F184", "F213"}
    cadence = rec["hltds_like_cadence_days"]
    print(f"OpenUniverse TDS per-filter revisit: {rec['per_filter_median_revisit_days']} -> {cadence} d")
    assert 3.0 < cadence < 8.0
    assert all(3.0 < v < 8.0 for v in rec["per_filter_median_revisit_days"].values())


@pytest.mark.skipif(not REAL_POINTSOURCE.exists(), reason="real pointsource sample absent")
def test_real_pointsource_catalogue_is_all_stars(conf):
    ps = OU.read_pointsource_catalog(REAL_POINTSOURCE, conf, max_rows=100)
    assert len(ps) == 100 and ps["object_type"].eq("star").all()
    assert all(isinstance(v, str) for v in ps["id"]) and ps["pm"].gt(0).all()
