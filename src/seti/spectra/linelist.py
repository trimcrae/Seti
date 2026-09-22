"""Comprehensive rest-frame line identification for narrow-line survivors.

The in-funnel and triage line lists are deliberately short (the strongest
features only).  That leaves two leaks which the per-exposure persistence test
cannot close, because a real stellar line *is* persistent:

* **series members the short lists stop at** --- the absorption list carries the
  Balmer series only down to H9 (3835 A); a hot star with a catalogue z of
  +0.001 (300 km/s) puts H11 at 3775.5 A and H12 at 3755.5 A, exactly where two
  DESI "survivors" sit;
* **lines that are only ever strong in one class of star** --- Fe II / [Fe II]
  fluorescence in Miras, the C2 Swan and CN bands of carbon stars, the TiO / CaH
  / VO band heads whose *gaps* the matched filter reads as unresolved emission
  in M dwarfs, the N I / C I / S I multiplets of A stars in the far red.

This module carries a labelled, species-tagged list (air values converted to the
vacuum scale of the survey spectra at definition time) and tests each survivor
at **its own catalogue redshift** (stellar frame) and at z = 0 (ISM, sky, DIB,
city-light frame).  A match is a kill; a non-match is reported with the nearest
line and its velocity so the reader can judge the margin.  On the runner the
NIST ASD can additionally be queried for *context* (the strongest catalogued
atomic lines within a few Angstrom); that is information, not a kill, because
every optical wavelength is within a few Angstrom of *some* weak NIST line.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .reject import air_to_vacuum

C_KMS = 299792.458

# (air wavelength A, label, frame) -- frame "star" (shift by the star's z) or
# "obs" (fixed in the observed frame: airglow, ISM, DIB, street lamps).
_STAR = "star"
_OBS = "obs"

_LINES_AIR: list[tuple[float, str, str]] = []


def _add(frame: str, label: str, waves) -> None:
    for w in waves:
        _LINES_AIR.append((float(w), label, frame))


# --- hydrogen ------------------------------------------------------------------
_add(_STAR, "H Balmer", [6562.79, 4861.33, 4340.47, 4101.73, 3970.07, 3889.05,
                         3835.38, 3797.90, 3770.63, 3750.15, 3734.37, 3721.94,
                         3711.97, 3703.86, 3697.15, 3691.56, 3686.83, 3682.81])
_add(_STAR, "H Paschen", [9545.97, 9229.01, 9014.91, 8862.78, 8750.47, 8665.02,
                          8598.39, 8545.38, 8502.48, 8467.25, 8437.96, 8413.32,
                          8392.40, 8374.48, 8359.00, 8345.55, 8333.78, 8323.42])
# --- helium ------------------------------------------------------------------
_add(_STAR, "He I", [3819.61, 3867.48, 3888.65, 3964.73, 4009.27, 4026.19, 4120.82,
                     4143.76, 4387.93, 4437.55, 4471.48, 4713.15, 4921.93, 5015.68,
                     5047.74, 5875.62, 6678.15, 7065.19, 7281.35])
_add(_STAR, "He II", [4199.83, 4541.59, 4685.68, 5411.52, 6560.10])
# --- light metals ---------------------------------------------------------------
_add(_STAR, "Ca I", [4226.73, 4283.01, 4289.37, 4302.53, 4318.65, 4425.44, 4434.96,
                     4454.78, 5588.76, 5594.47, 5857.45, 6102.72, 6122.22, 6162.17,
                     6169.04, 6439.08, 6449.81, 6462.57, 6471.66, 6493.78, 6499.65,
                     6717.68, 7148.15, 7326.15])
_add(_STAR, "Ca II", [3933.66, 3968.47, 8498.02, 8542.09, 8662.14])
_add(_STAR, "Mg I", [3829.36, 3832.30, 3838.29, 4571.10, 4702.99, 5167.32, 5172.68,
                     5183.60, 5528.41, 5711.09, 8806.76])
_add(_STAR, "Mg II", [4481.13, 4481.33, 7877.13, 7896.37])
_add(_STAR, "Na I", [5889.95, 5895.92, 5682.63, 5688.20, 6154.23, 6160.75, 8183.26,
                     8194.82])
_add(_STAR, "K I", [7664.90, 7698.96, 4044.14, 4047.21])
_add(_STAR, "Al I", [3944.01, 3961.52, 6696.02, 6698.67])
_add(_STAR, "Si I", [3905.52, 4102.94, 5701.10, 5772.15, 5948.54, 7405.77, 7423.50,
                     8728.01, 8742.45, 8752.01])
_add(_STAR, "Si II", [6347.11, 6371.37, 4128.07, 4130.89])
_add(_STAR, "Li I", [6707.76])
# --- iron peak ------------------------------------------------------------------
_add(_STAR, "Fe I", [3719.93, 3722.56, 3733.32, 3734.86, 3737.13, 3745.56, 3748.26,
                     3758.23, 3763.79, 3767.19, 3795.00, 3815.84, 3820.43, 3824.44,
                     3825.88, 3827.82, 3834.22, 3840.44, 3841.05, 3856.37, 3859.91,
                     3865.52, 3878.02, 3878.57, 3886.28, 3895.66, 3899.71, 3902.95,
                     3906.48, 3920.26, 3922.91, 3927.92, 3930.30, 4005.24, 4045.81,
                     4063.59, 4071.74, 4132.06, 4143.87, 4202.03, 4216.18, 4250.79,
                     4260.47, 4271.76, 4282.40, 4307.90, 4325.76, 4383.55, 4404.75,
                     4415.12, 4427.31, 4461.65, 4482.17, 4528.61, 4871.32, 4891.49,
                     4920.50, 4957.60, 5041.76, 5051.63, 5110.41, 5139.25, 5167.49,
                     5171.60, 5227.19, 5232.94, 5266.56, 5269.54, 5270.36, 5324.18,
                     5328.04, 5328.53, 5341.02, 5371.49, 5397.13, 5405.77, 5424.07,
                     5429.70, 5434.52, 5446.92, 5455.61, 5497.52, 5501.47, 5506.78,
                     5569.62, 5572.84, 5586.76, 5615.64, 6136.62, 6137.69, 6191.56,
                     6230.72, 6252.56, 6393.60, 6400.00, 6411.65, 6421.35, 6430.85,
                     6494.98, 6592.91, 6677.99, 7511.02, 8327.06, 8387.77, 8688.62,
                     8824.22, 8688.63])
_add(_STAR, "Fe II", [4173.46, 4178.86, 4233.17, 4303.18, 4351.77, 4385.39, 4416.83,
                      4508.29, 4515.34, 4520.22, 4522.63, 4549.47, 4555.89, 4583.84,
                      4629.34, 4923.93, 5018.44, 5169.03, 5197.58, 5234.63, 5276.00,
                      5316.62, 5362.87, 6247.56, 6456.38, 6516.08])
_add(_STAR, "[Fe II]", [4243.98, 4276.83, 4287.39, 4359.34, 4413.78, 4416.27, 4452.10,
                        4457.95, 4474.91, 4814.53, 4874.49, 4889.63, 4905.35, 5158.78,
                        5261.62, 5273.35, 5333.65, 5376.45, 7155.16, 7172.00, 7388.17,
                        7452.54, 8616.95, 8891.91])
_add(_STAR, "Ti I", [3998.64, 4533.24, 4534.78, 4981.73, 4991.07, 4999.50, 5007.21,
                     5014.19, 5036.46, 5039.96, 5064.65, 5173.74, 5192.97, 5210.39,
                     5866.45, 6258.10, 6261.10])
_add(_STAR, "Ti II", [3759.29, 3761.32, 4290.22, 4300.04, 4301.92, 4395.03, 4399.77,
                      4443.80, 4468.49, 4501.27, 4533.96, 4549.62, 4571.97, 5129.16,
                      5188.69, 5226.54, 5336.79])
_add(_STAR, "Cr I", [4254.33, 4274.80, 4289.72, 5204.51, 5206.04, 5208.42, 5345.80,
                     5409.78])
_add(_STAR, "Cr II", [4558.65, 4588.20, 4618.80, 4824.13])
_add(_STAR, "Mn I", [4030.75, 4033.06, 4034.48, 4041.35, 4754.04, 4783.42, 4823.51,
                     6013.49, 6016.64, 6021.79])
_add(_STAR, "Ni I", [5476.90, 5754.66, 6643.63, 6767.77])
_add(_STAR, "V I", [4379.24, 4384.71, 4389.97, 4406.64, 6090.21, 6119.53])
_add(_STAR, "Co I", [3845.47, 3873.95, 4121.32])
_add(_STAR, "Sc II", [4246.82, 4320.73, 4400.39, 5031.02, 5526.79, 6604.60])
_add(_STAR, "Cu I", [5105.54, 5218.20, 5782.13])
_add(_STAR, "Zn I", [4722.15, 4810.53])
# --- heavy elements (s-process stars) --------------------------------------------
_add(_STAR, "Sr II", [4077.71, 4215.52])
_add(_STAR, "Sr I", [4607.33, 4872.49, 7070.07])
_add(_STAR, "Y II", [4883.68, 5087.42, 5200.41])
_add(_STAR, "Zr I", [4772.31, 4805.87, 6127.44, 6134.55, 6143.18])
_add(_STAR, "Ba II", [4554.03, 4934.08, 5853.67, 6141.71, 6496.90])
_add(_STAR, "La II", [3949.10, 4086.71, 4123.22, 4322.51, 6390.48])
_add(_STAR, "Ce II", [4186.60, 4562.36, 4628.16])
_add(_STAR, "Nd II", [4061.08, 4109.45, 5319.81])
_add(_STAR, "Eu II", [4129.72, 4205.04, 6645.10])
_add(_STAR, "Tc I", [4238.19, 4262.27, 4297.06])
# --- CNO and sulphur (A-type, far red) ----------------------------------------------
_add(_STAR, "O I", [7771.94, 7774.17, 7775.39, 8446.36, 8446.76, 6155.99, 6156.78,
                    6158.19, 9260.85, 9262.67, 9266.01])
_add(_STAR, "N I", [8680.28, 8683.40, 8686.15, 8703.25, 8711.70, 8718.83, 8728.90,
                    7423.64, 7442.30, 7468.31, 8184.86, 8188.01, 8216.34, 8223.13,
                    8242.39, 9392.79])
_add(_STAR, "C I", [9061.43, 9062.47, 9078.28, 9088.51, 9094.83, 9111.81, 8335.15,
                    4771.74, 5052.17, 6587.61, 7111.47, 7113.18, 7115.17, 7116.99])
_add(_STAR, "S I", [9212.87, 9228.09, 9237.54, 8693.93, 8694.62, 6743.58, 6748.79,
                    6757.15, 4694.11, 4695.44])
# --- nebular / forbidden (emission-line stars, CVs, misclassified galaxies) -------
_add(_STAR, "[O II]", [3726.03, 3728.82, 7319.99, 7330.73])
_add(_STAR, "[O III]", [4363.21, 4958.91, 5006.84])
_add(_STAR, "[O I]", [5577.34, 6300.30, 6363.78])
_add(_STAR, "[N II]", [5754.64, 6548.05, 6583.45])
_add(_STAR, "[S II]", [4068.60, 4076.35, 6716.44, 6730.82])
_add(_STAR, "[S III]", [6312.06, 9068.60, 9530.60])
_add(_STAR, "[Ne III]", [3868.76, 3967.47])
_add(_STAR, "[Ar III]", [7135.79, 7751.11])
_add(_STAR, "[Ca II]", [7291.47, 7323.89])
_add(_STAR, "[Ni II]", [7377.83, 7411.61])
_add(_STAR, "[N I]", [5197.90, 5200.26])
_add(_STAR, "[Cr II]", [8000.08, 8125.30, 8229.67])
# --- molecular band heads (M / S / C stars; the GAPS read as pseudo-emission) ------
_add(_STAR, "TiO head", [4954.6, 5167.2, 5448.1, 5598.4, 5759.4, 5810.0, 5847.0,
                         6158.5, 6187.7, 6215.9, 6651.2, 6681.4, 6714.4, 7054.2,
                         7087.6, 7125.5, 7589.3, 7627.9, 7666.3, 8432.1, 8442.4,
                         8859.4, 8937.2, 9209.9])
_add(_STAR, "VO head", [7334.0, 7393.0, 7851.0, 7896.0, 7940.0, 8521.0, 8624.0])
_add(_STAR, "CaH head", [6382.0, 6750.0, 6908.0, 6950.0])
_add(_STAR, "MgH head", [4780.0, 5211.0])
_add(_STAR, "CH G band", [4300.0, 4323.0])
_add(_STAR, "CN head", [3590.0, 3883.4, 4216.0, 6206.0, 6332.0, 6478.0, 6926.0,
                        7088.0, 7259.0, 7437.0, 7876.0, 8125.0, 9140.0, 9186.0])
_add(_STAR, "C2 Swan", [4382.5, 4737.1, 5165.2, 5635.5, 6122.1, 6191.2])
_add(_STAR, "ZrO head", [5551.0, 5718.0, 5849.0, 6229.0, 6378.0, 6474.0, 6495.0,
                         6925.0])
_add(_STAR, "YO head", [5972.0, 6132.0])
# --- observed-frame: ISM, DIBs ---------------------------------------------------
_add(_OBS, "ISM Na I", [5889.95, 5895.92])
_add(_OBS, "ISM Ca II", [3933.66, 3968.47])
_add(_OBS, "ISM K I", [7664.90, 7698.96])
_add(_OBS, "ISM CH", [4300.31])
_add(_OBS, "ISM CH+", [4232.55])
_add(_OBS, "DIB", [4428.0, 4726.8, 4762.6, 4780.0, 4963.9, 5487.7, 5494.1, 5705.1,
                   5780.5, 5797.1, 5849.8, 6010.6, 6089.8, 6113.2, 6195.9, 6203.6,
                   6269.8, 6283.8, 6376.1, 6379.3, 6425.7, 6439.5, 6445.3, 6613.6,
                   6660.7, 6699.3, 6993.2, 7224.0, 7367.1, 7558.4, 7562.2, 7581.7,
                   8621.0, 8648.0])
# --- observed-frame: airglow (OH Meinel, O2, [O I], Na) and city lights -----------
_add(_OBS, "sky [O I]", [5577.34, 6300.30, 6363.78])
_add(_OBS, "sky Na", [5889.95, 5895.92])
_add(_OBS, "sky OH", [
    5238.7, 5243.7, 5251.5, 5312.3, 5321.9, 5331.9, 5343.6, 5355.9, 5439.8, 5482.4,
    5496.9, 5524.0, 5540.1, 5555.7, 5567.3, 5866.2, 5871.4, 5878.3, 5889.9, 5901.5,
    5913.6, 5929.2, 5945.6, 5963.3, 5983.5, 6004.6, 6027.8, 6051.6, 6076.6, 6103.8,
    6135.1, 6193.7, 6204.5, 6222.1, 6235.9, 6247.9, 6257.9, 6264.9, 6287.6, 6296.9,
    6306.9, 6321.4, 6329.7, 6363.9, 6386.4, 6449.7, 6465.5, 6478.4, 6498.7, 6504.0,
    6522.4, 6533.0, 6544.0, 6553.6, 6562.0, 6568.8, 6577.2, 6596.6, 6604.1, 6627.6,
    6634.2, 6661.9, 6685.1, 6704.5, 6742.6, 6753.3, 6863.9, 6871.1, 6889.3, 6900.8,
    6912.6, 6923.2, 6939.5, 6948.9, 6978.4, 7003.9, 7238.8, 7240.2, 7244.9, 7276.4,
    7284.4, 7303.7, 7316.3, 7329.1, 7340.9, 7358.7, 7369.3, 7392.2, 7402.0, 7524.1,
    7571.7, 7712.2, 7715.0, 7750.6, 7759.9, 7794.1, 7808.5, 7821.5, 7841.3, 7853.3,
    7913.7, 7921.1, 7949.2, 7964.7, 7993.3, 8025.7, 8062.0, 8288.6, 8298.4, 8310.7,
    8344.6, 8352.8, 8382.4, 8399.2, 8415.2, 8430.2, 8452.3, 8465.4, 8493.3, 8505.0,
    8538.9, 8548.5, 8620.8, 8655.0, 8761.4, 8767.1, 8778.4, 8791.2, 8827.1, 8836.1,
    8849.6, 8885.9, 8903.1, 8919.6, 8943.4, 8958.1, 8988.4, 9002.0, 9038.0, 9049.3,
    9375.9, 9419.7, 9439.7, 9458.5, 9476.9, 9505.7, 9521.9, 9555.0, 9569.5, 9607.6,
    9622.1, 9720.7, 9740.4, 9763.0, 9789.5, 9799.9,
])
_add(_OBS, "sky O2 (0-1) band", [8645.4, 8654.1, 8660.5, 8665.7, 8670.2, 8676.0,
                                  8684.6, 8691.1, 8696.4, 8702.9, 8709.7, 8716.7,
                                  8723.9])
_add(_OBS, "sky O2 (0-0) band", [7596.4, 7621.0, 7643.5, 7665.0, 7683.0])
_add(_OBS, "city Hg I", [4046.56, 4358.33, 5460.74, 5769.60, 5790.66])
_add(_OBS, "city Na (HPS)", [5682.63, 5688.20, 6154.23, 6160.75, 8183.26, 8194.82])

_WAVE_VAC = air_to_vacuum(np.array([w for w, _, _ in _LINES_AIR]))
_LABEL = np.array([lbl for _, lbl, _ in _LINES_AIR])
_FRAME = np.array([fr for _, _, fr in _LINES_AIR])

# Species that appear in EMISSION in stars (chromospheres, flares, winds, shells,
# Miras, CVs, Be/Ae stars) or whose band-head gaps mimic emission, plus every
# observed-frame feature.  An emission-mode survivor is killed only by these; a
# photospheric Fe I / Ni I / Ti I absorption line a few hundred km/s away is not
# an explanation for an emission spike and is reported, not enforced.
EMISSION_CAPABLE = {
    "H Balmer", "H Paschen", "He I", "He II", "Ca II", "Na I", "Mg I", "Mg II", "K I",
    "O I", "N I", "C I", "Si II", "Fe II", "[Fe II]", "[O II]", "[O III]", "[O I]", "[N II]",
    "[S II]", "[S III]", "[Ne III]", "[Ar III]", "[Ca II]", "[Ni II]", "[N I]", "[Cr II]",
    "Ti I", "Ti II", "Cr I", "Cr II", "Ba II", "Sr II", "Li I",
    "TiO head", "VO head", "CaH head", "MgH head", "CH G band", "CN head", "C2 Swan",
    "ZrO head", "YO head",
}
# Pure emission features cannot explain an ABSORPTION survivor.
EMISSION_ONLY = {"[Fe II]", "[O II]", "[O III]", "[O I]", "[N II]", "[S II]", "[S III]",
                 "[Ne III]", "[Ar III]", "[Ca II]", "[Ni II]", "[N I]", "[Cr II]",
                 "sky [O I]", "sky Na", "sky OH", "sky O2 (0-1) band", "sky O2 (0-0) band",
                 "city Hg I", "city Na (HPS)"}


def n_lines() -> int:
    return int(_WAVE_VAC.size)


def identify_rest_frame(lam_obs: float, z: float = 0.0, tol_kms: float = 300.0,
                        mode: str = "emission") -> dict:
    """Best known-line match for an observed (vacuum) wavelength.

    Star-frame lines are shifted by the catalogue redshift ``z``; observed-frame
    lines (sky, ISM, DIB, city lights) are not.  ``known_line_match`` is True
    only for a line that can physically produce the searched feature (emission
    mode: emission-capable species, band heads and sky; absorption mode: every
    photospheric / ISM / DIB line, not pure-emission features) within
    ``tol_kms``.  The nearest line of *any* kind is reported alongside, as is
    the closest star-frame line at z = 0 (a catalogue RV can be wrong; that is
    information, not a kill).
    """
    lam = float(lam_obs)
    z = float(z) if np.isfinite(z) else 0.0
    star = _FRAME == _STAR
    shifted = np.where(star, _WAVE_VAC * (1.0 + z), _WAVE_VAC)
    dv = (lam - shifted) / shifted * C_KMS
    if mode == "absorption":
        eligible = ~np.isin(_LABEL, sorted(EMISSION_ONLY))
    else:
        eligible = np.isin(_LABEL, sorted(EMISSION_CAPABLE)) | (~star)
    k_any = int(np.argmin(np.abs(dv)))
    k = int(np.argmin(np.where(eligible, np.abs(dv), np.inf)))
    dv0 = (lam - _WAVE_VAC) / _WAVE_VAC * C_KMS
    k0 = int(np.argmin(np.abs(np.where(star, dv0, np.inf))))
    return {
        "known_line_match": bool(abs(dv[k]) <= tol_kms),
        "known_line_label": str(_LABEL[k]),
        "known_line_wave_vac": round(float(_WAVE_VAC[k]), 2),
        "known_line_dv_kms": round(float(dv[k]), 1),
        "known_line_frame": str(_FRAME[k]),
        "nearest_any_label": str(_LABEL[k_any]),
        "nearest_any_dv_kms": round(float(dv[k_any]), 1),
        "z0_star_label": str(_LABEL[k0]),
        "z0_star_dv_kms": round(float(dv0[k0]), 1),
    }


# ---------------------------------------------------------------------------
# NIST ASD context (runner-side; network)
# ---------------------------------------------------------------------------

_NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
NIST_SPECIES = ["H I", "He I", "He II", "C I", "N I", "O I", "Na I", "Mg I", "Mg II",
                "Al I", "Si I", "Si II", "S I", "K I", "Ca I", "Ca II", "Sc II",
                "Ti I", "Ti II", "V I", "Cr I", "Cr II", "Mn I", "Fe I", "Fe II",
                "Co I", "Ni I", "Cu I", "Zn I", "Sr I", "Sr II", "Y II", "Zr I",
                "Zr II", "Ba II", "La II", "Ce II", "Nd II", "Eu II", "Li I"]


def _parse_intensity(s: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else float("nan")


def fetch_nist_species(species: str, lo: float = 3500.0, hi: float = 9900.0,
                       timeout: float = 90.0) -> list[dict]:
    """Observed-wavelength NIST ASD lines (air A -> vacuum) with intensities."""
    import requests

    params = {
        "spectra": species, "low_w": f"{lo:.1f}", "upp_w": f"{hi:.1f}", "unit": 0,
        "format": 3, "line_out": 0, "en_unit": 0, "output": 0, "bibrefs": 0,
        "page_size": 15, "show_obs_wl": 1, "show_calc_wl": 0, "unc_out": 0,
        "order_out": 0, "show_av": 2, "tsb_value": 0, "A_out": 0, "intens_out": "on",
        "allowed_out": 1, "forbid_out": 1, "remove_js": "on",
    }
    r = requests.get(_NIST_URL, params=params, timeout=timeout)
    r.raise_for_status()
    text = r.text
    if "<html" in text[:200].lower():
        text = re.sub(r"<[^>]+>", "", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = None
    out: list[dict] = []
    for ln in lines:
        cells = [c.strip().strip('"') for c in ln.split("\t")]
        if header is None:
            low = [c.lower() for c in cells]
            if any("obs_wl" in c for c in low) or any("wavelength" in c for c in low):
                header = low
            continue
        if len(cells) < 2:
            continue
        iw = next((i for i, c in enumerate(header) if "obs_wl" in c or "wavelength" in c), 0)
        ii = next((i for i, c in enumerate(header) if "intens" in c), None)
        try:
            w = float(cells[iw])
        except (ValueError, IndexError):
            continue
        if not (lo - 1 <= w <= hi + 1):
            continue
        inten = _parse_intensity(cells[ii]) if ii is not None and ii < len(cells) else float("nan")
        out.append({"species": species, "wave_air": w,
                    "wave_vac": round(float(air_to_vacuum(w)), 3),
                    "intensity": inten})
    return out


def build_nist_cache(path: Path, species: list[str] | None = None,
                     lo: float = 3500.0, hi: float = 9900.0) -> dict:
    """Fetch every species once and cache to ``path`` (idempotent)."""
    import time
    path = Path(path)
    cache: dict = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text())
        except Exception:
            cache = {}
    for sp in species or NIST_SPECIES:
        if sp in cache and cache[sp].get("status") == "ok":
            continue
        try:
            rows = fetch_nist_species(sp, lo, hi)
            cache[sp] = {"status": "ok", "n": len(rows),
                         "lines": [[r["wave_vac"], r["intensity"]] for r in rows]}
            print(f"[linelist] NIST {sp}: {len(rows)} lines")
        except Exception as exc:  # noqa: BLE001
            cache[sp] = {"status": f"failed: {exc!r}", "n": 0, "lines": []}
            print(f"[linelist] NIST {sp} failed: {exc!r}")
        time.sleep(1.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return cache


def nist_context(cache: dict, lam_obs: float, z: float = 0.0, win_A: float = 4.0,
                 top: int = 3) -> str:
    """The strongest catalogued atomic lines within ``win_A`` (star frame)."""
    lam_rest = float(lam_obs) / (1.0 + (float(z) if np.isfinite(z) else 0.0))
    hits = []
    for sp, d in cache.items():
        for w, inten in d.get("lines", []):
            if abs(w - lam_rest) <= win_A:
                hits.append((inten if np.isfinite(inten) else -1.0, sp, w))
    hits.sort(reverse=True)
    return "; ".join(f"{sp} {w:.2f} (I={inten:.0f})" if inten >= 0 else f"{sp} {w:.2f}"
                     for inten, sp, w in hits[:top])


__all__ = ["identify_rest_frame", "n_lines", "fetch_nist_species", "build_nist_cache",
           "nist_context", "NIST_SPECIES"]
