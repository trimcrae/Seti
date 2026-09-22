"""PDS3 / PDS4 labels, IMG rasters and map georeferencing for CRYPT.

Everything here is a pure function of bytes on disk: no network.  The Diviner
level-3/4 volume ``lrodlr_1002`` carries PDS3 ``.lbl`` labels (detached or
attached) and was migrated to PDS4 (``.xml``) in 2022–23; ShadowCam is PDS4
from birth.  Both label dialects are parsed into one :class:`RasterMeta`, so
the physics never sees which archive a raster came from.

Georeferencing is deliberately small: the polar products are polar
stereographic on a 1737.4 km sphere; global mosaics are equirectangular.
The PDS3 pixel convention (``LINE_PROJECTION_OFFSET`` /
``SAMPLE_PROJECTION_OFFSET`` are the line/sample of the projection origin
measured from pixel (1, 1)) is carried in one place, :meth:`Georef.pix_to_lonlat`,
and the ``origin_check`` field records how far the parsed origin sits from the
raster centre, because a mis-read convention shows up there first.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MOON_RADIUS_M = 1_737_400.0

# ---------------------------------------------------------------------------
# PDS3
# ---------------------------------------------------------------------------
_KV_RX = re.compile(r"^\s*(\^?[A-Za-z0-9_:]+)\s*=\s*(.*)$")


def _strip_units(v: str) -> tuple[str, str | None]:
    m = re.match(r"^(.*?)\s*<([^>]*)>\s*$", v.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return v.strip(), None


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def parse_pds3_label(text: str) -> dict:
    """Flat ``{KEY: value}`` plus ``{OBJECT.KEY: value}`` for every object level.

    Values are strings with quotes removed; a unit in angle brackets is kept
    under ``KEY__unit``.  Parenthesised lists become Python lists of strings.
    Continuation lines (a value that opens a quote or bracket and does not
    close it) are joined.  The parse stops at ``END``.
    """
    out: dict = {}
    stack: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.strip().startswith("/*") or not line.strip():
            continue
        if line.strip() == "END":
            break
        m = _KV_RX.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        # join continuation lines for unbalanced quotes / parentheses
        while (val.count('"') % 2 == 1 or val.count("(") > val.count(")")) and i < len(lines):
            val += " " + lines[i].strip()
            i += 1
        val = val.split("/*")[0].strip() if not val.startswith('"') else val.strip()
        if key in ("OBJECT", "GROUP"):
            stack.append(_unquote(val))
            continue
        if key in ("END_OBJECT", "END_GROUP"):
            if stack:
                stack.pop()
            continue
        raw, unit = _strip_units(val)
        if raw.startswith("(") and raw.endswith(")"):
            parsed: object = [_unquote(x) for x in re.split(r",\s*", raw[1:-1].strip()) if x.strip()]
        else:
            parsed = _unquote(raw)
        prefix = ".".join(stack)
        full = f"{prefix}.{key}" if prefix else key
        out[full] = parsed
        if unit is not None:
            out[full + "__unit"] = unit
        # first-seen wins for the bare key so top-level keys are not shadowed
        out.setdefault(key, parsed)
        if unit is not None:
            out.setdefault(key + "__unit", unit)
    return out


def _f(d: dict, *keys: str, default=None):
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            v = v[0] if v else None
        try:
            return float(str(v).strip().strip('"'))
        except (TypeError, ValueError):
            continue
    return default


def _s(d: dict, *keys: str, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v if not isinstance(v, list) else (v[0] if v else default)
    return default


# ---------------------------------------------------------------------------
# the one raster description both dialects produce
# ---------------------------------------------------------------------------
@dataclass
class Georef:
    """Map geometry of a raster.  ``projection`` is ``polar_stereographic`` or
    ``equirectangular``; ``map_scale_m`` is metres per pixel at the reference
    latitude; offsets are the 0-based fractional pixel coordinates of the
    projection origin (pole, or lon0/lat0)."""

    projection: str = "unknown"
    center_lat: float = 0.0
    center_lon: float = 0.0
    map_scale_m: float = float("nan")
    line_offset: float = float("nan")
    sample_offset: float = float("nan")
    lines: int = 0
    samples: int = 0
    radius_m: float = MOON_RADIUS_M
    map_resolution_ppd: float = float("nan")
    lon_direction: str = "east"
    origin_check_px: float = float("nan")

    @property
    def pixel_area_m2(self) -> float:
        return float(self.map_scale_m) ** 2

    def pix_to_lonlat(self, line, sample):
        """0-based pixel indices (centre of the pixel) → (lon_east_deg, lat_deg)."""
        i = np.asarray(line, dtype=float)
        j = np.asarray(sample, dtype=float)
        if self.projection == "polar_stereographic":
            x = (j - self.sample_offset) * self.map_scale_m
            y = (self.line_offset - i) * self.map_scale_m
            rho = np.hypot(x, y)
            R = self.radius_m
            c = 2.0 * np.arctan(rho / (2.0 * R))
            if self.center_lat > 0:
                lat = 90.0 - np.degrees(c)
                lon = np.degrees(np.arctan2(x, -y))
            else:
                lat = -90.0 + np.degrees(c)
                lon = np.degrees(np.arctan2(x, y))
            lon = (lon + self.center_lon) % 360.0
            return lon, lat
        if self.projection == "equirectangular":
            ppd = self.map_resolution_ppd
            lon = (j - self.sample_offset) / ppd + self.center_lon
            lat = self.center_lat - (i - self.line_offset) / ppd
            return lon % 360.0, lat
        raise ValueError(f"unsupported projection {self.projection!r}")

    def lonlat_to_pix(self, lon, lat):
        """(lon_east_deg, lat_deg) → 0-based fractional (line, sample)."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        if self.projection == "polar_stereographic":
            R = self.radius_m
            lam = np.radians(lon - self.center_lon)
            phi = np.radians(lat)
            if self.center_lat > 0:
                rho = 2.0 * R * np.tan(np.pi / 4.0 - phi / 2.0)
                x, y = rho * np.sin(lam), -rho * np.cos(lam)
            else:
                rho = 2.0 * R * np.tan(np.pi / 4.0 + phi / 2.0)
                x, y = rho * np.sin(lam), rho * np.cos(lam)
            j = x / self.map_scale_m + self.sample_offset
            i = self.line_offset - y / self.map_scale_m
            return i, j
        if self.projection == "equirectangular":
            ppd = self.map_resolution_ppd
            dlon = (lon - self.center_lon + 180.0) % 360.0 - 180.0
            j = dlon * ppd + self.sample_offset
            i = self.line_offset - (lat - self.center_lat) * ppd
            return i, j
        raise ValueError(f"unsupported projection {self.projection!r}")

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                for k, v in self.__dict__.items()}


@dataclass
class RasterMeta:
    """What a label says about its image: shape, sample encoding, scaling,
    missing constant, where the bytes start, and the map geometry."""

    label_path: str = ""
    dialect: str = ""                 # pds3 | pds4
    data_file: str = ""               # file holding the pixels (may equal the label)
    lines: int = 0
    samples: int = 0
    bands: int = 1
    dtype: str = "<f4"                # numpy dtype string
    scaling_factor: float = 1.0
    offset: float = 0.0
    missing: list = field(default_factory=list)
    valid_min: float | None = None
    valid_max: float | None = None
    byte_offset: int = 0
    unit: str | None = None
    product_id: str = ""
    description: str = ""
    georef: Georef = field(default_factory=Georef)
    keywords: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("georef", "keywords")}
        d["georef"] = self.georef.as_dict()
        d["keywords"] = {k: v for k, v in list(self.keywords.items())[:80]}
        return d


_PDS3_DTYPES = {
    ("PC_REAL", 32): "<f4", ("PC_REAL", 64): "<f8",
    ("IEEE_REAL", 32): ">f4", ("IEEE_REAL", 64): ">f8",
    ("REAL", 32): ">f4", ("REAL", 64): ">f8",
    ("MSB_INTEGER", 16): ">i2", ("MSB_INTEGER", 32): ">i4", ("MSB_INTEGER", 8): "i1",
    ("LSB_INTEGER", 16): "<i2", ("LSB_INTEGER", 32): "<i4", ("LSB_INTEGER", 8): "i1",
    ("PC_INTEGER", 16): "<i2", ("PC_INTEGER", 32): "<i4", ("PC_INTEGER", 8): "i1",
    ("MSB_UNSIGNED_INTEGER", 16): ">u2", ("MSB_UNSIGNED_INTEGER", 32): ">u4",
    ("MSB_UNSIGNED_INTEGER", 8): "u1",
    ("LSB_UNSIGNED_INTEGER", 16): "<u2", ("LSB_UNSIGNED_INTEGER", 32): "<u4",
    ("LSB_UNSIGNED_INTEGER", 8): "u1",
    ("PC_UNSIGNED_INTEGER", 16): "<u2", ("PC_UNSIGNED_INTEGER", 32): "<u4",
    ("PC_UNSIGNED_INTEGER", 8): "u1",
    ("UNSIGNED_INTEGER", 8): "u1", ("UNSIGNED_INTEGER", 16): ">u2",
    ("INTEGER", 16): ">i2", ("INTEGER", 32): ">i4", ("INTEGER", 8): "i1",
}

_PDS4_DTYPES = {
    "IEEE754LSBSingle": "<f4", "IEEE754LSBDouble": "<f8",
    "IEEE754MSBSingle": ">f4", "IEEE754MSBDouble": ">f8",
    "SignedLSB2": "<i2", "SignedLSB4": "<i4", "SignedMSB2": ">i2", "SignedMSB4": ">i4",
    "UnsignedLSB2": "<u2", "UnsignedLSB4": "<u4", "UnsignedMSB2": ">u2", "UnsignedMSB4": ">u4",
    "SignedByte": "i1", "UnsignedByte": "u1",
}


def _scale_to_m(value: float, unit: str | None) -> float:
    u = (unit or "").upper().replace(" ", "")
    if "KM" in u:
        return value * 1000.0
    if u in ("M/PIXEL", "M/PIX", "METERS/PIXEL", "M", "METER/PIXEL", "METERS/PIX"):
        return value
    # PDS3 Diviner polar labels give MAP_SCALE in km/pixel; a bare number with
    # no unit below 10 is read as km, otherwise as metres.
    return value * 1000.0 if value < 10.0 else value


def georef_from_pds3(lbl: dict, lines: int, samples: int) -> Georef:
    proj = str(_s(lbl, "IMAGE_MAP_PROJECTION.MAP_PROJECTION_TYPE", "MAP_PROJECTION_TYPE",
                   default="")).upper().replace("_", " ")
    g = Georef(lines=lines, samples=samples)
    g.center_lat = _f(lbl, "IMAGE_MAP_PROJECTION.CENTER_LATITUDE", "CENTER_LATITUDE", default=0.0)
    g.center_lon = _f(lbl, "IMAGE_MAP_PROJECTION.CENTER_LONGITUDE", "CENTER_LONGITUDE", default=0.0)
    g.radius_m = _scale_to_m(_f(lbl, "IMAGE_MAP_PROJECTION.A_AXIS_RADIUS", "A_AXIS_RADIUS",
                                default=MOON_RADIUS_M / 1000.0),
                             _s(lbl, "IMAGE_MAP_PROJECTION.A_AXIS_RADIUS__unit", "A_AXIS_RADIUS__unit"))
    ms = _f(lbl, "IMAGE_MAP_PROJECTION.MAP_SCALE", "MAP_SCALE")
    if ms is not None:
        g.map_scale_m = _scale_to_m(ms, _s(lbl, "IMAGE_MAP_PROJECTION.MAP_SCALE__unit", "MAP_SCALE__unit"))
    g.map_resolution_ppd = _f(lbl, "IMAGE_MAP_PROJECTION.MAP_RESOLUTION", "MAP_RESOLUTION",
                              default=float("nan"))
    lo = _f(lbl, "IMAGE_MAP_PROJECTION.LINE_PROJECTION_OFFSET", "LINE_PROJECTION_OFFSET")
    so = _f(lbl, "IMAGE_MAP_PROJECTION.SAMPLE_PROJECTION_OFFSET", "SAMPLE_PROJECTION_OFFSET")
    # PDS3: the offsets are the origin's position measured from the CENTRE of
    # pixel (1,1), in pixels.  With 0-based pixel centres at integers the
    # origin therefore sits at 0-based coordinate == offset.  Check: the LOLA
    # 16 ppd global product (5760 samples, centre longitude 180) carries
    # SAMPLE_PROJECTION_OFFSET = 2879.5, the boundary between the two central
    # pixels, which is where lon 180 must fall.  ``origin_check_px`` records
    # how far the parsed origin sits from the raster centre.
    if lo is not None:
        g.line_offset = lo
    if so is not None:
        g.sample_offset = so
    g.lon_direction = str(_s(lbl, "IMAGE_MAP_PROJECTION.POSITIVE_LONGITUDE_DIRECTION",
                             "POSITIVE_LONGITUDE_DIRECTION", default="EAST")).lower()
    if "STEREOGRAPHIC" in proj:
        g.projection = "polar_stereographic"
        if not math.isfinite(g.map_scale_m) and math.isfinite(g.map_resolution_ppd):
            g.map_scale_m = 2 * math.pi * g.radius_m / 360.0 / g.map_resolution_ppd
        if lines and samples and math.isfinite(g.line_offset):
            g.origin_check_px = float(math.hypot(g.line_offset - (lines - 1) / 2.0,
                                                 g.sample_offset - (samples - 1) / 2.0))
    elif "CYLINDRICAL" in proj or "EQUIRECTANGULAR" in proj or "SIMPLE" in proj:
        g.projection = "equirectangular"
        if not math.isfinite(g.map_resolution_ppd) and math.isfinite(g.map_scale_m):
            g.map_resolution_ppd = 2 * math.pi * g.radius_m / 360.0 / g.map_scale_m
        if not math.isfinite(g.map_scale_m) and math.isfinite(g.map_resolution_ppd):
            g.map_scale_m = 2 * math.pi * g.radius_m / 360.0 / g.map_resolution_ppd
    else:
        g.projection = proj.lower().replace(" ", "_") or "unknown"
    return g


def read_pds3_label(path: Path) -> RasterMeta:
    """Parse a PDS3 label (detached ``.lbl`` or the head of an attached ``.img``)."""
    path = Path(path)
    raw = path.read_bytes()
    head = raw[: min(len(raw), 65536)].decode("latin-1", errors="replace")
    lbl = parse_pds3_label(head)
    meta = RasterMeta(label_path=str(path), dialect="pds3")
    meta.lines = int(_f(lbl, "IMAGE.LINES", "LINES", default=0) or 0)
    meta.samples = int(_f(lbl, "IMAGE.LINE_SAMPLES", "LINE_SAMPLES", default=0) or 0)
    meta.bands = int(_f(lbl, "IMAGE.BANDS", "BANDS", default=1) or 1)
    st = str(_s(lbl, "IMAGE.SAMPLE_TYPE", "SAMPLE_TYPE", default="PC_REAL")).upper()
    bits = int(_f(lbl, "IMAGE.SAMPLE_BITS", "SAMPLE_BITS", default=32) or 32)
    meta.dtype = _PDS3_DTYPES.get((st, bits), "<f4")
    meta.scaling_factor = _f(lbl, "IMAGE.SCALING_FACTOR", "SCALING_FACTOR", default=1.0)
    meta.offset = _f(lbl, "IMAGE.OFFSET", "OFFSET", default=0.0)
    for k in ("IMAGE.MISSING_CONSTANT", "MISSING_CONSTANT", "IMAGE.CORE_NULL", "CORE_NULL",
              "IMAGE.INVALID_CONSTANT", "INVALID_CONSTANT", "IMAGE.NULL", "NULL"):
        v = _f(lbl, k)
        if v is not None and v not in meta.missing:
            meta.missing.append(v)
    meta.valid_min = _f(lbl, "IMAGE.VALID_MINIMUM", "VALID_MINIMUM")
    meta.valid_max = _f(lbl, "IMAGE.VALID_MAXIMUM", "VALID_MAXIMUM")
    meta.unit = _s(lbl, "IMAGE.UNIT", "UNIT", "IMAGE.SCALING_FACTOR__unit")
    meta.product_id = str(_s(lbl, "PRODUCT_ID", default=""))
    meta.description = str(_s(lbl, "IMAGE.DESCRIPTION", "DESCRIPTION", default=""))
    # pointer: ^IMAGE = ("file", n) | n | n <BYTES>
    ptr = lbl.get("^IMAGE")
    record_bytes = int(_f(lbl, "RECORD_BYTES", default=0) or 0)
    data_file, start = path.name, 0
    if isinstance(ptr, list):
        data_file = ptr[0]
        if len(ptr) > 1:
            n = float(ptr[1])
            unit = lbl.get("^IMAGE__unit", "")
            start = int(n) if "BYTE" in str(unit).upper() else int((n - 1) * record_bytes)
    elif ptr is not None:
        pv = str(ptr).strip().strip('"')
        if re.match(r"^[0-9.]+$", pv):
            n = float(pv)
            unit = lbl.get("^IMAGE__unit", "")
            start = int(n) if "BYTE" in str(unit).upper() else int((n - 1) * record_bytes)
        else:
            data_file = pv
    if data_file.lower() != path.name.lower():
        # detached label: the data file sits beside the label, name case may differ
        cand = path.with_name(data_file)
        if not cand.exists():
            for p in path.parent.glob("*"):
                if p.name.lower() == data_file.lower():
                    cand = p
                    break
        meta.data_file = str(cand)
    else:
        meta.data_file = str(path)
    meta.byte_offset = start
    meta.georef = georef_from_pds3(lbl, meta.lines, meta.samples)
    meta.keywords = {k: v for k, v in lbl.items() if not k.endswith("__unit")
                     and isinstance(v, str) and len(v) < 200}
    return meta


# ---------------------------------------------------------------------------
# PDS4
# ---------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _find_text(el, name: str, default=None):
    for e in el.iter():
        if _local(e.tag) == name and e.text is not None and e.text.strip():
            return e.text.strip()
    return default


def _find_float(el, name: str, default=None):
    t = _find_text(el, name)
    if t is None:
        return default
    try:
        return float(t)
    except ValueError:
        return default


def _find_unit(el, name: str):
    for e in el.iter():
        if _local(e.tag) == name:
            return e.attrib.get("unit")
    return None


def read_pds4_label(path: Path) -> RasterMeta:
    """Parse a PDS4 XML label for a 2-D (or 3-D) array image."""
    path = Path(path)
    root = ET.fromstring(path.read_bytes())
    meta = RasterMeta(label_path=str(path), dialect="pds4")
    meta.product_id = _find_text(root, "logical_identifier", "") or ""
    meta.description = (_find_text(root, "description", "") or "")[:500]
    fname = _find_text(root, "file_name")
    arr = None
    for e in root.iter():
        if _local(e.tag) in ("Array_2D_Image", "Array_2D_Map", "Array_2D", "Array_3D_Image",
                             "Array_3D_Spectrum", "Array_3D"):
            arr = e
            break
    if arr is not None:
        meta.byte_offset = int(_find_float(arr, "offset", 0) or 0)
        dt = _find_text(arr, "data_type", "IEEE754LSBSingle")
        meta.dtype = _PDS4_DTYPES.get(dt, "<f4")
        meta.scaling_factor = _find_float(arr, "scaling_factor", 1.0)
        meta.offset = _find_float(arr, "value_offset", 0.0)
        meta.unit = _find_text(arr, "unit")
        axes = []
        for ax in arr.iter():
            if _local(ax.tag) == "Axis_Array":
                axes.append((_find_text(ax, "axis_name", "").lower(),
                             int(_find_float(ax, "elements", 0) or 0)))
        for name, n in axes:
            if name.startswith("line") or name in ("y", "row", "rows"):
                meta.lines = n
            elif name.startswith("sample") or name in ("x", "column", "columns"):
                meta.samples = n
            elif name.startswith("band"):
                meta.bands = n
        if not meta.lines and len(axes) >= 2:
            meta.lines, meta.samples = axes[0][1], axes[1][1]
        for e in arr.iter():
            if _local(e.tag) in ("missing_constant", "invalid_constant", "saturated_constant",
                                 "high_instrument_saturation", "low_instrument_saturation"):
                try:
                    v = float(e.text)
                    if v not in meta.missing:
                        meta.missing.append(v)
                except (TypeError, ValueError):
                    pass
            if _local(e.tag) == "valid_minimum":
                meta.valid_min = _find_float(e, "valid_minimum")
            if _local(e.tag) == "valid_maximum":
                meta.valid_max = _find_float(e, "valid_maximum")
    data_file = fname or path.with_suffix(".img").name
    cand = path.with_name(data_file)
    if not cand.exists():
        for p in path.parent.glob("*"):
            if p.name.lower() == data_file.lower():
                cand = p
                break
    meta.data_file = str(cand)
    meta.georef = georef_from_pds4(root, meta.lines, meta.samples)
    kw = {}
    for e in root.iter():
        if e.text and e.text.strip() and len(e.text.strip()) < 120 and len(list(e)) == 0:
            kw.setdefault(_local(e.tag), e.text.strip())
    meta.keywords = kw
    return meta


def georef_from_pds4(root, lines: int, samples: int) -> Georef:
    g = Georef(lines=lines, samples=samples)
    proj_el = None
    for e in root.iter():
        if _local(e.tag) in ("Polar_Stereographic", "Equirectangular", "Simple_Cylindrical",
                             "Orthographic", "Lambert_Azimuthal_Equal_Area"):
            proj_el = e
            break
    if proj_el is None:
        return g
    name = _local(proj_el.tag)
    g.center_lat = _find_float(proj_el, "latitude_of_projection_origin",
                               _find_float(proj_el, "standard_parallel_1", 0.0)) or 0.0
    g.center_lon = _find_float(proj_el, "straight_vertical_longitude_from_pole",
                               _find_float(proj_el, "longitude_of_central_meridian", 0.0)) or 0.0
    rad = _find_float(root, "semi_major_radius", None)
    if rad is not None:
        g.radius_m = _scale_to_m(rad, _find_unit(root, "semi_major_radius"))
    px = _find_float(root, "pixel_resolution_x", None)
    if px is not None:
        g.map_scale_m = _scale_to_m(px, _find_unit(root, "pixel_resolution_x") or "m/pixel")
    ppd = _find_float(root, "pixel_scale_x", None)
    if ppd is not None:
        g.map_resolution_ppd = ppd
    ulx = _find_float(root, "upperleft_corner_x", None)
    uly = _find_float(root, "upperleft_corner_y", None)
    if "Stereographic" in name:
        g.projection = "polar_stereographic"
        if not math.isfinite(g.map_scale_m) and math.isfinite(g.map_resolution_ppd):
            g.map_scale_m = 2 * math.pi * g.radius_m / 360.0 / g.map_resolution_ppd
        if ulx is not None and uly is not None and math.isfinite(g.map_scale_m):
            ux = _scale_to_m(ulx, _find_unit(root, "upperleft_corner_x") or "m")
            uy = _scale_to_m(uly, _find_unit(root, "upperleft_corner_y") or "m")
            # upper-left corner of the upper-left pixel → origin (0,0) map coords
            g.sample_offset = -ux / g.map_scale_m - 0.5
            g.line_offset = uy / g.map_scale_m - 0.5
        else:
            lo = _find_float(root, "line_projection_offset", None)
            so = _find_float(root, "sample_projection_offset", None)
            if lo is not None and so is not None:
                g.line_offset, g.sample_offset = lo, so
        if lines and samples and math.isfinite(g.line_offset):
            g.origin_check_px = float(math.hypot(g.line_offset - (lines - 1) / 2.0,
                                                 g.sample_offset - (samples - 1) / 2.0))
    else:
        g.projection = "equirectangular"
        if not math.isfinite(g.map_resolution_ppd) and math.isfinite(g.map_scale_m):
            g.map_resolution_ppd = 2 * math.pi * g.radius_m / 360.0 / g.map_scale_m
        if not math.isfinite(g.map_scale_m) and math.isfinite(g.map_resolution_ppd):
            g.map_scale_m = 2 * math.pi * g.radius_m / 360.0 / g.map_resolution_ppd
        lo = _find_float(root, "line_projection_offset", None)
        so = _find_float(root, "sample_projection_offset", None)
        if lo is not None and so is not None:
            g.line_offset, g.sample_offset = lo - 0.5, so - 0.5
        elif ulx is not None and uly is not None and math.isfinite(g.map_resolution_ppd):
            g.sample_offset = -(ulx - g.center_lon) * g.map_resolution_ppd - 0.5 if abs(ulx) <= 360 else float("nan")
            g.line_offset = (uly - g.center_lat) * g.map_resolution_ppd - 0.5 if abs(uly) <= 90 else float("nan")
    return g


def read_label(path: Path) -> RasterMeta:
    """Dispatch on the label dialect (``.xml`` → PDS4; anything else → PDS3)."""
    path = Path(path)
    if path.suffix.lower() == ".xml":
        return read_pds4_label(path)
    return read_pds3_label(path)


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------
def read_raster(meta: RasterMeta, *, band: int = 0) -> np.ndarray:
    """The scaled float32 image with missing/invalid samples as NaN."""
    p = Path(meta.data_file)
    n = meta.lines * meta.samples * max(1, meta.bands)
    dt = np.dtype(meta.dtype)
    with p.open("rb") as fh:
        fh.seek(meta.byte_offset)
        buf = fh.read(n * dt.itemsize)
    if len(buf) < n * dt.itemsize:
        raise ValueError(f"{p}: expected {n * dt.itemsize} bytes of pixels at offset "
                         f"{meta.byte_offset}, found {len(buf)}")
    raw = np.frombuffer(buf, dtype=dt)
    if meta.bands > 1:
        raw = raw.reshape(meta.bands, meta.lines, meta.samples)[band]
    else:
        raw = raw.reshape(meta.lines, meta.samples)
    bad = np.zeros(raw.shape, dtype=bool)
    for m in meta.missing:
        bad |= raw == np.asarray(m).astype(dt) if dt.kind in "iu" else np.isclose(raw, m)
    img = raw.astype(np.float32) * np.float32(meta.scaling_factor) + np.float32(meta.offset)
    if meta.valid_min is not None:
        bad |= raw.astype(np.float64) < meta.valid_min
    if meta.valid_max is not None:
        bad |= raw.astype(np.float64) > meta.valid_max
    bad |= ~np.isfinite(img)
    img[bad] = np.nan
    return img


def write_pds3_raster(img: np.ndarray, path: Path, *, georef: Georef, product_id: str = "SYNTH",
                      description: str = "", missing: float = -32768.0,
                      scaling: float = 0.01, offset: float = 0.0, unit: str = "K") -> Path:
    """Write ``img`` as a PDS3 detached-label raster (``.lbl`` + ``.img``),
    16-bit MSB integers with scaling — the Diviner GDR encoding — so the
    reader is tested against the format it will meet.  Returns the label path."""
    path = Path(path)
    lbl_path = path.with_suffix(".lbl")
    img_path = path.with_suffix(".img")
    q = np.where(np.isfinite(img), np.round((img - offset) / scaling), missing)
    q = np.clip(q, -32768, 32767).astype(">i2")
    img_path.write_bytes(q.tobytes())
    lines, samples = img.shape
    proj = "POLAR STEREOGRAPHIC" if georef.projection == "polar_stereographic" else "SIMPLE CYLINDRICAL"
    txt = "\n".join([
        "PDS_VERSION_ID = PDS3",
        "RECORD_TYPE = FIXED_LENGTH",
        f"RECORD_BYTES = {samples * 2}",
        f"FILE_RECORDS = {lines}",
        f'^IMAGE = "{img_path.name}"',
        f'PRODUCT_ID = "{product_id}"',
        f'DESCRIPTION = "{description}"',
        "OBJECT = IMAGE",
        f"  LINES = {lines}",
        f"  LINE_SAMPLES = {samples}",
        "  SAMPLE_TYPE = MSB_INTEGER",
        "  SAMPLE_BITS = 16",
        f"  SCALING_FACTOR = {scaling}",
        f"  OFFSET = {offset}",
        f"  MISSING_CONSTANT = {int(missing)}",
        f"  UNIT = {unit}",
        "END_OBJECT = IMAGE",
        "OBJECT = IMAGE_MAP_PROJECTION",
        f'  MAP_PROJECTION_TYPE = "{proj}"',
        f"  A_AXIS_RADIUS = {georef.radius_m / 1000.0:.3f} <KM>",
        f"  CENTER_LATITUDE = {georef.center_lat} <DEG>",
        f"  CENTER_LONGITUDE = {georef.center_lon} <DEG>",
        f"  MAP_SCALE = {georef.map_scale_m / 1000.0:.6f} <KM/PIXEL>",
        f"  MAP_RESOLUTION = {(2 * math.pi * georef.radius_m / 360.0 / georef.map_scale_m):.4f} <PIXEL/DEGREE>",
        f"  LINE_PROJECTION_OFFSET = {georef.line_offset:.3f}",
        f"  SAMPLE_PROJECTION_OFFSET = {georef.sample_offset:.3f}",
        "  POSITIVE_LONGITUDE_DIRECTION = EAST",
        "END_OBJECT = IMAGE_MAP_PROJECTION",
        "END",
        "",
    ])
    lbl_path.write_text(txt)
    return lbl_path


def polar_georef(pole: str, n_px: int, scale_m: float = 240.0) -> Georef:
    """A square polar-stereographic grid centred on the pole (origin at the
    grid centre), the layout of the Diviner and LOLA 80–90° products."""
    g = Georef(projection="polar_stereographic", center_lat=90.0 if pole == "north" else -90.0,
               center_lon=0.0, map_scale_m=scale_m, lines=n_px, samples=n_px)
    g.line_offset = (n_px - 1) / 2.0
    g.sample_offset = (n_px - 1) / 2.0
    g.origin_check_px = 0.0
    return g


__all__ = ["Georef", "MOON_RADIUS_M", "RasterMeta", "georef_from_pds3", "georef_from_pds4",
           "parse_pds3_label", "polar_georef", "read_label", "read_pds3_label",
           "read_pds4_label", "read_raster", "write_pds3_raster"]
