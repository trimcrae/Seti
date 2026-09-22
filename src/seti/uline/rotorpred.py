"""Turn the embedded rotational constants into searchable line lists.

``src/seti/data_assets/rotor_constants.yaml`` holds, for every species this
channel wants and no catalogue carries, the published A, B, C (and whatever
quartic constants and dipole components could be established), one block per
isotopologue, each with its provenance and its uncertainty class.  This module
reads that file, builds :class:`seti.uline.rotor.RotorConstants`, predicts each
isotopologue's spectrum with the Watson A-reduced Hamiltonian, weights the
isotopologues by their terrestrial-chlorine abundances and returns ONE line
table per species in the same ``.cat`` schema the JPL and CDMS entries use --- so
the pattern search treats a predicted species exactly like a catalogued one,
except for the flags it carries.

The flags are the point.  Every block is ``verify: true``; a predicted line's
``err_mhz`` is the model's own honest statement of how far it is trusted, and
:func:`predict_species_lines` reports, per isotopologue, whether the quartic
constants are known.  **They usually are not, and that is the binding limit,
not the missing catalogue entry**: for a heavy rotor an unmodelled ΔJ displaces
an R-branch line at J ≈ 30 by of order 10³ MHz, two orders of magnitude beyond
any survey's matching tolerance.  ``searchable`` in the returned metadata says
so per species — the median predicted-line error against the tolerance the
surveys actually allow — and ``docs/uline.md`` §6 carries the consequence.

Three kinds of block are understood:

``asymmetric`` (the default)
    ``A, B, C`` with a representation and any known distortion constants,
    exactly as :class:`RotorConstants` takes them.
``symmetric``
    ``B`` and ``axial`` for a C₃ᵥ top, with ``k3_weight`` the A : E nuclear-spin
    ratio.  It is turned into the asymmetric form (A = B = B₀, C = axial, IIIr
    for an oblate top; A = axial, B = C = B₀, Ir for a prolate one) and run
    through the same Hamiltonian, which is exact in that limit and is checked
    against the closed-form symmetric-top energies in the offline suite.
``scale_from``
    an isotopologue whose constants are the parent's, moved by the ratio of the
    moments of inertia of a common r₀ geometry
    (:func:`seti.uline.rotor.isotopologue_constants`).  The shift is structural,
    so the error is the geometry's error times the shift; the block's
    ``abc_uncertainty_mhz`` states it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .lines import CATDIR_TEMPS
from .rotor import (
    DISTORTION_OMEGA_CM,
    QUARTIC,
    SEXTIC,
    RotorConstants,
    isotopologue_constants,
    predict_lines,
)

#: Uncertainty classes, worst first.  ``lab`` is a laboratory value quoted from
#: a paper; ``recalled`` and ``snippet`` are the author's reconstruction of the
#: literature and a search-engine excerpt of it; ``structure`` is derived by
#: scaling a measured isotopologue with an approximate geometry; ``estimate`` is
#: order-of-magnitude and is never used as a frequency.
QUALITY_ORDER = ("estimate", "structure", "recalled", "snippet", "lab")


def assets_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data_assets" / "rotor_constants.yaml"


def load_rotor_assets(path: Path | str | None = None) -> dict:
    """The parsed constants file; ``{}`` (never an exception) when unreadable."""
    try:
        import yaml
        p = Path(path) if path is not None else assets_path()
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text()) or {}
    except Exception as exc:                                   # noqa: BLE001
        print(f"[uline] rotor constants not loaded ({exc!r})")
        return {}


@dataclass
class Isotopologue:
    """One isotopologue's constants with everything needed to judge them."""

    species: str
    name: str
    constants: RotorConstants
    abundance: float = 1.0
    quality: str = "recalled"
    abc_uncertainty_mhz: float = 1.0
    source: str = ""
    hyperfine: dict | None = None
    scaled_from: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {"species": self.species, "name": self.name, "abundance": self.abundance,
             "quality": self.quality, "abc_uncertainty_mhz": self.abc_uncertainty_mhz,
             "source": self.source, "hyperfine": self.hyperfine,
             "scaled_from": self.scaled_from, "notes": list(self.notes),
             "quartic_known": bool(self.constants.quartic_known),
             "any_quartic_known": bool(self.constants.any_quartic_known),
             "sextic_known": bool(self.constants.sextic_known)}
        d["constants"] = self.constants.as_dict()
        return d


def _symmetric_to_asymmetric(blk: dict, name: str) -> dict:
    """A ``kind: symmetric`` block as A ≥ B ≥ C plus the right representation.

    Oblate (axial < B₀): A = B = B₀, C = axial, z = c (IIIr), dipole along c.
    Prolate (axial > B₀): A = axial, B = C = B₀, z = a (Ir), dipole along a.
    The K = 3n nuclear-spin weight becomes the ``k_mod3`` rule on that axis.
    """
    b0 = float(blk["B"])
    axial = blk.get("axial")
    mu = float(blk.get("mu") or blk.get("mu_debye") or 0.0)
    k3 = float(blk.get("k3_weight") or 1.0)
    out = {"DJ": blk.get("DJ"), "DJK": blk.get("DJK"), "DK": blk.get("DK"),
           "dJ": 0.0, "dK": 0.0, "name": name}
    if axial is None:
        # No axial constant: the K ladder is unknown.  Take a prolate top with
        # A only marginally above B, which keeps A ≥ B ≥ C legal and leaves the
        # K-dependence of the ENERGY (and so of the Boltzmann factor) flat —
        # the state of knowledge, stated rather than invented.
        out.update({"A": b0 * 1.000001, "B": b0, "C": b0, "representation": "Ir", "mu_a": mu})
        out["spin_weights"] = {"rule": "k_mod3", "axis": "a", "multiple": k3, "other": 1.0}
        out["_axial_known"] = False
        return out
    axial = float(axial)
    if axial < b0:                                             # oblate
        out.update({"A": b0, "B": b0, "C": axial, "representation": "IIIr", "mu_c": mu})
        out["spin_weights"] = {"rule": "k_mod3", "axis": "c", "multiple": k3, "other": 1.0}
    else:                                                      # prolate
        out.update({"A": axial, "B": b0, "C": b0, "representation": "Ir", "mu_a": mu})
        out["spin_weights"] = {"rule": "k_mod3", "axis": "a", "multiple": k3, "other": 1.0}
    out["_axial_known"] = True
    return out


def _constants_from_block(species: str, blk: dict) -> dict:
    """Normalise one isotopologue block to :class:`RotorConstants` keyword form."""
    consts = dict(blk.get("constants") or {})
    name = str(blk.get("name") or species)
    if str(blk.get("kind", "")).lower() == "symmetric" or (
            "B" in consts and "A" not in consts and "C" not in consts):
        kw = _symmetric_to_asymmetric(consts, name)
    else:
        kw = {k: consts.get(k) for k in ("A", "B", "C", *QUARTIC, *SEXTIC,
                                         "mu_a", "mu_b", "mu_c")}
        kw["representation"] = consts.get("representation", "Ir")
        kw["spin_weights"] = dict(blk.get("spin_weights") or {"rule": "none"})
        kw["name"] = name
        kw["_axial_known"] = True
    kw.setdefault("spin_weights", dict(blk.get("spin_weights") or {"rule": "none"}))
    for k in ("mu_a", "mu_b", "mu_c"):
        kw[k] = float(kw.get(k) or 0.0)
    kw["source"] = str(blk.get("source") or "")
    kw["verify"] = True
    return kw


def species_isotopologues(assets: dict, species: str) -> list[Isotopologue]:
    """Every isotopologue of ``species`` in the assets, constants built and scaled.

    A block with ``scale_from`` is rebuilt from the named parent's constants and
    the two geometries under ``geometry.parent`` / ``geometry.substituted``; when
    either geometry is missing the block's own constants are kept and a note
    says the isotopic shift was NOT applied, which is a far larger error than
    the block advertises and must not pass silently.
    """
    spec = ((assets.get("species") or {}).get(species) or {})
    blocks = list(spec.get("isotopologues") or [])
    geometry = spec.get("geometry") or {}
    built: dict[str, Isotopologue] = {}
    out: list[Isotopologue] = []
    for blk in blocks:
        name = str(blk.get("name") or species)
        kw = _constants_from_block(species, blk)
        axial_known = bool(kw.pop("_axial_known", True))
        notes: list[str] = []
        if not axial_known:
            notes.append("axial constant unknown: the K ladder is Boltzmann-flat")
        try:
            c = RotorConstants(**kw)
        except (TypeError, ValueError) as exc:
            out.append(Isotopologue(species=species, name=name,
                                    constants=RotorConstants(A=1.0, B=1.0, C=1.0, mu_a=1.0),
                                    abundance=0.0, quality="estimate",
                                    source=str(blk.get("source") or ""),
                                    notes=[f"UNUSABLE constants: {exc}"]))
            continue
        parent_name = blk.get("scale_from")
        if parent_name:
            par = built.get(str(parent_name))
            geo_p = geometry.get("parent")
            geo_s = (geometry.get("substituted") or {}).get(name)
            if par is not None and geo_p and geo_s:
                c = isotopologue_constants(
                    par.constants, [(float(m), tuple(map(float, r))) for m, r in geo_p],
                    [(float(m), tuple(map(float, r))) for m, r in geo_s],
                    name=name, source=str(blk.get("source") or ""),
                    spin_weights=dict(c.spin_weights), mu_a=c.mu_a, mu_b=c.mu_b, mu_c=c.mu_c,
                    representation=c.representation)
                notes.append(f"A,B,C scaled from {parent_name} by the r0 moments of inertia")
            else:
                notes.append(f"scale_from={parent_name!r} NOT applied "
                             f"(parent or geometry missing): constants are the parent's, "
                             f"so the isotopic shift (~1-2 % of B) is an UNMODELLED error")
        iso = Isotopologue(species=species, name=name, constants=c,
                           abundance=float(blk.get("abundance", 1.0)),
                           quality=str(blk.get("quality", "recalled")),
                           abc_uncertainty_mhz=float(blk.get("abc_uncertainty_mhz", 1.0)),
                           source=str(blk.get("source") or ""),
                           hyperfine=spec.get("hyperfine"),
                           scaled_from=str(parent_name) if parent_name else None,
                           notes=notes)
        built[name] = iso
        out.append(iso)
    return out


def predictable_species(assets: dict, role: str | None = None) -> list[str]:
    """Species in the assets, optionally restricted to ``role`` (target/validation)."""
    return [sp for sp, blk in ((assets.get("species") or {}).items())
            if role is None or str(blk.get("role", "")) == role]


def predict_species_lines(assets: dict, species: str, *, fmin_mhz: float = 0.0,
                          fmax_mhz: float = 2.0e6, temps=CATDIR_TEMPS,
                          lgint_floor: float = -9.0, err_base_mhz: float = 0.5,
                          err_rel: float = 1e-5, j_max: int | None = None,
                          distortion_omega_cm: float = DISTORTION_OMEGA_CM,
                          ) -> tuple[pd.DataFrame, dict[str, dict], dict]:
    """``(lines, entries, meta)`` for one species, every isotopologue stacked.

    ``lines`` is in the ``.cat`` schema with an extra ``isotopologue`` column;
    ``entries`` maps ``predicted:<species>:<isotopologue>`` to that
    isotopologue's partition function so :func:`seti.uline.lines.rescale_lgint`
    can move it to any T_ex; ``meta`` carries the provenance, the quality class,
    the quartic-known flag and the predicted-error distribution — the numbers a
    reader needs to decide how much a coincidence is worth.
    """
    isos = species_isotopologues(assets, species)
    frames, entries = [], {}
    per_iso = []
    for i, iso in enumerate(isos):
        rec = iso.as_dict()
        c = iso.constants
        if iso.abundance <= 0 or (c.mu_a == c.mu_b == c.mu_c == 0.0):
            rec.update({"n_lines": 0, "skipped": "no dipole component or zero abundance"})
            per_iso.append(rec)
            continue
        eid = f"predicted:{species}:{iso.name}"
        try:
            df, ent = predict_lines(
                c, fmin_mhz=fmin_mhz, fmax_mhz=fmax_mhz, temps=temps, j_max=j_max,
                lgint_floor=lgint_floor, tag=900000 + i, err_base_mhz=err_base_mhz,
                err_rel=err_rel, err_per_j_mhz=2.0 * float(iso.abc_uncertainty_mhz),
                hyperfine=iso.hyperfine, abundance=float(iso.abundance),
                distortion_omega_cm=float(distortion_omega_cm))
        except Exception as exc:                               # noqa: BLE001
            rec.update({"n_lines": 0, "skipped": f"prediction failed: {exc!r}"})
            per_iso.append(rec)
            continue
        if len(df):
            df["entry_id"] = eid
            df["isotopologue"] = iso.name
            frames.append(df)
            # An entry with no line in band would leave a partition function
            # nothing ever rescales; the inventory below still names it.
            entries[eid] = {**ent.as_dict(), "species": species, "isotopologue": iso.name}
        rec.update({"n_lines": int(len(df)), "j_max_used": int(df["j_up"].max()) if len(df) else 0,
                    "err_mhz_median": float(df["err_mhz"].median()) if len(df) else None,
                    "err_mhz_p95": float(df["err_mhz"].quantile(0.95)) if len(df) else None,
                    "freq_span_mhz": [float(df["freq_mhz"].min()), float(df["freq_mhz"].max())]
                    if len(df) else None})
        per_iso.append(rec)
    frames = [f for f in frames if len(f)]
    lines = (pd.concat(frames, ignore_index=True).sort_values("freq_mhz").reset_index(drop=True)
             if frames else pd.DataFrame())
    meta = {
        "species": species,
        "role": str(((assets.get("species") or {}).get(species) or {}).get("role", "")),
        "literature": list(((assets.get("species") or {}).get(species) or {}).get("literature")
                           or []),
        # Some species are MODEL-limited, not only constant-limited: SO2F2 is
        # accidentally near-spherical and Watson's A-reduction fails for it, so
        # the published quartic set would not by itself make this A-reduced
        # predictor right.  Carried through to summary.json so it cannot be
        # mistaken later for an unexplained residual.
        "hamiltonian_caveat": ((assets.get("species") or {}).get(species)
                               or {}).get("hamiltonian_caveat"),
        "n_isotopologues": len(isos),
        "isotopologues": per_iso,
        "n_lines": int(len(lines)),
        "verify": True,
        "quality": min((i.quality for i in isos), key=lambda q: QUALITY_ORDER.index(q)
                       if q in QUALITY_ORDER else 0) if isos else "unknown",
        "quartic_known": bool(isos) and all(i.constants.quartic_known for i in isos),
        "any_quartic_known": bool(isos) and any(i.constants.any_quartic_known for i in isos),
        "distortion_omega_cm": float(distortion_omega_cm),
    }
    if len(lines):
        meta["err_mhz"] = {"median": float(lines["err_mhz"].median()),
                           "p05": float(lines["err_mhz"].quantile(0.05)),
                           "p95": float(lines["err_mhz"].quantile(0.95)),
                           "max": float(lines["err_mhz"].max())}
    return lines, entries, meta


def searchability(meta: dict, *, tolerance_mhz: float) -> dict:
    """Is this species' prediction sharp enough to be matched at ``tolerance_mhz``?

    The pattern search pairs a predicted feature with a U-line inside
    ``max(linewidth, σ_pred, σ_U)``.  When σ_pred is far larger than the
    linewidth the "tolerance" is the prediction's own ignorance, the number of
    chance alignments rises with it, and a coincidence carries no information:
    the rigid-shift false-alarm probability then cannot reach the gate.  This
    turns that into a reported number instead of a surprise --- a species is
    ``frequency_limited`` when the median predicted error exceeds the survey's
    linewidth, and the fix is named (obtain the quartic constants), because
    nothing about the data can repair it.
    """
    err = (meta or {}).get("err_mhz") or {}
    med = err.get("median")
    if med is None:
        return {"status": "NO_LINES", "tolerance_mhz": float(tolerance_mhz)}
    ratio = float(med) / float(tolerance_mhz) if tolerance_mhz > 0 else float("inf")
    if ratio <= 1.0:
        status = "SEARCHABLE"
    elif ratio <= 10.0:
        status = "DEGRADED"
    else:
        status = "FREQUENCY_LIMITED"
    out = {"status": status, "median_err_mhz": float(med),
           "tolerance_mhz": float(tolerance_mhz), "err_over_tolerance": ratio,
           "quartic_known": bool(meta.get("quartic_known"))}
    if status != "SEARCHABLE" and not meta.get("quartic_known"):
        out["remedy"] = ("the quartic centrifugal-distortion constants are unknown for this "
                         "species; obtaining them (not more data) is what makes it searchable")
    return out


def rotor_error_model(conf: dict | None) -> dict:
    """``base_mhz`` / ``rel`` for predicted lines, from ``config/uline.yaml``."""
    em = ((conf or {}).get("predicted") or {}).get("error_model") or {}
    return {"base_mhz": float(em.get("base_mhz", 0.5)),
            "rel": float(em.get("rel_with_distortion", 1e-5)),
            "distortion_omega_cm": float(em.get("distortion_omega_cm", DISTORTION_OMEGA_CM))}


def summarise_assets(assets: dict) -> dict:
    """One row per species: role, isotopologues, quality, what is known."""
    out = {}
    for sp in (assets.get("species") or {}):
        isos = species_isotopologues(assets, sp)
        out[sp] = {
            "role": str(((assets.get("species") or {}).get(sp) or {}).get("role", "")),
            "n_isotopologues": len(isos),
            "isotopologues": [{"name": i.name, "abundance": i.abundance, "quality": i.quality,
                               "abc_uncertainty_mhz": i.abc_uncertainty_mhz,
                               "quartic_known": bool(i.constants.quartic_known),
                               "kappa": float(i.constants.kappa),
                               "dipole_debye": {"a": i.constants.mu_a, "b": i.constants.mu_b,
                                                "c": i.constants.mu_c},
                               "scaled_from": i.scaled_from, "notes": i.notes}
                              for i in isos],
            "hyperfine": ((assets.get("species") or {}).get(sp) or {}).get("hyperfine"),
            "literature": list(((assets.get("species") or {}).get(sp) or {}).get("literature")
                               or []),
        }
    return out


def validation_targets(assets: dict) -> list[dict]:
    """``[{species, database, tag}]`` — the catalogued molecules the predictor is
    checked against (``role: validation`` blocks' ``validate_against``)."""
    out = []
    for sp, blk in (assets.get("species") or {}).items():
        for va in (blk.get("validate_against") or []):
            out.append({"species": sp, "database": str(va.get("database", "jpl")),
                        "tag": int(va.get("tag"))})
    return out


def literature_sources(assets: dict) -> list[dict]:
    return [dict(x) for x in (assets.get("literature") or [])]


def merge_fetched_constants(assets: dict, fetched: dict) -> tuple[dict, list[dict]]:
    """Apply constants a ``litfetch`` run established, reporting every change.

    ``fetched`` is ``{species: {isotopologue: {constant: value, ...}}}``.  A
    value is written only where the assets carry ``None``, or where the fetched
    value differs from the embedded one, and **every** write is returned as a
    record with both numbers so nothing changes silently.  Nothing is ever
    deleted, and a species absent from the assets is ignored.
    """
    import copy
    out = copy.deepcopy(assets or {})
    changes: list[dict] = []
    for sp, per_iso in (fetched or {}).items():
        blk = (out.get("species") or {}).get(sp)
        if not blk:
            continue
        for iso in (blk.get("isotopologues") or []):
            new = (per_iso or {}).get(str(iso.get("name")))
            if not new:
                continue
            consts = iso.setdefault("constants", {})
            for k, v in new.items():
                if k not in ("A", "B", "C", *QUARTIC, *SEXTIC) or v is None:
                    continue
                old = consts.get(k)
                if old is not None and np.isclose(float(old), float(v), rtol=1e-9):
                    continue
                consts[k] = float(v)
                changes.append({"species": sp, "isotopologue": str(iso.get("name")),
                                "constant": k, "was": old, "now": float(v)})
    return out, changes


__all__ = ["QUALITY_ORDER", "Isotopologue", "assets_path", "literature_sources",
           "load_rotor_assets", "merge_fetched_constants", "predict_species_lines",
           "predictable_species", "rotor_error_model", "searchability", "species_isotopologues",
           "summarise_assets", "validation_targets"]
