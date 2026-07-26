"""EMBER -- a search for mid-infrared waste heat that switched off.

Signature S1 of ``docs/necrosignatures.md``: an infrared excess present at an
early epoch (IRAS 1983, AKARI 2006-07) and absent at a later one (WISE 2010).

The whole waste-heat literature is single-epoch by construction, and says so:
its roadmap paper argues that "the signature is as long-lived as the underlying
technology". EMBER inverts that assumption and asks the time-domain question
nobody has asked of these catalogues in the fading direction.

See ``docs/ember.md`` for the novelty status, the per-epoch-pair systematics
verdict, and the honest limitations.
"""
from .bands import BANDS, CANDIDATE_PAIRS, EPOCH_LADDER, audit_pair
from .crossepoch import (
    CessationResult,
    ExcessMeasurement,
    PhotosphereLocus,
    adjudicate_ladder,
    beam_sum_consistency,
    calibrate_null,
    cessation,
    cessation_mc,
    fit_dust_temperature,
    fit_photosphere_locus,
    measure_excess,
)
from .vet import RULES, VetResult, vet_all, vet_source

__all__ = [
    "BANDS",
    "CANDIDATE_PAIRS",
    "EPOCH_LADDER",
    "RULES",
    "CessationResult",
    "ExcessMeasurement",
    "PhotosphereLocus",
    "VetResult",
    "adjudicate_ladder",
    "audit_pair",
    "beam_sum_consistency",
    "calibrate_null",
    "cessation",
    "cessation_mc",
    "fit_dust_temperature",
    "fit_photosphere_locus",
    "measure_excess",
    "vet_all",
    "vet_source",
]
