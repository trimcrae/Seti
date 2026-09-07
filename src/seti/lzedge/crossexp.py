"""Cross-experiment accounting for an edge signal: windows, exposures, seasons.

For a model normalised to one expected event in LZ WS2024 (sigma_n = sigma_1),
what does each other high-energy nuclear-recoil search expect — counting only
the part of its window that reaches the recoil energies the model produces,
and only the part of its calendar on which the edge signal is 'on'?  A search
that ran December–May, or whose window stops below the recoil energies at the
edge, has no say however large its exposure.

The table is deliberately explicit: every entry names its source.  PLACEHOLDER
entries are to be replaced from results/lzlit/ text before anything is quoted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rate import Efficiency
from .timing import EventModel, LiveTime


@dataclass
class HighEnergySearch:
    name: str
    E_lo_keV: float
    E_hi_keV: float
    exposure_tonne_year: float
    start: str
    end: str
    efficiency: float = 1.0        # flat placeholder unless a table is given
    source: str = ""
    eff_table: list | None = None

    def efficiency_obj(self) -> Efficiency:
        if self.eff_table:
            return Efficiency.from_table(self.eff_table)
        return Efficiency.flat(self.E_lo_keV, self.E_hi_keV, self.efficiency)

    def livetime(self) -> LiveTime:
        return LiveTime.uniform(self.start, self.end)


# Every number here is a bibliographic assertion; the ``source`` says from where.
SEARCHES: list[HighEnergySearch] = [
    HighEnergySearch("XENON100 EFT (2017)", 6.6, 240.0, 34e-3 * 224.6 / 365.25,
                     "2011-02-28", "2012-03-31", 1.0,
                     source="arXiv:1705.02614 abstract: 6.6-240 keV_nr, 34 kg x 224.6 live days, 13 months in 2011-2012 (window edges and flat efficiency are PLACEHOLDER)"),
    HighEnergySearch("LZ WS2022 EFT (2024)", 5.0, 60.0, 0.9,
                     "2021-12-23", "2022-05-12", 1.0,
                     source="PLACEHOLDER window/efficiency; exposure ~0.9 t yr (60 live days x 5.5 t), Dec 2021-May 2022 from arXiv:2207.03764; EFT window from arXiv:2404.17666 to be read"),
]


def expected_counts(model_factory, sigma_1_cm2: float, searches=SEARCHES, step_days: float = 3.0) -> list[dict]:
    """For each search: expected events at sigma_1, with and without the seasonal weighting.

    ``model_factory(efficiency) -> EventModel`` builds the model for a search's window.
    """
    out = []
    for s in searches:
        model: EventModel = model_factory(s.efficiency_obj())
        model.sigma_n_cm2 = sigma_1_cm2
        live = s.livetime()
        dates, w = live.grid(step_days)
        r = model.rate_curve(dates)                        # events / (t yr) at sigma_1 on each date
        mean_seasonal = float(np.sum(r * w) / np.sum(w)) if np.sum(w) > 0 else 0.0
        # the annual mean, for the counterfactual of a calendar-blind exposure
        year = int(s.start[:4])
        from .earth import year_grid
        r_year = model.rate_curve(year_grid(year, 73))
        mean_annual = float(np.mean(r_year))
        out.append({"search": s.name, "window_keV": [s.E_lo_keV, s.E_hi_keV],
                    "exposure_tonne_year": s.exposure_tonne_year, "calendar": [s.start, s.end],
                    "expected_seasonal": mean_seasonal * s.exposure_tonne_year,
                    "expected_if_calendar_blind": mean_annual * s.exposure_tonne_year,
                    "seasonal_factor": (mean_seasonal / mean_annual) if mean_annual > 0 else None,
                    "source": s.source})
    return out
