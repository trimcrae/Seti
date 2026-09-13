# ISOTOPE — the impossible grain

**Signature S46** (`docs/necrofrontier.md` §2.XII, purity residue P1): *isotopic
purity in a solid without the nucleosynthetic package that nature attaches to
it.*  The first build in the necrofrontier order (§4: "one download, 2×10⁴
objects, a correlated natural signature the artificial case lacks by
construction").

---

## 1. The claim

Natural solids carry isotope anomalies in only two ways.  Either as
**correlated nucleosynthetic packages** — a supernova SiC X grain is ²⁸Si-rich
(δ²⁹Si down to ≈ −650 ‰, δ³⁰Si down to ≈ −1000 ‰ at the Si/S-zone end-member,
Lin et al. 2010, ApJ 709, 1157) **and** carries the rest of the ejecta: ⁴⁴Ti
(read as a ⁴⁴Ca excess), ²⁶Al/²⁷Al ~ 0.1–0.6, a ¹⁵N excess, a ¹²C excess — or as
**per-mil mass-dependent fractionation**, δ²⁹Si ≈ ½ δ³⁰Si at small amplitude.
There is no natural process that strips one element to a single isotope and
leaves its partners at solar.

Technology does exactly that.  Quantum-grade ²⁸Si is 99.99 %+ pure (δ²⁹Si ≈
δ³⁰Si ≈ −999 ‰) and is made from ordinary feedstock, so its C, N, Al–Mg and
Ca–Ti — whatever traces it carries — are solar/terrestrial.  ¹²C diamond is the
carbon analogue.  So the observable is not purity, which supernova grains reach,
but **purity with solar partners**.

The Presolar Grain Database SiC release (Stephan et al. 2024, ApJS 270, 27;
20,230 grains) tabulates per grain δ²⁹Si, δ³⁰Si, ¹²C/¹³C, ¹⁴N/¹⁵N, ²⁶Al/²⁷Al and
the Ca–Ti system where measured, with a grain classification.  It is the one
public dataset in which this question can be asked of every object at once.

## 2. Novelty position

From `docs/necrofrontier.md` §2 (S46 *Record*) and §5 (group **g1, isotope
purity**, 10 fetches across three sweep runs, 1,713 abstracts): **unoccupied**.
No paper reads stable-isotope purity in solids as a technosignature.  The
nearest neighbours, all of which are *element-pattern* or *fluid* signatures:

| Adjacent work | Why it is not this |
|---|---|
| Whitmire & Wright 1980 — fission waste in stellar photospheres | An element abundance pattern in a star, not an isotope ratio in a solid |
| Catling, Krissansen-Totton & Robinson 2025 (ApJ 979, 137) — D/H depletion by fusion | A planetary-water isotope ratio; one element, one process, a fluid |
| Ellery 2025 (arXiv:2510.00082) — Th/Nd, Th/Ba from lunar reactors | Element ratios from fission products |
| Carrigan 2010 "Starry Messages" — interstellar archaeology | Framework only, no isotope statistic |
| The presolar-grain literature itself (Zinner; Hoppe; Nittler & Ciesla 2016; Stephan+2024) | Classifies grains by their *natural* packages; the un-packaged residue is the "U/unusual" bin, never read for artificiality |

The channel therefore adds one statistic to a table the field already
maintains, and the statistic's null is set by the table's own classified X
grains rather than by a model.  Verification of the position is the sweep
record in `results/necrofrontier/` (g1); nothing above was read in full text in
this sandbox.

## 3. Method

`src/seti/isotope/`: `acquire.py` (routes, readers, schema), `purity.py` (the
detector, pure functions), `run.py` (stages).  Thresholds and header patterns
live in `config/isotope.yaml`.

**Acquisition — a route ledger.**  Tried in order, every outcome recorded:
(1) a `--table-url` a human found; (2) a `--table-path` / the cached download;
(3) a crawl of `presolar.physics.wustl.edu/presolar-grain-database/` (https and
http, 120 s timeout) for links ending `.xlsx/.xls/.csv/.zip`; (4) an EarthChem
Library HTML search for "Presolar Grain Database" — the site's own search form
is discovered from `home.php` and used first, then guessed search URLs — with
record pages followed to their download links; (5) a DataCite lookup of the
IEDA/EarthChem DOI by title, then the landing page's download links.  A page
that answers but carries no grain table is `REACHED_NO_TABLE`, not a hit.  If
nothing lands the verdict is `NO_DATA_REACHED` and the run names every route it
tried.

**Readers.**  csv (sniffed dialect, three encodings), xlsx through `openpyxl`
when importable and otherwise a stdlib `zipfile` + XML reader (shared strings,
inline strings, sparse rows), zip containers recursed.  The header row is
*found* — the row in the first 60 whose cells resolve the most roles — and the
sheet with the best header wins, so a "Read me" sheet or a two-line preamble
cannot derail the parse.

**Schema at runtime.**  Every role (grain id, meteorite, type, δ²⁹Si, δ³⁰Si,
¹²C/¹³C, ¹⁴N/¹⁵N, ²⁶Al/²⁷Al, ⁴⁴Ti/⁴⁸Ti, δ⁴⁴Ca and each one's error) is resolved
by regex over a normalised header (δ/Δ/delta → `d`, ± → `pm`, σ → `sig`,
super/subscripts → digits, punctuation removed; `err[δ(29Si/28Si)]` →
`errd29si28si`).  Value roles exclude the error marker; error roles require it;
a unit multiplier in the header (`×10⁻³`) is applied.  **A role that is not found
is `None` and the test that needs it is recorded as not run — never passed.**
`"<0.001"` entries become upper limits, which matter for ²⁶Al/²⁷Al.

**The empirical envelope.**  From the grains whose type matches the X regex:
the minimum δ²⁹Si and the minimum δ³⁰Si (each with the grain that set it), the
2-D convex hull of the X population, and the mixing line from solar (0, 0) to
the most distant X grain.  A grain is **beyond the envelope** when δ²⁹Si + 3σ <
min δ²⁹Si(X) *and* δ³⁰Si + 3σ < min δ³⁰Si(X), with its own errors.  Without a
classified X grain the configured Lin et al. end-member (−650, −1000) is the
floor — which nothing can be beyond in δ³⁰Si, so the fallback is conservative
and the summary says it was used.  The X *package* (the partner values a
supernova grain carries) is the median of the database's own X grains per
ratio where ≥ 10 of them carry it, else the configured fallback.

**The partner test.**  Per measured partner: *z* to solar (¹²C/¹³C = 89,
¹⁴N/¹⁵N = 272, ²⁶Al/²⁷Al = 0 one-sided, ⁴⁴Ti/⁴⁸Ti = 0.0221 or δ⁴⁴Ca = 0),
`solar_consistent` = |*z*| ≤ 3, and *z* to the X package, `package_excluded` =
|*z*| ≥ 3.  ²⁶Al/²⁷Al upper limits are solar-consistent by construction and
exclude the package below 0.01.  ⁴⁴Ti/⁴⁸Ti and δ⁴⁴Ca share one slot so a grain
is never double-counted.

**Classes.**  `NO_SILICON` (value or error absent — untestable);
`ORDINARY`; `FRACTIONATION` (δ²⁹ ≈ ½ δ³⁰ within errors, |δ| ≤ 60 ‰, resolved from
zero); `NATURAL_PACKAGE` (beyond the envelope with a partner resolved away from
solar — a supernova grain, whatever its label); `INSUFFICIENT_PANEL` (beyond the
envelope with no partner, or fewer than two, measured — **never a candidate**);
`PURITY_CANDIDATE` (beyond the envelope, ≥ 2 partners measured, all
solar-consistent; grade A when every partner also excludes the package, B when
some do, C when none has the power to); `CARBON_PURITY_CANDIDATE` (the analogue:
¹²C/¹³C − 3σ above the most extreme *classified* natural grain, with ¹⁴N/¹⁵N
solar and both Si ratios within 3σ of solar).  The flag
`CONTAMINATION_SUSPECT` marks δ²⁹Si ≈ δ³⁰Si ≲ −950 ‰ with no C and no N measured
— what a silicon-wafer fragment looks like in a SIMS mount.

**Outputs** (`results/isotope/`): `probe.json`, `acquire.json` (route ledger,
role resolution), `screen.json` (envelope, counts, type census), `grains.csv`
(every grain, artifact only), `candidates.csv`, `frontier.csv` (the 25 most
²⁸Si-pure grains regardless of class, so nothing hides behind a label), and
`summary.json` with `verdict` ∈ {`NO_DATA_REACHED`, `NO_PURITY_CANDIDATE`,
`PURITY_CANDIDATES`}, counts per class, the role resolution, the envelope
numbers, and the candidates.

## 4. Contamination ledger

A candidate is pending until each of these is excluded, in order:

* **Wafer / mount fragment.**  Pure ²⁸Si with nothing else measured is what a
  chip of the silicon wafer the grains are pressed into looks like.  The C and
  N isotopes of a real SiC grain decide; the flag `CONTAMINATION_SUSPECT` and
  the `INSUFFICIENT_PANEL` class hold the line.  A candidate must carry C or N.
* **Mislabelled supernova grain.**  The envelope is set by grains the authors
  *classified* X; a grain typed "U" or "D" or blank at −700/−900 with ²⁶Al and
  ⁴⁴Ca is an X grain whatever its label, and the partner test names it
  `NATURAL_PACKAGE`.  The frontier list shows the label of every extreme grain.
* **SIMS / NanoSIMS precision and instrumental mass fractionation** (10–50 ‰):
  the 3σ margins use each grain's own tabulated errors; a grain without an
  error is `NO_SILICON`, not a candidate.  Mass-dependent fractionation is its
  own class.
* **Type D and the unclassified residue** are where measurement problems
  concentrate and also where a hit would hide; both facts are why the partner
  test, not the label, is the discriminant.
* **Unit multipliers and limits** in headers (`×10⁻³`, `<`) are parsed, and the
  scale applied is written into the role resolution for a human to check.
* **Duplicate grains** across the database's source papers: the grain id and
  meteorite are carried; the assess stage does not yet dedupe by literature
  origin (a candidate must be checked against its source paper by hand).
* **Solar ¹⁴N/¹⁵N is not one number.**  The reference is 272 (terrestrial air);
  the solar-wind value is ~441.  A grain "solar" in N at 3σ under 272 is
  terrestrial-like, which for a contamination hypothesis is the relevant
  reference; the value is a config entry, not a constant.

## 5. The data-access blocker, stated plainly

The table was **not reached** from the GitHub runner in the three
`necrofrontier-probe` runs of 2026-09-13 (`results/necrofrontier/probe.json` on
`main`): `presolar.physics.wustl.edu` timed out at connect on both 443 and 80
(90 s), the DOI asserted from memory returned "DOI Not Found", and the ADS
record sits behind an AWS WAF challenge.  `ecl.earthchem.org` answers.  The
channel therefore ships with a search over the EarthChem Library and a DataCite
DOI lookup as routes 4 and 5, and with a `table_url` workflow input for the
moment a human locates the release.  Until a run lands the table, every
`summary.json` this channel writes says `NO_DATA_REACHED` and lists the routes
tried; no grain has been classified, and nothing here is a result.  The sandbox
has no network at all, so the suite (`tests/test_isotope.py`) exercises every
route with a scripted fetcher and never touches a socket.

## 6. Running

```
python -m seti.isotope.run --stage probe
python -m seti.isotope.run --stage all --table-url https://…/PGD_SiC_2024.xlsx
python -m seti.isotope.run --stage screen,assess            # from the cached table
python -m pytest tests/test_isotope.py -q -p no:cacheprovider
```

Workflow: `.github/workflows/isotope.yml` (`workflow_dispatch`; inputs `stage`,
`table_url`), which commits `probe.json`, `acquire.json`, `screen.json`,
`summary.json`, `candidates.csv` and `frontier.csv` back through
`scripts/commit_results.sh`.
