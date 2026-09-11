# GOO — grey goo as a technosignature: propagation, the record, and our own risk

*Goo is what a replicator becomes when nothing stops it.*

**Not a search channel.** A computation-and-compilation paper in the LZEDGE
mould (`docs/lzedge.md`): no archive is queried; the model is `seti.goo`, the
inputs are `config/goo.yaml`, the manuscript is `paper/goo/`, and the
observational record it compiles is the published surveys plus this
repository's own committed channel results.  A companion perspective paper on
the other self-replicating technologies is `paper/replicators/`.

Built 2026-09-11 from a user question: *are there papers on how long it would
take grey goo to propagate through the galaxy / universe and looking for signs
of it? If not, write one, including how the findings (or lack of findings)
should change our thoughts on the likelihood of us grey-gooing ourselves; a
generalised version on other ways we could accidentally kill ourselves with
self-replicating technology; and a target journal.*

---

## 1. The question, and why it was open

Two literatures, fifty and forty years old, have never met:

* **Grey goo** (Drexler 1986 → Freitas 2000 "global ecophagy" → Phoenix &
  Drexler 2004 retraction → the existential-risk surveys) stops at the planet.
* **Self-replicating probes** (Hart 1975; Tipler 1980; Freitas 1980; Newman &
  Sagan 1981; Landis 1998; Bjørk 2007; Cotta & Morales 2009; Wiley 2011;
  Nicholson & Forgan 2013; Carroll-Nellenback+2019; Armstrong & Sandberg 2013;
  Olson 2015; Hanson+2021) assumes design, purpose and control.

The nearest neighbours: Sagan & Newman 1983 (probes would be "an
uncontrollable danger", so nobody builds them — a goo argument used against
Tipler); Stevens, Forgan & O'Malley-James 2016 (grey goo listed among
self-destruction modes, planetary signature sketched); Ellery 2022a,b, 2025
(designed probes as solar-system technosignatures and how to curb them);
Lacki 2025 (collisional cascades of dead megaswarms — the residue physics we
lean on).  Nobody has computed how far an *uncontrolled* replicator goes,
compiled what the data say, or drawn the implication for the civilisation
asking.  The runner-side sweep `results/goolit/` (28 phrase queries, decoy-aware
scan; `scripts/goolit_fetch.py`, `goolit.yml`) is the evidence for that claim
and was read on 2026-09-11 (run 34595836444, 121/121 fetches OK): **0 of 58
arXiv abstracts across the 28 phrase queries name grey/gray goo or ecophagy
at all**, none pairs uncontrolled replication with interstellar/galactic
propagation, and the only "self-replicating AND technosignature" hits are
Ellery's designed-probe papers. Of the manuscript's 109 references, 85 were
verified against the arXiv or Crossref record (two author lists that had been
left as placeholders were filled from it: Kowald 2015, JBIS 68, 383 for the
error-catastrophe paper; Chen, Ni & Ong 2022, EPJ Plus for the Lotka–Volterra
paper; Hanson+2021's third author corrected to McCarter; three DOIs
corrected), 16 are books/reports the APIs do not carry, and the JBIS/QJRAS/JET
papers (Freitas 1980, Landis 1998, Bostrom 2002, Hart 1975, Tipler 1980,
Sagan & Newman 1983) are marked *asserted* in `refs.bib` because neither index
carries them. Limitation of the novelty claim: the phrase queries reach arXiv
abstracts only; a JBIS- or IJA-only paper that never touched arXiv would not
be seen, and Sagan & Newman 1983 is exactly such a paper — the nearest
neighbours in the non-arXiv record are known from citation chains, not from
this sweep.

## 2. What the model says (all numbers from `results/goo/*.json`)

Organised by **dispersal class**, because dispersal — not replication rate —
sets how far a goo goes.

| scale | governing physics | timescale | residue / visibility |
|---|---|---|---|
| **planet, in place** | the surface can only radiate $4\pi R^2\sigma(T^4-T_{\rm eq}^4)$ at the temperature the machines survive; energy source (sunlight / crustal fission / oceanic D) | biosphere: hours (exponential-limited); crust: **5 kyr** at 500 K; bulk: **1.1 Myr** — independent of $\tau$ from 100 s to a year | a hot dead planet, then a refined dead surface; unobservable with any current instrument |
| **planet, lifted** | gravitational binding $3GM^2/5R$ | 7 d at $L_\odot$ (Dyson's number); 15 kyr at the 2000 K surface ceiling; 58 Myr on absorbed sunlight | — |
| **system, small bodies** | branching front, transport-limited | belt **14 yr**; Kuiper belt **1 kyr** | a partial/complete swarm at the feedstock's orbital temperature (169 K belt, 44 K Kuiper): a Dyson-scale waste-heat source while alive |
| **system, planets** | binding energy at the surface radiative ceiling | terrestrial 8 kyr; giant 143 kyr | — |
| **dead residue** | Lacki 2025 collisional time $P_{\rm orb}/f$; PR drag; blowout | micron devices gone in 4 yr; mm grains 10 kyr; m slabs 10 Myr but at $f=4\times10^{-7}$ | **theorem: a residue above the detection floor $f_{\min}$ lives at most $P_{\rm orb}/f_{\min}$** — 4 kyr at 2.7 AU for $10^{-3}$. Detectable dead goo is short-lived; the surveys bound *live* goo |
| **galaxy, passive** | radiation-pressure blowout ($\beta = 0.57Q/\rho s$; blowout below 0.57 µm); repulsion at the target star limits arrivals to $\beta < 1.38$, i.e. the **0.21–0.57 µm** window; epicyclic annulus 1.6 kpc; differential rotation phase-mixes it in **1.2 Gyr**; capture $\pi R^2(1+v_{\rm esc}^2/v^2)$ | belt's worth of 0.5 µm devices ($2.3\times10^{36}$) → **~200 intact arrivals/yr per Earth-like planet in the annulus** before survival losses; cumulative in 5 Gyr: $1.5\times10^5$ if the ISM survival time is $10^8$ yr, $6\times10^{10}$ at $10^9$ yr, ~0 at $10^7$ yr | no front — a cloud; each seeded system is visible only while alive |
| **galaxy, active** | $v_f = d/(d/v+\tau_b)$ | 0.64 Myr (0.1c, 10 yr) — 6.4 Myr (0.01c, 100 yr) — 500 Myr (30 km/s, 1 kyr) | Kardashev-III reprocessing while alive |
| **cosmos** | comoving reach $v\int dt/a$ | event horizon 16.6 Gly; $7\times10^8$ galaxies at 0.5c, $5\times10^9$ at 0.99c | — |

## 2A. The contact graph (`seti.goo.contact`, added 2026-09-11)

The user's reframing: *a model of how long it takes everything in the galaxy
to be touched by something that has been touched by something else.* Every
natural channel that moves mass between systems, each with four numbers
(carriers emitted or the measured standing density; speed; survival;
cross-section), every rate from the literature (`results/goolit_contact/`:
44 references, 40 verified, the number-bearing sentences recorded verbatim in
`REPORT.md`). Two definitions of touch — **strict** (lands on an Earth-sized
planet), **loose** (passes within 100 AU) — plus **capture** where a paper
gives the bound rate. Two questions — **natural** (every system emits: time
for everything to be touched) and **epidemic** (one system first: R0 = other
systems its material reaches over 5 Gyr; SI dynamics above R0 = 1).

| channel (source) | natural | R0 strict | R0 loose / capture |
|---|---|---|---|
| interstellar objects, n = 0.2 AU⁻³ (Do+2018; Jewitt & Seligman 2023; Dehnen+2022 capture 2 per kyr; Hands & Dehnen 2020; Portegies Zwart+2018) | **one impact on Earth per 135 Myr → 33 over its history**; 4×10⁴ passages within 100 AU per year; one bound capture per 500 yr | **2.8 — at the percolation threshold**; epidemic time 57 Gyr | 1.3×10¹⁴ / 7.6×10⁶ |
| impact ejecta from a planet (Melosh 2003; Worth+2013; Adams & Napier 2022) | one impact per 1.2 Tyr | 3×10⁻⁴ — planet-bound stays planet-bound | 1.5×10¹⁰ |
| Earth-grazing bodies captured by binaries (Siraj & Loeb 2020: 10⁷–10⁹) | — | 5×10⁻³ | 10⁸ (bound) |
| free-floating planets, 21 per star (Sumi+2023; Mróz+2017; Goulinski & Ribak 2018: 1 % of stars capture one) | one passage per 12 Gyr | ~0 | 0.33 / 0.01 |
| interstellar dust, ~10⁻⁴ m⁻² s⁻¹ (Grün+1993; Landgraf 2000; Krüger+2019) | **5×10¹⁷ grains on Earth per year** | 10²⁵ | 10³⁹ |
| blown-out belt devices (§2) | 200 per planet per year | 7×10¹⁹ (5×10¹⁴ at 10⁸ yr survival; ~0 at 10⁷) | 10³³ |
| stellar flybys within 2×10⁴ AU (Bailer-Jones+2018: 19.7 ± 2.2 per Myr within 1 pc, ∝ d²) | one per 5.4 Myr; all systems linked in 124 Myr | 0 (moves comets inward, not between stars) | 10³ |
| supernova ejecta (Knie+2004; Wallner+2016; Koll+2019) | one deposition per ~3 Myr | — | — |
| birth cluster, first 100 Myr (Belbruno+2012: 10¹⁴–3×10¹⁶ bodies to the nearest sibling; Adams & Spergel 2005: 10–16 000 captures per cluster; Levison+2010) | — | **5×10⁴ per sibling pair** — Earth has very likely been struck by material that condensed around another star | 10¹⁵ |

**Reading.** In the mass sense the Galaxy's contact graph is already complete.
In the strict single-source sense it is *at threshold* for interstellar
objects (R0 ≈ 3, generations of Gyr), far below for anything a planet ejects,
and far above only for dust-sized carriers — which are the ones least able to
survive. So contact is never the bottleneck for a goo among a system's small
bodies (size and survival are), and never the route for a goo confined to a
planet (R0 ≪ 1 through impacts and grazing bodies). The birth-cluster channel
is the one bulk rock-transfer epoch and no civilisation exists during it.

## 2B. Full text, not abstracts (user directive, 2026-09-11)

Abstract-level verification is not enough for the rates the models use, so
`scripts/goolit_fulltext.py` (`goolit-fulltext.yml`) fetches the **full text**
of every reference both sweeps name: arXiv e-print LaTeX sources (de-TeXed) or
PDFs (pypdf), open-access PDFs through the Unpaywall API for DOI-only papers,
and the web-only sources (Freitas 2000 ecophagy page, the FHI 2008 survey,
Bostrom 2002, Hanson 1998). Per paper it stores the plain text (arXiv / OA
licences; paywalled papers are recorded `NO_OA_TEXT` and never silently
skipped) and the sentences matching that paper's query terms — what the model
actually needs from it (`QUERIES` in the script). `results/goolit_fulltext/`.
Every number in `config/goo.yaml` is then re-read against the full-text
sentence that carries it; the two scan variables (rocks ejected from a system
over the age of the Earth; the Ulysses dust flux) are replaced where the text
supplies them.

## 3. Inference (`seti.goo.bayes`)

$N\sim{\rm Poisson}(\Lambda)$ civilisations in a 5 Gyr window; each makes an
interstellar-reaching goo with probability $p = p_{\rm goo}\,p_{\rm esc}$.

| datum | likelihood | shadowed? | memory |
|---|---|---|---|
| our belt and biosphere intact | $e^{-\Lambda p f}$ (SIA / naive) or 1 (SSA / anthropic shadow) | **yes** | 4 Gyr |
| no live swarm around $5\times10^6$ stars (Hephaistos II) | rate $< 3/(N_\star t_{\rm vis})$ | no | $\le P_{\rm orb}/f_{\min}$ (dead) / indefinite (live) |
| no galaxy with $>85\%$ reprocessing in $10^5$ (Ĝ III) | $e^{-N_{\rm gal}\Lambda p q}$ | no | indefinite (live) |

Prior-free 95 % exclusions on $p$: own galaxy naive 3/Λ (0.03 at Λ = 100);
own galaxy shadowed: none; external, persistent goo: $3\times10^{-7}$ at
Λ = 100 (**the binding constraint, five decades below the own-galaxy bound and
immune to the shadow**); external, dead residue visible for $10^{-4}$ of the
window: $3\times10^{-3}$.  Marginalised over log-uniform Λ ∈ [0.01, 10⁶]:
shadowed 0.25 (= prior), naive 0.031, external persistent $8\times10^{-6}$.

**The branch map** (the paper's central result).  The data touch only the
escaping branch: $p'_{\rm goo} = p_{\rm goo}[(1-p_{\rm esc}) + p_{\rm esc}\ell] /
[1 - p_{\rm goo}p_{\rm esc}(1-\ell)]$.  With $\ell = 10^{-3}$ the elicited
planet-scale accident probability moves by a factor **0.990** if 1 % of goo
accidents are interstellar, 0.90 if 10 % are, 0.50 if half are.  For a
civilisation at our stage $p_{\rm esc}$ is small: a molecular replicator loose
on a planet is planet-bound (§2 row 1); it needs space industry to be
system-scale, sub-micron size in space to be passive-interstellar, and a
starship to be active-interstellar.  **The astronomical silence is informative
about goo that travels and silent about goo that stays home; ours would stay
home.**  Corollary for the Great Filter: a planet-bound goo filter leaves a
galaxy exactly as quiet as the one we see.

## 4. Honest limits

* Substrate generality and one-seed sufficiency are assumed; nothing known can
  do this (Phoenix & Drexler 2004).  The numbers are conditional on the
  accident.
* The error catastrophe (arXiv:1605.02169) and replicator competition
  (arXiv:2209.14244) can stop a goo before it reaches a scale; a goo that dies
  of copying errors before crossing the Galaxy is, for the inference, a
  system-scale goo.
* The passive branch's ISM survival time has never been measured for a
  machine; the paper scans it ($10^6$–$10^9$ yr) and says so.
* The anthropic choice (SSA vs SIA) does not change the conclusion in §3,
  which rests on the external galaxies.
* Charter note: this paper is not a null-result write-up — it is a propagation
  calculation, a compiled record and an inference — but it does report what the
  record does *not* constrain, because that is the user's question.

## 5. Observations that would change the answer

1. A temperate rocky planet with a refined, uniform, chemically dead surface
   (the planetary residue) — beyond current reach, not beyond a large
   direct-imaging telescope.
2. Warm dust around an old, metal-poor or halo star (OSSUARY, S7; Lacki 2025
   named the population) — the system-scale residue where no natural
   background exists.
3. A Dyson candidate whose temperature matches its host's belt rather than its
   habitable zone: waste heat where the rock was, not where the energy is
   wanted — the discriminant between goo and purpose.

## 6. Signature taxonomy entries

* **S44 Ecophagic planet** — a temperate rocky planet radiating at the surface
  ceiling with no atmospheric disequilibrium, then a refined dead surface.
  Unsearchable at catalogue level; listed with the numbers that say why.
* **S45 Feedstock-temperature swarm** — a waste-heat excess whose colour
  temperature matches the host's small-body belt (170 K at 2.7 AU, 45 K at
  40 AU) rather than a habitable-zone or engineering optimum.  Searchable in
  the same catalogues as every Dyson search; the discriminant is the
  temperature–orbit pairing.  → OSSUARY's old/metal-poor sample is the first
  place to look.

## 7. Target journal

See `paper/goo/JOURNAL.md`.  Recommendation: **International Journal of
Astrobiology** (Cambridge) for the main paper — it published Stevens+2016,
Ellery 2022a,b, Nicholson & Forgan 2013, Bjørk 2007, Osmanov 2020 and Olson
2017, i.e. every nearest neighbour; **Acta Astronautica** (Armstrong &
Sandberg 2013's venue) as the alternative if the reviewers want the
cosmological reach foregrounded; **Futures** (Umbrello & Baum 2018; the
existential-risk special issues) for the generalised companion, with
**Risk Analysis** (Ćirković, Sandberg & Bostrom 2010) as the alternative.

## 8. Files

| path | role |
|---|---|
| `config/goo.yaml` | every parameter, with its source |
| `src/seti/goo/{planet,system,passive,active,bayes}.py` | the model |
| `src/seti/goo/run.py`, `figures.py` | stages, `results/goo/*.json`, `paper/goo/numbers.tex`, `results/goo/figures/` |
| `src/seti/goo/contact.py` | the contact graph (channels, strict/loose/capture touches, natural and epidemic questions) |
| `tests/test_goo.py`, `tests/test_goo_contact.py` | 36 offline tests against hand calculations |
| `scripts/goolit_contact_fetch.py`, `.github/workflows/goolit-contact.yml` | runner-side fetch of the mass-transport literature → `results/goolit_contact/` |
| `paper/goo/main.tex`, `refs.bib` | the manuscript |
| `paper/replicators/main.tex` | the generalised companion |
| `scripts/goolit_fetch.py`, `.github/workflows/goolit.yml` | runner-side novelty sweep and per-reference verification → `results/goolit/` |
