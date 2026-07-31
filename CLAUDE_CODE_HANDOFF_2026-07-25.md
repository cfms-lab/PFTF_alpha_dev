# Claude Code Handoff — PFTF_alpha_dev (2026-07-25)

> Rewritten fresh at the user's request. Prior handoff content intentionally discarded.
> This captures the state at the end of the manuscript+scoping session so work can resume later.

## TL;DR — where things stand

- **The PFTF_alpha paper is an honest negative/limits result** (local SPD anisotropic alpha does
  NOT beat prior-art on real held-out data). Target journal = **RPJ (Rapid Prototyping Journal,
  Emerald)**.
- **DECISION (user, 2026-07-25): the paper is SHELVED — "장점 부각 실패".** A purely negative
  result is not submittable as-is; it needs **at least one clear headline advantage**, which we do
  not currently have (M1>B4 and G4 fail-closed are only *secondary* gains).
- **Manuscripts are complete and committed** (expanded + RPJ house-style template applied, renamed,
  plus a beginner explainer). Nothing left to do on the drafts themselves.
- **Next candidate to create a headline advantage:** a global **graph-cut inside/outside tet
  labeling** method. A full feasibility scope is written and committed. **Recommendation: run the
  Phase-0 pilot (~1 week) ONLY, and only after the user accepts a strategic thesis pivot** (see
  "Pending decision" below). No pilot code has been written yet.

## Pending decision (blocks the next action)

Running the graph-cut path means **pivoting the paper's thesis away from PFTF**. If the graph-cut
works, the headline advantage belongs to the *graph-cut*, not PFTF — P1/P2 still lose, so the
paper becomes "**global labeling beats all local anisotropic alpha, including our PFTF**"
(local-vs-global), not "PFTF works." That is a legitimate, arguably stronger paper, but it is a
different thesis. **Do not start Phase-0 coding until the user confirms this pivot is acceptable.**

## Immediate next action (once/if approved)

Phase-0 pilot from `docs/GRAPHCUT_LABELING_SCOPE.md` §5 — build tet-adjacency + one data term +
`scipy.sparse.csgraph.maximum_flow` + label→mesh on **one** synthetic thin-gap case, and check
whether the cut separates the two sheets **without opening a hole** (visual + Betti/false-bridge on
that one case). This isolates the single make-or-break risk (§3 of the scope: no sensor visibility
exists, so the data term must be synthesized from unoriented PCA normals). ~1 week. GO/NO-GO gate
before any further investment.

## What was done this session (all committed & pushed to origin/main)

Commits (newest first):
- `2264336` — add `docs/GRAPHCUT_LABELING_SCOPE.md` (feasibility scope, grounded code map).
- `3a9dff6` — park held Figure 1 mockup in `draft/pics/PFTF_alpha_fig1_mock.{svg,png}` (NOT in paper).
- `86e7d9d` — rename manuscripts to `PFTF_alpha_*` + add beginner explainer `PFTF_alpha_소개`.
- `f4eac90` — expand RPJ manuscript + apply Emerald house-style template.
- (`4204911` and earlier = previous session; see git log.)

`main == origin/main`, working tree clean apart from this handoff file.

## Draft file inventory (`draft/`)

- `PFTF_alpha_en.{tex,pdf}` — **peer-blind** RPJ submission (double-anonymous). ~5.3k/9000 words.
  RPJ template applied: 12pt A4 double-spaced, structured abstract (Purpose/Design/Findings/
  Practical implications/Originality-value), keywords + "Article classification: Research paper",
  peer-blind short-title running header, Roman-numeral tables, Harvard (author-date) natbib. 14 pp.
- `PFTF_alpha_kr.{tex,pdf}` — internal detailed Korean version (no length limit, author info OK).
- `PFTF_alpha_소개.{tex,pdf}` — beginner explainer for grad students / humanities high-schoolers.
  Analogy-driven (star-constellation cloud, ball-rolling alpha, rugby-ball anisotropy,
  self-driving-car fail-closed), footnote-heavy, no math beyond high-school. xelatex+kotex, 5 pp.
- `pics/PFTF_alpha_fig1_mock.{svg,png}` — **held** Figure 1 mockup (4-panel "rescore vs. reconnect"
  representative diagram). NOT inserted into any paper. See "Figure proposal" below.

## Hard constraints (still apply)

- `PFTF_alpha_en.tex` is **peer-blind**: no author name/affiliation/email/ORCID/acknowledgments/
  funding/self-repo links/local paths. Verify with a grep before any commit that touches it.
- **RPJ length = ≤9,000 words total** (abstract + refs + every table/figure counted ~280 words) —
  RPJ governs, overriding the generic 4,000-word default in `논문생성규칙.md`.
- **Do NOT build PDF/DOCX except on explicit user request.** (This session the user explicitly asked
  for the 소개 PDF and the manuscript rebuilds.)
- **References: only 5 verified sources** (edelsbrunner1992/1994, hoppe1992, labatut2009,
  zhou2016thingi10k). **Do not fabricate new bib entries.**
- Claim boundaries: **"beats prior art" is forbidden** (P3 refutes it); G4 certifies only base
  Delaunay connectivity, **not** the anisotropic/PFTF complex. `promotion_supported=false`.

## Build commands (verified working — MiKTeX)

```bash
# English (pdflatex; resolves natbib + \ref over passes)
latexmk -cd -pdf -g -interaction=nonstopmode "draft/PFTF_alpha_en.tex"
# Korean detailed + beginner explainer (xelatex + kotex for Hangul)
latexmk -cd -xelatex -g -interaction=nonstopmode "draft/PFTF_alpha_kr.tex"
latexmk -cd -xelatex -g -interaction=nonstopmode "draft/PFTF_alpha_소개.tex"
```
LaTeX aux artifacts are gitignored (`draft/*.aux,*.log,*.fls,*.fdb_latexmk,*.out,*.xdv,*.toc`);
delete stale ones after renames. All three built clean this session (0 errors; one harmless 2.7pt
overfull hbox in en).

## The research story (definitive, negative-for-promotion but honest)

Promotion rule = beat B4 AND B5 on held-out geometry+topology (Gate 1) + exact fail-closed fallback
(Gate 2). **Gate 1 NOT met.**
- **P3 real held-out (headline):** on real meshes B5 (normal-anisotropic, prior art) wins (mean F
  0.868); no PFTF method beats it — P1 0.727, P2 0.728, M1 0.774. Casewise B4/B5 envelope margins
  all below zero (M1 −0.094 CI [−0.11,−0.08], 2/40 cleared; P1/P2 −0.14, 0/40). Local SPD metric
  adds NO value on real data. Reverses the synthetic picture (there B5 was degenerate due to a panel
  thin-gap pathology).
- **M1 (the real positive, secondary):** `weighted_alpha.py`. Regular/power alpha via 4D lift,
  density weights, proper power circumradius scoring. scale 0.375 strictly dominates B4 on synthetic
  and beats B4 on real. Does NOT beat B5.
- **G4 (Gate 2, secondary):** `g4_fallback.py`. Deployed fail-closed exact fallback; no-silent-
  false-safe; certifies only base Delaunay. 9 failure-mode tests; 96-pt panel always fails closed.
- **Universal negatives:** all removal-based false-bridge interventions fail (resampling-persistence
  AUC≈0.50, backoff, oriented-normal removal/offset/signed labeling, etc.). Two causes: deleting a
  cell from a fixed complex opens a new hole; and the hardest family's inter-sheet gap < sample
  spacing (41% of kNN edges cross sheets) so NO local method has the info. Only connectivity change
  (M1) helps, and even it can't resolve the under-resolved gap.
- **One untested path = global graph-cut labeling** → the subject of the scope doc.

## Graph-cut scope (the resume point) — `docs/GRAPHCUT_LABELING_SCOPE.md`

Grounded in a full code read. Key points:
- **Reusable (~80% of infra):** all endpoints (`surface.py:366 evaluate_surface`), PCA normals
  (`adaptive.py:624`), Delaunay (`filtration.py:137`), tet dual-graph+bridge code (`adaptive.py:750`),
  **`scipy.sparse.csgraph.maximum_flow`** (no new dependency; deps are only numpy+scipy).
- **Build-new:** (1) tet-adjacency graph with an outside/source node (`.neighbors` currently
  discarded at `filtration.py:140`); (2) **data term = the crux** (no sensor rays/visibility exist;
  normals are unoriented — must synthesize from oriented normals / hull-source / free-space proxy);
  (3) smoothness term; (4) max-flow + label→`SurfaceMesh`; (5) single-shot harness path (harness is
  built around monotone score sweeps `surface_at`/`_freeze_multiplier`); (6) registration in
  `BaselineID` (`baselines.py:41`), `_adaptive` (`real_heldout.py:153`), METHODS/CANDIDATES.
- **Effort:** Phase 0 pilot ~1 week (go/no-go); full held-out verdict ~3–4 weeks; new code ~1 module
  400–600 lines + tests.
- **Dominant risk:** does *global* consistency of normals-only signal separate the thin gap where
  *local* B5 (which also uses normals) failed? Plausible, not guaranteed — the pilot answers it.
- **Recommendation:** GO for Phase 0 only, pending the thesis-pivot decision above.

## Figure proposal (held, not inserted)

Recommended Figure 1 大표도 = 4-panel "rescore vs. reconnect" schematic in the lab's family style
(gray wireframe, red source/blue receiver, bold (a)–(d), indigo/green accents): (a) scan→α-complex
→mesh→print motivation; (b) B4/B5/P1/P2 rescore ONE fixed complex → false bridge survives (✗);
(c) M1 changes connectivity via the weighted lift → bridge gone (✓); (d) real held-out verdict —
all candidate margins below the casewise B4/B5 envelope, B5 best, "▶ FN=0 (G4)". Mockup at
`draft/pics/PFTF_alpha_fig1_mock.{svg,png}`. The lab's shared `signature_en_*` family figures live in
`D:\cfms-research-vault\_attachments\` (variant `f456` highlights FN=0, matching G4). If the paper
resumes, also consider adding a `signature_en_*` figure for family positioning.

## Obsidian vault state (SEPARATE git repo — I did NOT commit it; commit only if user asks)

- `D:\cfms-research-vault\Papers\투고현황.md`: PFTF_alpha row = **보류 / ToDo: 장점 부각 실패**;
  frontmatter `graph_badges` has `PFTF_alpha: "장점부각실패"` (non-standard 2nd field → slate border);
  "다음 액션" has a 장점부각실패 ToDo item. Detailed TODO row updated to 보류 with the graph-cut
  candidate noted.
- `D:\cfms-research-vault\graphify-out\graph.html`: PFTF_alpha node now labeled `[장점부각실패]`
  (regenerated via `graphify-out/status_badges.py`, the canonical mechanism). **Do NOT touch
  `graph_발견.html`.** To change the badge later: edit `graph_badges` in 투고현황.md, then run
  `python status_badges.py` from `graphify-out/`.

## Conventions / verify-before-commit

- `uv run ruff check .` (E,F,I,UP,B, line-length 88); `uv run pytest -q` (last known full suite 161
  passed — a graph-cut method must add `tests/test_<module>.py`). Not re-run this session (docs/
  manuscript only changed).
- Commit message file via `git commit -F` (PowerShell heredoc `@'...'@` FAILS in the Bash tool — use
  a Bash heredoc to `$TEMP/...`). End messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Memory files (`~/.claude/projects/.../memory/`): `rpj-template-manuscript` (has the shelve status +
  template facts), `p3-real-heldout-gate1`, `m1-weighted-alpha`, `m2-thin-gap-exhausted`,
  `g4-fail-closed-deployment`, `p1-bridge-intervention-findings` (+ MEMORY.md index).

## Do NOT redo (settled)

- Journal = RPJ; framing = negative/limits. Gate 1 unachievable synthetically (thin-gap) and on real
  data (B5 wins) — do NOT run more *local* method probes hoping to beat B5; that space is exhausted.
- M1 scoring MUST be weighted-power circumradius (not ordinary) — settled.
- The ONLY remaining path to a positive claim is the global graph-cut (this handoff's resume point),
  subject to the pilot and the thesis-pivot decision.
