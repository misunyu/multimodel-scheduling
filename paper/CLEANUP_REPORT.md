# paper/ cleanup report (PART B)

Scope: **only `paper/`**. `results/`, `scripts/`, and everything outside `paper/` untouched.
`paper/results_data/` (PART A package) preserved. **No file deleted** — see build caveat below; all
moved files are reversible in `paper/_unused_archive/`.

## Git safety net
At cleanup time `git status paper/` showed: `main_vision.tex` modified; all figures/tables and
`results_data/` untracked (`??`). **Not auto-committed** (per project workflow: the user runs git
themselves). Recommend committing before finalizing deletions.

## Dependency closure of the current paper (`main_vision.tex`)
Parsed references:
- `\includegraphics{vis_failure_examples.pdf}` — the **only** graphics include.
- `\bibliography{main}` → `main.bib`; `\bibliographystyle{splncs04}` → `splncs04.bst`.
- `\documentclass{llncs}` → `llncs.cls`; `\usepackage{accv, accvabbrv, orcidlink, axessibility, ...}`.
- **No `\input` / `\include`** — the tables are inline `tabular` in `main_vision.tex`.

## Kept (whitelist)
| file | why |
|---|---|
| `main_vision.tex` | the paper |
| `figures/vis_failure_examples.pdf` | only `\includegraphics` target (Fig.1) |
| `results_data/` | PART A reproduction package (preserved) |

## Deleted (build artifacts)
**None.** No `*.aux/*.log/*.bbl/...` or `__pycache__` were present in `paper/`.

## Archived for your review — `paper/_unused_archive/` (NOT deleted, reversible)
All provably unreferenced by `main_vision.tex` (no `\input`; not `\includegraphics`'d):

**(B) obsolete — old auto-generated table fragments, superseded by the inline tables** (`_unused_archive/tables/`, 13 files):
`capacity.tex, decomposition.tex, gen_decomp.tex, gen_decomp_absolute.tex, gen_decomp_appendix_v8.tex, gen_gain.tex, gen_gain_appendix_v8.tex, main_comparison.tex, natural.tex, partA.tex, per_class.tex, schedule_shift.tex, single_stream.tex`
(The current paper builds its tables inline — see PART A `results_data/` for the live regenerable versions; none of these are `\input`.)

**(C) ambiguous — unused figures (may be wanted for a future revision)** (`_unused_archive/figures/`, 5 files):
`gen_cstar.pdf, motiv_partB_worst.pdf, motiv_reversal.pdf, motiv_single_stream.pdf, motiv_size_dist.pdf`
(None are `\includegraphics`'d in the current `main_vision.tex`. Left for your decision.)

## Clean-build verification — **COULD NOT RUN**
Per the spec, deletion is gated on a passing `pdflatex → bibtex → pdflatex×2` build. That build
**cannot be run in this environment**:
- **No LaTeX toolchain** (`pdflatex`/`bibtex` not installed).
- **Build dependencies absent from the repo** (never in `paper/`): `main.bib`, `llncs.cls`,
  `accv.sty`, `accvabbrv.sty`, `splncs04.bst`, `orcidlink.sty`, `axessibility.sty`.
- `\includegraphics{vis_failure_examples.pdf}` has no path and there is no `\graphicspath`, so a
  root build also needs that PDF in `paper/` root (it currently lives in `figures/`).

Because a passing build cannot be demonstrated, **nothing was deleted** (the spec forbids
force-deletion without build verification). The archived files are only *moved*, fully reversible.

## Result
`paper/` now contains, besides the package:
```
paper/
├── main_vision.tex
├── figures/vis_failure_examples.pdf
├── results_data/            (PART A — kept)
└── _unused_archive/         (moved, reversible: 13 tables + 5 figures)
```

## Recommended next steps (for you, to finalize)
1. Provide the ACCV kit (`accv.sty`, `accvabbrv.sty`, `llncs.cls`, `splncs04.bst`, `orcidlink.sty`,
   `axessibility.sty`) and `main.bib`, and either move `vis_failure_examples.pdf` to `paper/` root
   or add `\graphicspath{{figures/}}`.
2. Run `pdflatex main_vision → bibtex main → pdflatex ×2`; confirm 0 errors / 0 undefined refs.
3. If it builds clean, the `(B)` table fragments in `_unused_archive/` are safe to delete; keep or
   delete the `(C)` figures per your judgement.
4. Add a `.gitignore` for build artifacts (`*.aux *.log *.bbl *.blg *.out *.synctex.gz`).

No file under `paper/` was lost; everything is either kept or in `_unused_archive/`.
