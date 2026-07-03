# rev9 PROPOSED EDITS — SEMANTIC-LOCK rows (prose untouched; for human review)

_generated 2026-06-04T15:07:00_

All edits below are PROPOSALS. `paper/main_vision.tex` prose was NOT touched.

## Lock row 1 — Table 1 large AP gap (`paper/main_vision.tex` line ~411)
- Old: AP$_\text{large}$ NPU − GPU = $+0.001$ ($+0.2\%$, $p=0.92$, n.s.) — **'unaffected on large'**.
- rev9: $-0.1000$ ($-21.0\%$). **Magnitude moved from ≈0 to non-negligible.**
- Impact: the 'large vehicles effectively unaffected' narrative in §5.1 (line ~416) is **refuted by rev9**. Editor must decide whether to keep historical Table 1 alongside rev9 or replace.

## Lock row 2 — §5.1 three-size narrative (`paper/main_vision.tex` line ~416)
- Old: '$-47\%$ on small, $-19.5\%$ on medium, effectively zero ($+0.2\%$, n.s.) on large'.
- rev9: small $-56.4\%$, medium $-35.2\%$, large $-21.0\%$.
- Impact: 'large effectively zero' is no longer true. Editor must rewrite either as (a) cite Table 1 as historical and the rev9 number as the current reproducible value, or (b) drop the parenthetical.

## Lock row 3 — §5.3 C1 inversion magnitude (`paper/main_vision.tex` line ~543)
- Old: '$+25\%$ at L1$_{\text{light}}$ ($0.105$ vs $0.084$) and $+23\%$ at L2$_{\text{LM}}$ ($0.065$ vs $0.053$)'.
- rev9: L1_light $0.084$ vs $0.082$ ($+2.4\%$); L2_LM $0.053$ vs $0.029$ ($+79.3\%$).
- Impact: SIGN PRESERVED (reversal still holds). MAGNITUDES different (smaller at L1, larger or different at L2). Editor decides whether to keep +25%/+23% with footnote or use rev9 magnitudes.

## Lock row 4 — Decomposition $97\%$ caption (`paper/main_vision.tex` line ~793) and abstract
- Old: $97\%$ of NPU-path loss from quantization.
- rev9: $169\%$.
- Impact: narrative direction preserved (quantization dominates). Magnitude change small. Editor likely safe to update one number.

## Lock row 5 — Worst/mean ratio '1.75–3.5×' (abstract, §6, conclusion)
- L1_light: old $1.75\times$ → rev9 $+0.95\times$. sign preserved
- L2_LM: old $3.5\times$ → rev9 $+0.31\times$. sign preserved
