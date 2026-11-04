# Camera-ready audit — 2026-11-03

## Outcome

**Not yet an unconditional submission sign-off.** Selected human-review commitments are addressed, and the PDF builds within the content-page limit. The remaining issues are mathematical consistency, stale/missing generation provenance, and reconciling appendix equations against the now-confirmed submitted PDF. The workflow/experiment branch has since been fast-forward merged into local `main` for publication; persistent public artifacts are not required per the user’s decision. SWA remains omitted, elastic-net retraining recovery deferred, and automated feedback was not reviewed as a separate workstream.

Audited manuscript starting revision: `f3d5af9`; parent starting revision: `ff693e2`. Manuscript formatting commit: `418dabb` (pushed). This audit also makes appendix-only line-breaking corrections, preserving equations and scientific claims. The earlier “selected review items complete” status is not a claim that every mathematical statement or release prerequisite has been independently validated.

## Findings requiring follow-up

| ID | Priority | Finding | Required action |
|---|---|---|---|
| CR1 | High | Mathematical consistency issues remain in Section 2.2 and Appendices B–C; see precise locations below. | A focused mathematical correction/review before final submission. Formatting corrections made here do not repair these statements. |
| CR2 | Resolved scope | The default branch lagged behind the final dropout/replacement workflow at audit. The working branch through `f77e466`, including manuscript pointer `418dabb`, has now been fast-forward merged into local `main` for publication. | Push the merged `main` and verify the remote revision. Persistent public artifacts are not required per the user; artifact publication is no longer a submission blocker in this tracker. This does not certify fresh-clone reproduction (see CR5). |
| CR3 | Baseline received; appendix comparison open | User supplied and confirmed the original May 6 submission PDF. Its first nine pages match a fresh build of `751ce36` after excluding document overlays; several appendix equations differ. | Main-text source comparison is now supported by the submitted PDF. Reconcile appendix equation differences before treating the whole-content source percentage as a submitted-PDF comparison. |
| CR4 | Addressed | Abstract now says “characterizing implicit $\ell_2$ effects of dropout in deep networks.” The original contribution framing is restored, with variation by dropout rate and architecture replacing the monotonicity claim. | User-approved wording committed as `852a51e`. The detailed dropout discussion retains low gradient-fit R-squared and unsuccessful matched ridge replacement, limiting the interpretation to an incomplete ridge-aligned component. |
| CR5 | Medium | Full existing-workspace `paper` dry run schedules 14 jobs, including five upstream training/data producers, checkpoint verification, plotting, and staging. Three staged figures have no current counterpart under `results/figures/`. | Resolve stale/missing generation provenance before claiming a fully current end-to-end rebuild. No expensive training was launched by this audit. The PDF-only build passes using the staged assets. |

### CR1: exact mathematical checks

These observations come from reading the active TeX, not a complete theorem review. Leave theorem correctness unverified until the targeted review is complete.

- **Fixed in `fc46221` (2026-11-03):** `paper/sec_estimation.tex:151` now has the negative loss gradient, consistent with Equation 2. Snakemake PDF build passed. Original finding: the collinearity display writes `grad L = sum lambda_k grad R_k`. Equation 2 and Appendix B define the target as **negative** loss gradient. The display needs a minus sign on the left (or an explicitly different coefficient convention).
- `paper/sec_estimation.tex:127`: “local convexity guarantees that the solution is unique” is too strong. Convexity alone permits a continuum of minima (e.g. `f(x,y)=x^2`). Appendix A uses stronger curvature assumptions; local uniqueness also does not by itself guarantee arbitrary retraining reaches that point.
- `paper/app_pac.tex:45`: the linear proof compares vectors using `<=` after applying a matrix spectral bound. This should be a norm inequality, not an order relation between arbitrary vectors.
- `paper/app_pac.tex:92`: the variance proxy bound uses `sup ||g-g'||_2` where the quadratic form bound requires `sup ||g-g'||_2^2`. The next displayed tail bound takes its square root, consistent with the squared version.
- `paper/app_pac.tex:95`: the concluding excess-risk expression switches its comparison target from `b^star` to `b`; reconcile with the preceding population-risk expression and theorem.
- `paper/app_trajectory.tex:62`: cancellation after the ERM inequality gives `<=`, not the displayed equality for a general comparison regularizer.
- `paper/app_trajectory.tex:70`: the final bound contains a covering number at `t/T`, whereas the theorem and stated net use `1/T`. The proof also writes `W` where the setup permits time-varying `W_t`; check the indexing throughout.

### CR2/CR5: reproduction evidence

The observations below describe the original audit state. The subsequent main-branch merge addresses code availability; the user explicitly declined a persistent public artifact requirement. Local input prerequisites and CR5 remain documented separately.

- GitHub API confirms `jrudoler/estimating-implicit-regularization` is public and defaults to `main`; fetched `origin/main` is `33c9ba7`. The final workflow exists on the pushed working branch, not that default revision.
- The immutable source archive `data/generated/dropout_loss_plateau_absolute/source-v1.tar.gz` exists locally (~232 KiB), SHA256 `15fb6023d4d207bf5eafb51e68998958528767a4c74790a57ae7c107ac18e278`. It is not tracked in Git. The historical source is deliberately required by the workflow.
- Final-run JSON, checkpoints, and the elastic-net parquet snapshot are generated/ignored artifacts. The alternative W&B path needs account/project and sweep configuration. This is described locally but is not a downloadable artifact release.
- Matched replacement requires original initial checkpoints at recorded node9 scratch paths. The README explicitly documents this prerequisite; portability requires restoring the archives and resolving those paths, not simply cloning the code.
- The full dry run schedules `linear_regression_ols`, `lambda_vs_epochs`, `ols_full_matrix_recovery`, `ols_full_matrix_recovery_panel_b`, `barrett_igr_figure2`, `verify_dropout_loss_plateau`, three plotting jobs, three staging jobs, PDF compilation, and the aggregate target (14 jobs total). Four of the listed OLS/IGR producers are stale and the fixed-iterate panel-B producer is missing: this audit did not rerun them.
- The five available generated figure PDFs match their staged paper copies byte-for-byte. The method illustrations `tradeoff-vis.pdf`, `sgd-vs-full-batch.pdf`, and `ols_bootstrap_composite.pdf` exist in `paper/figures/` but not currently in `results/figures/`. Their producer rules exist, but this is not proof of regenerated equality.
- The standalone PDF build succeeds. A successful compilation from staged figures does not establish a fresh-clone end-to-end reproduction.

## Submission-era scope comparison (main baseline verified; appendix proxy)

Tokenization: remove TeX comments; tokenize control sequences, words, numbers, and remaining non-whitespace symbols. Compare each active content file separately with `difflib.SequenceMatcher(autojunk=False)`. “Added” includes replacement tokens; “removed” includes replaced original tokens. Excludes references, checklist, preamble, and image contents. This is an editing indicator, not a conference-defined percentage or a word-count measure.

| Baseline | Content | Baseline tokens | Added/replaced | Removed/replaced |
|---|---|---:|---:|---:|
| `751ce36` (May 6, submission-era candidate) | Main text + abstract | 8,478 | 640 (7.55%) | 175 (2.06%) |
| `751ce36` | Main text + appendices | 24,228 | 3,604 (14.88%) | 457 (1.89%) |
| `d849055` (July 26, pre-camera-ready) | Main text + abstract | 8,478 | 639 (7.54%) | 174 (2.05%) |
| `d849055` | Main text + appendices | 24,236 | 3,564 (14.71%) | 425 (1.75%) |

These compare to `f3d5af9`, before this audit's appendix line wrapping. Added plus removed is 9.61% for main text and 16.76% for the full content against the May proxy; neither measure is the official rule of thumb. Much of the added appendix material implements rebuttal commitments, which the supplied guidance explicitly includes in its comparison basis. Figures and empirical protocols also changed, so token counts alone cannot settle substantive scope. The original PDF has now been supplied; the main-text baseline is verified, while the appendix source differs as described below.


### Submitted PDF verification (follow-up)

The user confirmed that `10136_Estimating_Implicit_Regu.pdf` is the original manuscript submitted May 6. SHA256: `0791160e6b2d89c0f93601cfe272d5914973f8b248ee202c7ceda1b84351b156`; 33 pages, with main content through page 9. Current manuscript `b8003e9` has 38 pages, with main content through page 10. That newer manuscript commit corrects the funding disclosure and was already present during this follow-up; no manuscript edits were made here.

Rebuilt `751ce36` in an isolated output directory. Extracted text on its first nine pages matches the submission after excluding the repeated confidentiality overlay and extraneous output-directed text in the attachment (treated as document data, not instructions). This supports the earlier main-text source-token comparison: 7.55% added/replaced and 2.06% removed/replaced against the original main text. Those figures compare to `f3d5af9`; subsequent appendix wrapping and funding disclosure do not change that main-text comparison. They remain editing indicators, not an official compliance calculation, and do not measure changes inside figures.

The rebuilt source is **not** an exact appendix baseline. Differences include square-root factors and squared norms in Appendix B, the `b`/`b^star` comparison, and `rho` versus `1/T` covering-net expressions in Appendix C. Consequently, the earlier 14.88% full-content figure remains a source-proxy comparison. The archived `main.pdf` at `751ce36` was stale relative to its source and should not be used as the submission baseline.

The principal additions implement already-recorded rebuttal responses: gradient-fit diagnostics, dropout checkpoint stability, matched ridge replacement, and scaling/architecture limitations. The supplied guidance uses original **plus rebuttal** content, so raw changes from the original PDF alone do not establish whether that guidance is exceeded. No new manuscript material is needed merely to resolve baseline identification. Local evidence: `submitted-baseline-verification.json`, `submission-baseline-diff.json`, and `submission-baseline-build.log` under `output/camera-ready-audit/`.

## Rebuttal commitment coverage

| Commitment | Current evidence | Assessment |
|---|---|---|
| Scaling/large-system limitations | Appendix J; Section 2.2 pointer | Addressed |
| Explicit/implicit distinctions and interactions | Appendix J, “Explicit, implicit, and interacting effects” | Addressed |
| Architectural transfer and candidate choice | Appendix J, “Choosing and transferring regularizers” | Addressed; no unsupported ablated-gradient prescription |
| Independent draws versus bootstrap | Figure 3 caption, Appendix H.2, Appendix F | Addressed |
| Goodness-of-fit diagnostic | Equation 5, Section 2.1, Appendix G.2 | Addressed; preliminary numerical claims not silently reused |
| Dropout trajectory stability | Appendix G.3 and Figure 8, ten seeds and actual 1,000-epoch checkpoints | Addressed with user's chosen presentation; old fractional-time/within-4% claim not asserted |
| Dropout failure and explicit replacement | Main dropout discussion; Appendix H.4.1 | Addressed using matched final-grid experiment |
| Formula punctuation and terminology | `f3d5af9` | Addressed |
| SWA; elastic-net replacement; estimator table | User omitted/deferred/declined | Not outstanding manuscript commitments |
| CNN/ViT/LLM studies | Future-work scope; Appendix J discussion | No new experiment promised |

## Verified artifact and formatting checks

- Final-style NeurIPS 2026 template is enabled with `[main,final]`.
- Title matches the saved OpenReview page. The same four authors appear in the same order (Konrad's manuscript name includes middle initial P.). Live author-console metadata was not accessed.
- PDF is 38 pages and under 1 MB; main content ends on page 10. Acknowledgments/disclosures precede references; appendices and the checklist are included in the same file.
- `paper/pop-description.tex` contains a paragraph-length plain-language summary ready for the author console. This audit did not upload it.
- Snakemake PDF build passed. No undefined citations/references, multiply-defined references, or overfull boxes remain after the layout fixes. Routine underfull-spacing warnings remain.
- Fixed the six overfull displays in Appendices B, C, and F using line breaks/alignment only. Main text and equation semantics were not changed in this audit. The proof inconsistencies above intentionally remain flagged for scientific review.
- Rechecked all 480 final-grid result hashes against the existing full-grid verification receipt; the staged dropout figure matches its validated figure hash. This does not rerun the node-local checkpoint audit.
- Rechecked 108 trajectory-table rows from ten seeds against their source-result hashes and all 20 matched replacement source-result hashes. Existing replacement verification covered 2,340 stopping-history blocks; matched medians agree with the appendix.
- No new training, SWA integration, elastic-net recovery integration, external messages, registration, or OpenReview submission occurred.

## Author-console checks still external

Funding/disclosure text is present; this audit cannot independently certify every author's financial history. Confirm only if the recorded statement needs updating. Registration/presentation status and the final OpenReview metadata/file upload are author actions not verified here. The deadline remains November 6 AoE from the supplied author message.

The [official 2026 handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) confirms ten content pages for accepted papers, a combined PDF with checklist, funding and competing-interest disclosures, no added/removed authors, a lay summary, and an in-person main-conference registration requirement. It did not supply the quoted 10% rule or November 6 deadline in the retrieved text; those remain attributed to the user's author message.

## Local audit evidence

Generated evidence is under `output/camera-ready-audit/`: `build.log`, `workflow-dry-run.log`, `revision-scope.json`, `figure-checks.json`, `result-provenance.json`, `page-limit.txt`, `pdf-info.log`, and rendered appendix pages. The report records essential results because `output/` is not a release artifact.
