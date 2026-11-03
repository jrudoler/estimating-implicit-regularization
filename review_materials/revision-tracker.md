# Human-review revision tracker

Updated: 2026-11-03.

Scope: human reviews (GHL1, iMcK, rnFp), metareview, and author rebuttal commitments in [the source transcript](openreview-10136-reviews-discussion-2026-11-03.md). Automated feedback is deferred at the user's request.

Status describes manuscript completion, not merely whether code or results exist. The initial audit used the local manuscript and saved outputs; experiments were not rerun. Record manuscript commit and validation when closing an item. Completed manuscript edits are listed below with their commits and validation.

## Camera-ready editing constraints

User preference, recorded 2026-11-03: change the main text as little as possible. The camera-ready guidance supplied by the user gives a rule of thumb of no more than approximately 10% new content/modifications relative to original submission plus rebuttal content. This is an overall scope constraint, not a license for unlimited appendix additions or a precisely defined word-count rule.

- Prefer local wording corrections, short qualifications, and short appendix pointers over rewritten paragraphs or sections.
- Put detailed diagnostics, scaling discussion, estimator tables, and supporting results in compact appendix additions when appropriate.
- Keep corrections needed for accuracy in the main text wherever the affected claim occurs; an appendix caveat cannot repair a misleading main-text claim.
- Limit additions to human-review responses and rebuttal commitments/claims. Defer unrelated new results and optional extensions.
- For each completed item, record affected sections, whether main text changed, and commit. Before finalization, compare with the actual submitted manuscript and rebuttal, not merely the current local checkout. Locate that baseline before claiming a percentage; any textual diff measure is an editing aid, not certification of compliance.
- Per supplied guidance: main content limit is 10 pages; retain the checklist. Include acknowledgements/funding and applicable competing-interests disclosures after the conclusion and before references. Obtain factual disclosure details from authors; do not invent them.
- User-supplied deadline: November 6, 2026, AoE. Registration and OpenReview submission are author actions unless separately requested.

Minimal-edit plan: R1 gets a caption/methods wording change; R2–R3 go primarily in appendix discussion with only necessary main-text pointers; R4 becomes a compact scaling appendix; R5 gets a short main-text diagnostic definition/pointer and fuller appendix guidance. R6 belongs in appendix methods. R7 remains local editorial corrections. R8–R11 should use compact appendix evidence where possible, while correcting any affected main-text claim directly.

| ID | Source | Revision | Current status | Next action / evidence |
|---|---|---|---|---|
| R1 | iMcK | Distinguish independent dataset draws from bootstrap | Complete; committed and pushed | Commit `116f1e5`: Figure 3 caption and appendix methods now say “independent dataset draws”; bootstrap appendix explicitly excludes Figures 3–6. Main-text change limited to caption wording. |
| R2 | GHL1 | Define explicit versus implicit regularization and explain attribution/interactions | Complete; committed and pushed | `e55a5b5`: proposed paragraph in `paper/app_scaling.tex`; one appendix pointer added to Section 2.2. |
| R3 | GHL1 | Discuss architectural bias transfer and candidate-family selection | Complete; committed and pushed | `e55a5b5`: appendix discusses theory-guided families, common weight coordinates, and skips/pooling as future extensions. |
| R4 | All reviewers; metareview | Explain experimental scale and larger-system applicability | Complete; committed and pushed | `e55a5b5`: compact scaling appendix addresses complexity, r/p_eff, conditioning, gradient/endpoint costs, objectives/data, explicit penalties, access, and ablations. |
| R5 | rnFp; iMcK; metareview | Explain goodness of fit and diagnosing misspecification | Complete; committed and pushed | `492cbdd`: Section 2.1 defines uncentered gradient R-squared; Appendix G.2 explains interpretation, evaluation, noise/optimization, and limitations. Rebuttal numerical claim (>0.9) remains unverified and was not added. |
| R6 | iMcK | State estimator and replication scheme for each figure | Declined; optional | User chose not to add the appendix table. It was supplied in the rebuttal, not promised for the manuscript. R1 completes the explicit wording commitment. |
| R7 | rnFp | Formula punctuation and consistent optima/minima terminology | Open; editorial | Proofread active manuscript; preserve mathematical distinctions rather than mechanically replacing terms. |
| R8 | iMcK | Add dropout trajectory panel | Complete; committed and pushed | `d4956d2`: Appendix G.3 and Figure 8 show final-grid estimates every 1,000 epochs and stopping endpoints for ten seeds. User chose actual epochs over fractional times; historical within-4% result is not asserted for this replacement analysis. One main-text pointer. |
| R9 | iMcK; rnFp | Include dropout failure case and explicit-retraining validation | Results exist; not integrated | 80-run retraining study fails to reproduce target with tested penalties. Report controls and budget/stationarity limitations; reconcile with later dropout audits before finalizing narrative. |
| R10 | iMcK rebuttal | Incorporate reported SWA example | Results/figure exist; not integrated | Ten-seed summary and explicit replacement outputs exist. Check exact numbers; retain trajectory-conditional interpretation and matched schedule controls. |
| R11 | rnFp rebuttal | Qualify elastic-net retraining recovery if used | Results exist; not integrated | Seven matched endpoints show approximate functional recovery, not identical estimated-penalty weights. Optional supporting evidence for stronger validation. |
| R12 | GHL1; iMcK; rnFp | Additional CNN/ViT/LLM or architecture-transfer studies | Optional follow-up | Distinguish reviewer suggestions from commitments. Prioritize R4 and existing evidence before expanding experiments. |

## Dependencies affecting the review response

- Final dropout grid and manuscript are reconciled: Figure 5, caption, methods, introduction/empirical summary, and qualitative results were updated in `1f48e4b` and `062a85f`. Appendix G.3 uses ten runs from that same final grid. Interpret these as endpoint projections, per the user, not direct causal dropout penalties. R9 retraining evidence still requires a protocol-compatibility audit.
- Later stationarity pilots support only limited conclusions: dropout 0 and .3 passed practical tolerances in one seed; .5 remained unconfirmed. Stability is not stationarity, and stationarity is not evidence of positive ridge alignment.
- The final study is `dropout_loss_plateau_absolute`, not the older relative-loss study: 480/480 results complete; final status and checkpoint audit are in its own directory. The old study status must not be used to describe the final grid.
- Compute/workflow follow-up completed in manuscript commit `26bdd7f`: final 480-model study and SLURM resources documented; resources checked against job accounting. Main workflow now defaults to the absolute-loss protocol, connects source restoration, MNIST download, pilot gate, 480 runs, storage receipts, node-local audit, both figures, staging, and LaTeX dependencies. Parent workflow changes remain uncommitted.
- R9 uses older retraining code whose defaults are a 500-epoch cap and validation-loss stopping. Do not present those outputs as matched validation of the new loss-plateau endpoints without checking their executed configurations and target provenance.

## Completed changes

- **R1 — complete (2026-11-03), manuscript commit `116f1e5`, pushed to `origin/main`.** Changed only Figure 3 caption wording in the main text; matched the wording in appendix experiment details and added one sentence in the bootstrap appendix. Validation: `git diff --check` passed; full `latexmk` PDF build succeeded (33 total pages including appendices/checklist). Output: `output/manuscript-review-check/main.pdf`. Parent submodule pointer not committed.

- **R2–R4 — complete (2026-11-03), manuscript commit `e55a5b5`, pushed to `origin/main`.** Added approximately 500 words in `paper/app_scaling.tex`, following the rebuttals, including the proposed interaction paragraph and architectural-transfer discussion. Appended after existing appendices to preserve their labels. Main-text change: one pointer sentence in Section 2.2. `git diff --check` and full `latexmk` build passed; final LaTeX log has no undefined references. PDF is 34 total pages versus 33 before this addition. Parent submodule pointer not committed.

- **R5 — complete (2026-11-03), manuscript commit `492cbdd`, pushed to `origin/main`.** Added the diagnostic equation and brief interpretation before the two recovery criteria in Section 2.1; added practical guidance in `paper/app_normalization.tex` (Appendix G.2). Full `latexmk` build and diff check passed; final log has no undefined references. Main content now occupies 10 pages before references; total PDF is 36 pages including references/appendices/checklist. Numerical rebuttal claim (>0.9 for elastic net) was not imported without final-data verification. Parent submodule pointer not committed.

## Change log

- 2026-11-03: Created tracker from local audit. Deferred automated-feedback items at user request. Recommended R1–R5 and R7 as manuscript edits that do not require new experiments; R6 needs an additional protocol check.
- 2026-11-03: Recorded user's minimal-main-text preference and supplied camera-ready scope/page/disclosure guidance. No manuscript changes made.

- 2026-11-03: Completed R1 after refreshing manuscript from origin/main. Three manuscript files changed; only caption wording changed in main text. Commit `116f1e5` pushed successfully.

- 2026-11-03: Completed R2–R4 with compact scaling appendix and one main-text pointer; committed/pushed as `e55a5b5`.

- 2026-11-03: Addressed PDF Comment 1 on R3 in commit `b5222bf` (pushed). Replaced the vague skips/pooling sentence with a residual-block example: remove the parameter-free skip, evaluate the modified loss gradient at fixed original weights, fit a cancelling penalty, and separately validate retraining behavior. Appendix-only edit; main text unchanged. Full PDF build passed, 34 total pages, no undefined references in final log.

- 2026-11-03: Revised R3 per follow-up discussion, commit `1a83966` (pushed). Supersedes the residual-block estimation prescription in `b5222bf`: shared weight coordinates now motivate future controlled architectural-transfer studies without prescribing a no-skip gradient target. Omitted the qualifying sentence beginning “However, removing the mechanism” as requested. Appendix-only change; full PDF build and diff check passed (34 pages).

- 2026-11-03: Completed R5 as approved. Main-text additions now bring main content to the supplied 10-page limit; account for pagination before further additions.

- 2026-11-03: Refined R5 in `7e8e92a` (pushed): moved the diagnostic to the end of the evaluation discussion, after both validation modes and their intuition. Replaced “loss-gradient energy” with a regression R-squared explanation and “proportion of the squared loss-gradient magnitude explained”; clarified the uncentered zero baseline. PDF build and diff check passed; 36 total pages.

- 2026-11-03: R6 appendix table explicitly declined as optional. For R8, user selected the final dropout analysis as baseline. Current running study is `dropout_loss_plateau_absolute` (array 111699), not the older relative-loss study. Frozen runner saves initial/every-1000-epoch/stopping checkpoints, optimizer/RNG/stopping state, and ridge diagnostics. Intermediates reside on node9 scratch; home result JSON retains trajectory diagnostics and provenance. Exact 0.5T/0.8T checkpoints generally require replay from preceding saves and validation of replay; existing saved times support a normalized-time curve directly. No jobs or running sources changed.

- 2026-11-03: R8 replacement figure prepared from all ten completed depth-3/width-256/dropout-.3 absolute-loss runs: `results/figures/dropout_checkpoint_stability_loss_plateau_absolute.pdf`. Actual epochs on x-axis, signed ridge coefficient on logarithmic y-axis; 98 regular checkpoints use circles and 10 stopping endpoints use x markers. Initialization and pilot extensions excluded. No normalization by final epoch/coefficient. Common protocol/source and complete checkpoint schedules checked; rendering visually inspected. Source hashes and values saved in matching CSV under `results/data/`. New plot entrypoint/rule preserves the historical Lightning figure. Manuscript integration remains open.

- 2026-11-03: R8 integrated and pushed in manuscript commit `d4956d2`. New Appendix G.3, “Sensitivity to training duration and convergence,” and Figure 8 share page 24 and reference the main dropout section/figure and Appendix H.4 methods. Main-text change is one pointer after the trajectory-estimation discussion. All seeds use translucent light blue, regular checkpoints circles, stopping endpoints stronger x markers. Snakemake regenerated and staged the PDF; visually checked plot and compiled manuscript successfully (37 pages; no unresolved references). Coefficient stability is explicitly distinguished from stationarity and candidate-family adequacy. Parent submodule pointer remains uncommitted.

- 2026-11-03: Cross-chat reconciliation after reading the completed NeurIPS rebuttals task. Verified both staged dropout PDFs byte-for-byte against current generated figures, all ten trajectory source-result hashes, and each plotted stopping endpoint against current results. The main Figure 5 narrative and Appendix H.4 already reflect the final grid and the user’s endpoint-projection interpretation; no new manuscript edits needed for those passages. Compute appendix remains stale. R9 (failure/retraining) is next substantive commitment; R10 (SWA rebuttal evidence) and R7 (editorial checks) remain open. No manuscript changes made in this audit.

- 2026-11-03: Reproducibility wiring audited with a clean-workspace dry run (512 jobs, no generated results/figures present; immutable source archive and W&B configuration supplied). Existing-workspace dry run reuses all 480 dropout results; it schedules other stale/missing OLS/IGR outputs, which were not executed. All eight manuscript figure references are staged outputs. Replaced unmanaged method PNG reference with workflow-generated PDF. Built `paper/main.pdf` through the isolated Snakemake PDF rule (37 pages); committed/pushed compute appendix, reference fix, and compiled PDF as `26bdd7f`. Frozen-source restoration/idempotence/modified-source rejection checked; pilot gate materialized from the existing successful review. No training launched.

- 2026-11-03: At the user’s request, committed the manuscript workflow, its required dropout analysis and verification code, tests, final figures, and the paper submodule pointer at `26bdd7f` in the parent repository. Earlier notes about these workflow changes and this pointer remaining uncommitted are superseded. Unrelated exploratory analyses and raw data remain outside this commit.
