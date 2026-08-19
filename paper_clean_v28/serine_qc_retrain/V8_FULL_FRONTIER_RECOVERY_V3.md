# V8 full-frontier recovery V3

## Why the completed V2 run could not answer the scientific question

V2 correctly repaired the cyclic-start ProteinMPNN representation, preserved
the source-scoped V7-Ser/V6-non-Ser model, and independently reproduced every
physical methyl position.  Its six-round run still released no 3ZGC sequence.
Both the legacy and V2 manifests identify `3ZGC` as the only missing target.
V3 therefore generates only 3ZGC sequences.  The other 16 targets remain the
hash-pinned 31,500-row baseline and are read only for final integrity, coverage,
source, and position audits.  The two 3WNE and two 3ZGC rows shown as mandatory
length controls are fixed native/historical controls, not regenerated targets.

The failure diagnostic exposed a separate search-space omission.  The legacy
six-round ledgers contain 268,365 unique destination-scored sequences.  V2
retained 2,881 strict methyl hits and 996 baseline anchors in its initial beam,
but used the remaining 265,484 rows only to populate `seen`.  Consequently
those non-strict observations could neither serve as bridge states between the
methyl and receptor-plausibility objectives nor be generated again.  V2 also
ranked its exact base shortlist without a child-base estimate.  Its 0-release
result therefore establishes failure of that fixed restricted search, not
non-existence in the 20^7 sequence space.

## Frozen V3 repair

V3 does not relax a scientific gate and does not fabricate a release.

1. Hash-validate the immutable model, representation, baseline, legacy six
   rounds, and the completed V2 single-gate failure.
2. Make all 268,365 legacy observations and all 159,329 V2 methyl-screen
   observations available again: exact-score rows are reused directly and all
   other rows are eligible for deterministic frontier selection.  Legacy rows
   already present in the exact baseline are represented by that baseline and
   are excluded from the 16,384-row bridge budget.  Preserve every strict row
   separately and select 16,384 non-strict, non-baseline bridge rows for
   destination methyl replay plus exact cyclic-start ProteinMPNN scoring.
3. Fit a deterministic position/adjacent-pair surrogate to existing exact
   cyclic-base observations.  The surrogate has no release authority; it is
   used only to rank which sequence receives an expensive exact base score.
4. Seed the dual-objective beam with baseline anchors, legacy strict rows, the
   exact bridge, and all 24,576 exact cyclic-base rows from completed V2.
5. Run six resumable rounds.  Each round retains the strict methyl frontier,
   the predicted base frontier, their acquisition/Pareto frontier, physical
   argmax-position coverage, and deterministic sequence diversity before exact
   cyclic-base evaluation.  Objective lists are interleaved under explicit
   quotas; no insertion-order slice may silently remove a later objective.
6. Release only a novel real sequence whose rounded methyl probability is
   strictly greater than 0.6, whose exact cyclic-start ProteinMPNN score is at
   or above the unchanged baseline 1st-percentile floor, and whose full and
   independent batch-one replays agree within `2e-6`.
7. Recompute the floor and every release with fresh scorers in the independent
   three-pass final audit.  Do not create a structure handoff or permeability
   input.

The finite V3 budget can still fail honestly.  The only terminal success marker
is:

```text
===== ALL V8 V3 FULL-FRONTIER AUTOMATED GATES PASSED =====
```

`run_v8_autodl_recovery_v3.sh` runs entirely from the installed repository and
does not access GitHub.  It resumes hash-valid bridge/round artifacts after an
interruption.  The default log is
`/root/autodl-tmp/v8_full_frontier_recovery_v3.log`; the final scientific review
bundle is
`paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/v8_cyclic_base_v3_full_frontier_review_bundle.zip`.
The bundle contains every search artifact declared by the final manifest,
including all six methyl screens, all six exact cyclic-base shortlists, the
hash-pinned resume/frontier state, and local copies of the completed V2 failure
manifest and trace.
