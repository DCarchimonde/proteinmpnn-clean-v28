# V8 methyl-first 3ZGC recovery V4

## Scientific question

V3 completed its full fixed budget but released no 3ZGC candidate.  The sole
failed scientific check was `at_least_one_real_3zgc_candidate_is_released`.
V4 is one final, bounded feasibility search.  It does not retrain the model,
change either threshold, or repeat V2/V3 exact work.

## Meaning of the two gates

The methyl gate is the frozen V8 model decision at the physical residue:

```text
round(p_methyl, 8) > 0.60000000
```

The selected residue must be methylatable, the full annotation must contain at
least one lowercase methyl token, and an independent batch-one replay must
reproduce the probability and physical position within `2e-6`.  This is a
model-predicted N-methylation label, not experimental confirmation.

The cyclic-base score is the receptor-conditioned mean natural-residue log
probability.  For a seven-residue cyclic peptide it averages all seven joint
coordinate/sequence physical starts and all seven decoder-order rotations.
Higher (less negative) is better.  The unchanged 3ZGC floor is the nearest-rank
1st percentile of the 996 baseline sequences:

```text
cyclic_base_log_probability_mean >= -2.094945192337036
```

The floor is the tenth-lowest baseline value.  It retains 986/996 baseline
sequences and excludes the worst approximately one percent.

## Fixed V4 budget

- Reuse prior seen sequences: 501,537.
- Reuse prior exact cyclic-base rows: 69,413.
- Deterministically generate a local/crossover/residue-lattice pool.
- Acquisition-only methyl-logit and cyclic-base surrogates rank the pool.
- Exact methyl screen: 24,576 new sequences.
- Exact cyclic-base score: only new strict methyl hits, capped at 2,048.
- Maximum released joint candidates: 200.
- If no joint hit exists: retain at most ten methylated base-near-miss rows for
  explicit advisor review.

Surrogate predictions never authorize release or advisor review.

## Output classes

`released_joint_candidates.csv` contains only rows that pass both hard gates,
novelty, full annotation, and independent batch-one replay.

`methylated_base_near_miss_for_shangge_review.csv` is created only when the
joint release table is empty.  Every row in it must:

1. have `p_methyl > 0.6` under the unchanged frozen model;
2. contain at least one lowercase methyl token in `design_seq`;
3. reproduce the methyl call and physical position in batch-one mode;
4. be exact and forward-cyclic novel;
5. fail only the unchanged cyclic-base floor; and
6. be labelled `REVIEW_ONLY_NOT_FULLY_QUALIFIED` and never as released.

No non-methylated sequence is allowed in either candidate table.

## Outcome semantics

The manifest separates execution from scientific recovery:

- `execution_audit_gate=PASS` means the fixed protocol completed and all
  integrity rules passed.
- `scientific_joint_gate=PASS` means at least one row passed both hard gates.
- `scientific_joint_gate=FAIL` with execution PASS is an honest zero-joint-hit
  result; only the separately labelled methylated near-miss review table may be
  shown to an advisor.

The independent audit reloads the frozen model, recomputes methyl annotation
and cyclic-base score at batch size one, verifies the output class for every
row, and builds a deterministic hash-indexed review ZIP.
