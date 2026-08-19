# V8 cyclic-base recovery V2

## Why V1 released 0/2,881

The 2,881 3ZGC sequences were real destination-GPU methyl strict hits. Their
full cyclic annotation agreed with the search score within `2e-6`, and their
physical methyl probabilities were above the unchanged strict `>0.6` gate.
They were all rejected by the receptor-conditioned ProteinMPNN post-filter.

That post-filter averaged all decoder-order rotations, but kept a single
serialized cyclic-peptide start. ProteinMPNN relative-position features see the
artificial first/last array boundary, so decoder-order balancing alone is not a
complete cyclic representation audit.

This is separate from the earlier Ser provenance issue. V8 still requires the
hash-pinned source-scoped model: canonical shared tensors, V6 non-Ser experts,
and the V7 Ser expert. The V2 manifest records and rechecks those bitwise source
gates explicitly.

## Frozen V2 contract

1. Reconstruct and hash-check all six legacy search rounds.
2. Destination-rescore every legacy strict hit with the frozen methyl model.
3. For the ProteinMPNN score, jointly rotate peptide N/CA/C/O coordinates and
   natural sequence, reset peptide residue indices to `0..L-1`, keep the
   receptor fixed and visible, and average all `L` physical starts × all `L`
   decoder orders.
4. Recompute the baseline 1st-percentile floor with exactly the same scorer.
5. Persist every candidate's physical probability vector, actionable physical
   argmax position/residue, predicted methyl positions, and representation
   min/max/span.
6. If the corrected 2,881-hit re-audit releases at least one real candidate,
   stop before any new search. Otherwise run the fixed six-round, beam-512,
   4,096-offspring, 4,096-cyclic-base-shortlist dual-objective search. It has no
   early success stop and is resumable from hash-checked round artifacts.
7. Release still requires all of: strict rounded methyl probability `>0.6`,
   corrected cyclic-base score at or above the unchanged 1st-percentile floor,
   exact and forward-cyclic novelty, and an independent batch-one replay within
   `2e-6`.
8. Run an independent three-pass final audit with a fresh methyl scorer and a
   fresh cyclic-base scorer. Do not create a Shang-ge structure handoff or a
   permeability input.

The fixed fallback budget can still end in an honest failure if the frozen
model contains no sequence satisfying both hard gates. V2 never converts that
state into formal abstention, lowers the threshold, or fabricates a release.

## Runtime-resume implementation

The max-min Hamming diversity rule is unchanged, including its deterministic
tie order. Its nearest-distance values are cached and updated incrementally;
the production round-one shape completes this CPU-only selection in seconds
instead of recomputing every candidate-to-selected pair after every insertion.

Each conditional-search round now writes a configuration- and context-bound
in-flight checkpoint after the methyl screen and again after the cyclic-base
shortlist. The checkpoint pins the exact beam, seen set, generated sequence
set, provenance map, and artifact SHA256. A restart may reuse an in-flight
artifact only after its hash, complete sequence order, provenance fields,
scores, physical-start vector, and ensemble summary all validate. Completed
rounds continue to use the original full round-state audit.

## AutoDL entry point

`run_v8_autodl_recovery_v2.sh` performs the complete input/GPU preflight and
launches the V2 search, final audit, and compact review bundle under `nohup`.
Its log is `/root/autodl-tmp/v8_cyclic_base_recovery_v2.log` by default.

Success is only the literal terminal marker:

```text
===== ALL V8 V2 AUTOMATED GATES PASSED =====
```

The scientific review bundle is written to:

```text
paper_clean_v28_outputs/serine_qc_source_scoped_hybrid_v8/
v8_cyclic_base_v2_review_bundle.zip
```
