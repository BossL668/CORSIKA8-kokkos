# Phase 19: range-safe resident HybridCascade routing

## Problem

The resident charged engine can execute many branching wavefronts inside one
backend call. `HybridCascade`, however, shares one history-ID allocator between
the scalar CPU stack and GPU-produced particles. Supplying only a first ID is
not enough: an unbounded resident call could silently consume IDs that the CPU
allocator has not reserved.

## Bounded history interval

The resident API now accepts

```text
[first_secondary_history_id,
 secondary_history_id_limit_exclusive)
```

Before every branching wavefront it checks the worst-case multiplicity of the
currently supported charged final states:

```text
required IDs <= 3 * current_lepton_count
```

If that bound does not fit, the kernel is not launched. The complete current
queue is downloaded as a lossless checkpoint and
`history_range_exhausted=true` is returned.

The result also reports `secondary_history_ids_used`, which is the exact
number of accepted final-state child IDs consumed. Unused IDs in the reserved
interval remain intentionally skipped; they can never be reused by the CPU.

## Router reservation policy

`PhysicalCudaEmRouter` reserves a chunk from the shared stack before entering
the resident charged engine:

```text
minimum = 3 * input_count
chunk   = min(1,048,576, 64 * minimum)
chunk   = max(chunk, minimum)
```

This normally permits many internal wavefronts while capping one reservation
at roughly one million IDs. If branching grows faster than the remaining
range, the checkpoint re-enters the router staging queue and a fresh disjoint
range is reserved on the next scheduler pass.

The production path now uses the resident engine for charged particles.
Generated photons and checkpoint leptons return to the router staging queue;
observations, cuts, continuous deposits, radio-track records and explicit CPU
fallback events preserve their previous ownership paths.

New router diagnostics are:

- `resident_lepton_wavefronts`;
- `history_range_checkpoints`;
- the existing `reserved_history_ids`, now including complete resident
  intervals.

## Validation

The charged transport test reserves exactly `3 * input_count` IDs and confirms
that the deterministic synthetic cascade:

- stops specifically because the range is exhausted;
- reports `completed=false`;
- produces a non-empty lossless checkpoint;
- never assigns a generated photon or checkpoint child an ID at or above the
  exclusive limit;
- never consumes more IDs than were reserved.

The factor was raised from two to three when direct electron/positron pair
production became GPU-capable: one accepted Epair vertex produces the surviving
primary, an electron and a positron. Keeping the bound tied to the maximum
registered final-state multiplicity prevents a newly enabled process from
invalidating the history allocator silently.

Both direct router tests pass after switching from one charged wavefront per
host call to the resident range-safe path.
