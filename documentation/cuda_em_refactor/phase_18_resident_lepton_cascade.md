# Phase 18: resident charged-lepton cascade

## Goal

The first charged pipeline executed only one wavefront per host call. Although
selection, transport, vertex reselection and final-state generation were
device-chained, every continuation and child was downloaded and uploaded again
for the next wavefront.

This phase adds the device endpoint queues and a multi-wavefront resident
driver needed to remove that round trip.

## Stable endpoint compaction

Each source lepton owns two deterministic endpoint slots:

```text
slot = 2 * source_input_index + local_child_index
```

The device scatters all terminal outcomes into those slots:

- continuous-loss step -> next-lepton slot 0;
- atmosphere layer boundary -> next-lepton slot 0;
- magnetic step -> next-lepton slot 0;
- vertex no-interaction continuation -> next-lepton slot 0;
- bremsstrahlung LPM suppression -> next-lepton slot 0;
- bremsstrahlung -> surviving lepton slot 0, photon slot 0;
- annihilation -> photon slots 0 and 1;
- discrete ionization -> lepton slots 0 and 1;
- observation/escape -> observation slot;
- particle cut -> no continuation;
- explicit fallback -> no endpoint queue entry.

CUB exclusive scans compact the sparse arrays in source/child order. Atomic
compare-and-swap is used only to detect duplicate ownership of one slot. It is
not used to append particles, so execution timing cannot reorder histories.

The device result exposes:

- `next_leptons`;
- `generated_photons`;
- `observations`.

## Resident double buffering

`CudaEmBackend::runResidentLeptonCascadeForValidation()` alternates the two
preconfigured physical workspaces:

```text
workspace A: current lepton queue
workspace B: complete pipeline + compact next queue
swap(A, B)
repeat
```

The next charged queue never passes through host memory. Workspace growth is
still bounded by the configured CUDA memory fraction and fails explicitly if
the limit is exceeded.

The host receives only information that is currently required outside the
resident charged scheduler:

- generated photons, for the future mixed photon/lepton scheduler;
- explicit CPU fallback events;
- observations;
- lepton transport records needed by CPU CoREAS compatibility;
- compact final-state audit records;
- a lossless charged checkpoint if the caller's wavefront limit is reached.

## Statistics and identity

The resident loop updates the same per-stage counters as the one-wavefront
path:

- interaction selection;
- continuous/geometry/magnetic transport limits;
- Moliere trials and deflections;
- vertex reselection;
- Brems, annihilation and ionization final states;
- LPM suppression;
- observations and explicit fallback.

Secondary history IDs advance by the exact number of accepted final-state
children in every wavefront. Continuations preserve their original history ID
and advance only their step counter according to the process that produced
them.

## Validation

`testGpuLeptonTransport` now performs two new checks.

First, it reconstructs one wavefront's expected endpoint queues entirely on
the host from transport, vertex and final-state records, then compares every
device-compacted particle and observation bit-for-bit.

Second, it runs the same initial lepton batch in two modes:

1. host-driven repeated calls to the one-wavefront pipeline;
2. one resident multi-wavefront call.

It compares:

- lifecycle and wavefront count;
- generated photons;
- remaining checkpoint leptons;
- fallback events;
- observations;
- every transport record;
- every final-state record.

The resident path also demonstrably transfers fewer host-to-device bytes than
the repeated one-wavefront path.

## Production integration boundary

The validation API currently receives only a starting secondary history ID.
Before `PhysicalCudaEmRouter` can use an arbitrary-length resident call, it
must reserve a bounded ID interval from the shared CPU/GPU allocator and the
resident driver must checkpoint before that interval can be exhausted.

That range-aware checkpoint is required to prove that CPU and GPU can never
allocate the same history ID. Until it is implemented, the production router
continues using one-wavefront reservations even though the resident charged
engine itself is available and tested.
