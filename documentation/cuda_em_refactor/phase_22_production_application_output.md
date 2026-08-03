# Phase 22: production application, streaming output and strict failure state

## Scope

This phase connects the physical CUDA electromagnetic backend to
`c8_air_shower` without changing the default scalar backend. It also closes
three production-safety gaps:

1. GPU steps are streamed into the existing longitudinal, energy-loss,
   observation and CPU radio interfaces;
2. complete histories are no longer retained in host memory during a
   production shower;
3. CUDA failures produce an explicit incomplete-shower record and never
   silently switch to scalar PROPOSAL.

This phase is an application/integration milestone. It does **not** claim the
final \(5\times\) performance acceptance: the current photon scheduler still
crosses the host boundary between wavefronts, and detailed physical-kernel
timing is the next optimization target.

## Command-line backend selection

`c8_air_shower` now accepts

```text
--em-backend proposal|cuda
--gpu-device
--gpu-min-batch
--gpu-memory-fraction
--gpu-table-cache
--gpu-table-tolerance
--gpu-deterministic
```

The default remains

```text
--em-backend proposal
```

and continues to instantiate the original `Cascade + PROPOSAL` path. The
transport-identity extension in `setup::HybridStack` is inert on that path.

An explicitly requested CUDA backend fails before transport if

- CUDA support was not compiled;
- the device index is invalid;
- the table path is absent or invalid;
- table format, SHA-256, PROPOSAL version or interpolation tolerance is
  incompatible;
- the table cut differs from `--emcut`;
- the primary energy exceeds the table domain.

There is no implicit CPU fallback for any of those configuration failures.

## CORSIKA output adapter

`corsika/gpu/em/CorsikaOutputSink.hpp` implements the production boundary.

### Step output

For every `EmStepRecord`, the adapter

1. converts device positions in metres to CORSIKA `Point` objects;
2. forwards the segment and particle weight to `LongitudinalWriter`;
3. multiplies deposited energy by particle weight exactly once;
4. forwards nonzero deposition to `EnergyLossWriter`.

Photoelectric binding-energy deposition is attached to its photon transport
step before the step is published.

### Observation output

`ObservationPlane::writeParticle()` is a new narrow writer entry point. It
accepts the particle code, kinetic energy, position, direction, time and
weight. GPU total energy is converted to kinetic energy by subtracting the
CORSIKA particle mass.

Escaped particles remain diagnostic records and are not written as ground
particles.

### Radio compatibility

For each electron/positron segment, the adapter reconstructs

- a minimal pre-step particle facade;
- a `StraightTrajectory`;
- a CORSIKA `Step`.

It then calls the existing CPU CoREAS and ZHS continuous-process interfaces.
No GPU radio summation is introduced in this phase.

### Bounded host memory

`PhysicalCudaEmRouter::setRetainRecords(false)` activates production streaming:

- lepton steps and radio records are sent directly to the sink;
- photon steps are retained only until photoelectric deposition is annotated;
- observations and fallback diagnostics update counters without retaining
  complete per-event vectors.

The default remains `true` for unit tests and debugging APIs.

## Exact-cut CPU final-state fallback

CORSIKA normally maps a requested 0.5 MeV production threshold to the nearest
pre-generated PROPOSAL table below it, currently 0.4 MeV. A GPU table,
however, records the exact cut with which its interaction hash was generated.

Using a 0.4 MeV `SecondariesCalculator` to decode an interaction selected from
a 0.5 MeV table can reject a charged-particle fallback because the calculator
hash is intentionally different.

`InteractionModel` now has a separate cache

```text
(CORSIKA medium hash, projectile code, interaction hash)
    -> exact-cut specified calculator
```

`ProposalCpuFallbackHandler` prepares that calculator using the GPU table cut
before asking for final-state random numbers. The ordinary scalar `calc_`
cache is not replaced, so CPU rates and CPU random-number order remain
unchanged.

`testGpuProposalCpuFallback` constructs two real charged-particle calculators
with distinct cuts and proves that

- the interaction hashes differ;
- the exact-cut GPU record is accepted;
- the specified final state is generated;
- the scalar calculator is still available;
- a deliberately corrupted hash is rejected.

## Provenance and incomplete-output contract

`GpuEmRunOutput` is registered as the `gpu_em` OutputManager participant.

`gpu_em/config.yaml` records

- backend and schema version;
- GPU model, compute capability and total memory;
- CUDA driver and runtime versions;
- deterministic mode and memory/batch configuration;
- table path, format, SHA-256, PROPOSAL/generator versions;
- medium, energy domain, stochastic cut and measured interpolation errors.

`gpu_em/summary.yaml` records per shower

- `complete` and `status`;
- GPU and CPU particle-step counts;
- wavefront and maximum-batch counts;
- GPU final states and produced secondaries;
- generic and specified CPU fallbacks;
- fallbacks grouped by process ID;
- observation, escape and cut counts;
- weighted energy deposition and radio-track counts;
- thinning and implemented-process counters;
- queue overflows, peak device memory and table/workspace bytes;
- host-to-device and device-to-host byte counts;
- currently available kernel/transfer and specified-fallback timing fields.

If CUDA initialization or transport throws, the application

1. writes `complete: false`;
2. stores the exception message as `failure_reason`;
3. safely closes an in-progress shower;
4. closes the output library;
5. returns a nonzero exit status.

## Validation performed

The CUDA Release/RelWithDebInfo application was built in `corsika_venv` and
tested on an NVIDIA GeForce RTX 4060 Laptop GPU.

### CUDA regression

```text
23/23 CUDA tests passed
```

This includes the exact-cut specified fallback and physical hybrid router.

### Successful application smoke test

A 1 GeV photon shower with no thinning completed using the real
PROPOSAL 7.6.2 dry-air smoke table:

```text
GPU particle advances:          1425
CPU generic fallback steps:       85
specified CPU final states:         9
GPU final states:                 203
weighted GPU deposition: 0.9652555 GeV
peak device allocation:       749487 bytes
queue overflows:                    0
```

The total CORSIKA `dEdX` budget was 0.991824 GeV. The remaining energy is not
evidence of a duplicated GPU deposit: low-energy scalar fallback particles can
leave the configured atmosphere in the 50 microtesla field, and the existing
application total includes only `dEdX + ground`.

The smoke table tolerance is 0.1 and is **not** a production physics table.

### Strict failure test

A 101 GeV primary was deliberately run against a table ending at 100 GeV.
The process returned exit status 1 and wrote

```yaml
shower_0:
  complete: false
  status: incomplete
  failure_reason: primary energy exceeds the CUDA EM table energy domain
```

### Scalar default regression

The same application binary completed a 1 GeV photon shower with
`--em-backend proposal`. No CUDA table or device option was required.

## Remaining acceptance work

The following items are intentionally still open:

1. replace the host-crossing photon wavefront with a resident mixed
   photon/electron/positron scheduler;
2. accumulate longitudinal and energy-loss histograms on device, while keeping
   optional detailed records for radio/debugging;
3. measure physical kernel, transfer and generic CPU-fallback time separately;
4. remove the large fraction of capability/low-domain generic fallbacks;
5. generate a \(10^{-3}\) table covering the production UHE energy domain;
6. run the prescribed CPU/GPU shower statistics and \(5\times\) performance
   matrix.

These are required before describing the backend as a completed scientific
production accelerator.
