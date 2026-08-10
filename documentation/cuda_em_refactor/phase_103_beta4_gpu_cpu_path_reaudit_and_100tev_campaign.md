# Phase 103: beta4 GPU/CPU path re-audit and 100 TeV campaign gate

Date: 2026-08-09

## Scope

This audit was performed after the phase 97--101 corrections and before
starting the 500-event beta4 CUDA validation campaign.  The reference is the
scalar PROPOSAL process sequence used by `c8_air_shower`; the question is
whether a routed electromagnetic or muon history silently misses, duplicates,
or reorders an observable physics operation.

## Code-path result

No additional production-blocking semantic discrepancy was found.

| Scalar sequence contract | beta4 CUDA treatment | Audit result |
|---|---|---|
| PROPOSAL stochastic rates and selected loss | versioned rate/inverse-CDF table; explicit specified-process CPU completion when a selected final state is not device resident | covered; the CPU path does not resample the process |
| continuous energy loss | device range/inverse-range transport and deposited-energy record | covered |
| Moliere multiple scattering | device implementation of the production `MoliereInterpol` semantics | covered by CPU-oracle tests |
| LPM suppression | density-dependent photon-pair, bremsstrahlung and electron-pair suppression | covered |
| spherical atmosphere and magnetic tracking | five-layer atmosphere, chord grammage, boundary intersection and leapfrog magnetic step | covered after phase 97 |
| observation plane | the same horizontal plane through the configured shower core, not a spherical-radius substitute | covered after phase 98 |
| first-interaction writer | unthinned generation-zero parent and children are captured and replayed into the ordinary writer | covered after phase 99 |
| longitudinal and energy-loss profiles | step records or device-resident fixed-point histograms with the same shower-axis bin definitions | covered |
| muon-production parent profile | device muon vertices are accumulated before thinning using the incoming parent weight | covered |
| CoREAS and ZHS | electron/positron segments are accumulated by the CUDA radio backend; scalar fallback segments are retained by the ordinary CPU processes | covered |
| observation, thinning and ParticleCut ordering | observation and radio see the transported segment before terminal removal; the 10 ms physical-time cut and rest/kinetic-energy accounting match the scalar policy | covered after phase 100 |
| forced primary interaction/decay | resolved before GPU routing and executed once on the scalar stepper | covered after phase 101 |
| custom process compatibility | every continuous, secondaries, interaction, decay, boundary and stack-process contract must be registered before a CUDA shower starts | covered after phase 101 |
| rare or unsupported final states | explicit CPU fallback with process/component/loss identity and recorded reason | covered; no silent omission |

The remaining CPU/CUDA differences are intentional numerical or scheduling
differences rather than missing physics contracts: Philox history-keyed random
streams replace the scalar sequential stream, rate/loss tables have a finite
declared interpolation tolerance, and device histogram accumulation is
fixed-point and deterministic.  Consequently, independent showers are tested
for statistical equivalence; raw event-by-event identity with the scalar
random stream is not claimed.

## Verification

- All 29 GPU/framework/output tests selected by the audit passed.
- `testModules` initially could not initialize FLUKA because the test command
  omitted `FLUPRO`.  Re-running with `FLUPRO=/home/yuhanglu/fluka` and
  `FLUFOR=/usr/bin/gfortran` passed in 208.31 s.  This was an invocation error,
  not a code failure.
- The recent 10 PeV beta4 production records report zero unregistered
  continuous, secondaries, interaction, decay, boundary and stack processes.
  Specified PROPOSAL completions are recorded and drained in batches.

## Accepted CPU reference

The old local IGRF13/2025 NWU data are explicitly excluded.  The accepted
reference is stored on `psrpku2025` at:

```text
/data/yhlu/CorsikaData/corsika_validation_results/
  staging_beta2_fixed_proton_100TeV_vertical_emthin1e-6_igrf14_2027_cpu500_v1/
  cpu500
```

Its checked provenance is:

- 500 complete one-shower shards and 500 distinct seeds;
- executable SHA-256
  `3dfffb0dcc4b4c062e1b96c65bee41806a6c72876b58abb8158256ea48cfa476`;
- proton, 100000 GeV, zenith 0 deg, azimuth 0 deg;
- IGRF14 evaluated at 2027;
- EM cut 0.0005 GeV and EM thinning `1e-6` with the application-derived
  maximum weight;
- hadron, muon and tau cuts of 0.3 GeV;
- CoREAS and ZHS enabled with the NWU antenna file whose SHA-256 is
  `238a481851b4d39e9fcc18ed5afefd5ea90a806e235ad7aa6e3bddae0e138668`.

The beta4 CUDA campaign must reproduce this canonical configuration and use
the contract-0.18 dry-air table with a 105000000 MeV upper bound.  The table's
measured maximum rate and inverse-CDF errors are `4.9906e-4` and `4.3029e-4`,
respectively.
