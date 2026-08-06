# Phase 96: reusable-profile reset and scalar-PROPOSAL cut parity

## Scope

This phase diagnoses the large differences found in the 500-versus-500 beta2
validation at 1 TeV, zenith 27 degrees, azimuth 180 degrees and
`emthin=1e-4`. Two independent defects were present. They affected the energy
deposit and the muon-parent production profile; they were not statistical
fluctuations and were not plotting-normalization errors.

## 1. Cross-shower muon-energy-loss accumulation

The resident profile allocation contains eight contiguous histograms:

1. photons;
2. electrons;
3. positrons;
4. negative muons;
5. positive muons;
6. muon-parent production vertices;
7. non-muon energy loss;
8. muon energy loss.

`CudaEmBackend::beginShower()` reset only seven histograms. The eighth slot
therefore survived when the same initialized backend was reused for the next
shower. In the affected output, the per-event integrated CUDA deposit rose
almost linearly with shower ordinal (correlation 0.99969), from about 735 GeV
in event 0 to about 16.4 TeV in event 499. The scalar ensemble mean was
690.99 GeV, whereas the corrupted CUDA mean was 8577.47 GeV.

The allocation, reset and download paths now use one shared
`DeviceProfileHistogramCount` and `deviceProfileHistogramBytes()` definition.
The repeated-backend regression also compares the dedicated
`muon_energy_loss_GeV` vector.

## 2. Requested transport cut versus effective PROPOSAL stochastic cut

The scalar PROPOSAL backend does not always build a calculator at the exact
user production threshold. It selects the largest standard cached cut that is
not greater than the threshold. Thus a requested 0.5 MeV cut uses a 0.4 MeV
PROPOSAL stochastic calculator. The former CUDA preparation path generated its
rate table at exactly 0.5 MeV.

This difference is large for discrete muon ionization. Direct interpolation of
otherwise equivalent dry-air tables gives the following ratio of total rates,
`rate(cut=0.5 MeV) / rate(cut=0.4 MeV)`:

| Muon total energy | Rate ratio |
|---:|---:|
| 406 MeV | 0.7775 |
| 500 MeV | 0.7845 |
| 1 GeV | 0.7951 |
| 3 GeV | 0.7998 |
| 10 GeV | 0.8011 |
| 100 GeV | 0.8023 |

The approximately 20% lower table rate explains the observed 14.6% deficit in
the complete muon-parent profile after decay and other competing limits are
included.

The fix introduces `optimized_proposal_energy_cut()` as the shared scalar
contract and changes `c8_air_shower` to validate CUDA tables against that
resolved stochastic cut. `gpu_em_table_prepare` schema 2 now treats
`--em-cut-MeV` as the user transport/production cut, records the resolved
PROPOSAL cut separately, and passes the resolved value to `gpu_em_tablegen`.
The device flat-table view separately stores the electromagnetic transport cut,
so a 0.4 MeV stochastic table still applies the configured 0.5 MeV production
cut to photons, electrons and positrons. CPU selected-process fallbacks are also
constructed with the resolved 0.4 MeV calculator.

## 3. EM thinning eligibility in the shared lepton kernel

The charged-lepton final-state kernel is shared by electrons, positrons and
muons. Its two-child helper previously applied `EMThinning` to all of them,
although scalar `EMThinning` exits immediately unless the projectile is
electromagnetic. This did not cause the present 1 TeV discrepancy because the
0.1 GeV thinning threshold lies below the muon rest energy, but it would bias
higher-energy campaigns whose thinning threshold includes secondary muons.
The device helper now disables EM thinning for muon projectiles, and a
high-threshold regression requires every muon-ionization final state to retain
both children with `NotApplied` thinning metadata.

## 4. Strict handling of non-monotonic PROPOSAL inverse-CDF columns

Generating the corrected 0.4 MeV table exposed a small local non-monotonicity
in PROPOSAL's interpolated bremsstrahlung inverse CDF for the low-abundance
argon component. Smoothing that column or increasing the accepted error would
change the requested physics distribution. Instead, the table retains the
validated process rate and marks only the affected selected-loss column as
`proposal_selected_loss_cpu_fallback`. The GPU still selects the exact process,
target component and loss quantile; the existing CPU fallback then evaluates
the selected loss and final state. This path is explicit, counted and covered
by malformed-marker tests.

The automatic cache lookup now requires the stored generator contract version
to equal the running generator version. Incrementing the contract therefore
forces regeneration rather than accidentally reusing a table with older cut or
fallback semantics.

## Verification status

The host, rate-table, flat-table, automatic-preparation, interaction-selection,
ionization-final-state, lepton-transport and resident-wavefront tests pass after
the changes. A ten-event smoke ensemble using the regenerated table gives
physical, non-accumulating deposited energies between 516 and 822 GeV.

The formal replacement ensemble contains 500 complete CUDA showers and reuses
the same 500 scalar-reference showers as the pre-fix diagnosis. Its results are:

- mean deposited energy: 690.99 GeV scalar and 690.30 GeV CUDA, a -0.10%
  shift with an absolute mean-difference z score of 0.10;
- CUDA deposited-energy/event-index correlation: 0.0104 instead of 0.9997 in
  the corrupted sample;
- integrated all-parent muon-production profile: +1.44% CUDA/scalar shift,
  0.74 standard errors, Welch p=0.458 and two-sample KS p=0.413, instead of the
  pre-fix -14.6% and approximately 8-standard-error deficit;
- integrated electromagnetic tail after individual shower maximum: -0.42%,
  Welch p=0.841 and KS p=0.998;
- every registered longitudinal, muon-parent, ground-energy, ground-radius and
  ground-time curve passes the ensemble curve gate.

With `FLUPRO` explicitly set, the complete CUDA build test suite passes 34/34,
including the FLUKA module test. The old 500-event CUDA sample cannot be
repaired in place and remains useful only as a pre-fix diagnostic.
