# CUDA EM validation runners

The production executable and helper-application options are documented in the
[CUDA/FLUKA CLI reference](../../documentation/cuda_em_refactor/cli_reference.md).
This file documents validation workflows and the meaning of their acceptance
outputs.

## Performance acceptance

`run_performance_acceptance.py` runs scalar PROPOSAL and CUDA EM through the
same `c8_air_shower` binary. It fixes common numerical-library thread counts
to one, keeps the five-layer atmosphere and all explicit rare-process
fallbacks, and uses `--ring 0` by default so the speedup is not attributed to
GPU radio projection.

The primary acceptance configuration is:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/run_performance_acceptance.py \
  --executable ../corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table ../corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v10_muons_1e-3_1EeV.c8emrt \
  --output-root /tmp/c8_gpu_em_performance_acceptance \
  --energy-gev 1000000 \
  --events 1 \
  --repetitions 5 \
  --seed 24027 \
  --em-thinning 1e-4 \
  --maximum-weight 100 \
  --mu-cut-gev 0.3 \
  --cache-mode warm \
  --require-release-build \
  --minimum-speedup 5
```

The runner refuses to overwrite an existing output root. It records:

- the exact CPU and CUDA commands;
- the single-thread environment;
- external process wall time;
- backend-neutral per-shower time from `simulation_timing/summary.yaml`;
- cache-mode provenance and per-CUDA-run cache state;
- executable, immutable rate-table and runner SHA-256 identities;
- every process-level repetition and its execution order;
- median CPU and CUDA external and summed-shower timings;
- ratio-of-medians and paired-run speedup distributions.

The default is five independent process-level repetitions. Even repetitions
run scalar PROPOSAL first and odd repetitions run CUDA first, removing a fixed
thermal or filesystem-cache ordering advantage. The hard gate is

```text
median(CPU summed per-shower time) / median(CUDA summed per-shower time)
```

and not the fastest observed pair. `--events` controls the number of showers
inside each process; `--repetitions` controls how many fresh processes are
measured.

For a formal result, pass `--require-release-build`. The runner locates the
nearest `CMakeCache.txt` above the executable, hard-rejects any build type
other than `Release`, and records the C++/CUDA compilers and CUDA architecture
alongside the timing samples.

`--cache-mode warm` is the default production benchmark. For a true cold
benchmark, copy the immutable rate table to a fresh directory with no adjacent
`.moliere-initial-v1.c8cache`, then use `--cache-mode cold-first` or
`--cache-mode cold-each`. Cold modes refuse to remove a cache that existed
before the benchmark. `cold-each` records that every CUDA repetition started
without the derived cache and regenerated it. The legacy
`--skip-cache-warmup` option is retained as an alias for `--cache-mode as-is`;
it is not proof of a cold start.

The result is written to `benchmark_summary.json`. A speedup threshold failure
returns exit code 2; a command, cache or output error returns exit code 1.
The executable and rate table are hashed again after the last repetition; if
either changed while timing was in progress, no valid summary is produced.

This runner is a performance acceptance, not a physics-distribution
acceptance. CPU and CUDA showers use different random-number implementations,
so same-seed file equality is neither required nor sufficient. Ensemble
validation of \(X_{\max}\), profiles, ground particles and radio remains a
separate required workflow.

## Legacy-to-CUDA replay and radio statistics

### Exact single-event decision replay

`--cuda-replay-tape-out` records every scalar transport segment together
with the exact flat-atmosphere radio/observer snapshot.  The reference
executable remains scalar-only; recording does not link CUDA or change its
LIFO/PROPOSAL decisions.

`cuda_decision_replay` uploads the complete tape, verifies a byte hash and
sanity flags for every record on the GPU, then projects the recorded
electron/positron segments with the CUDA CoREAS and ZHS kernels:

```bash
legacy/c8_air_shower ... \
  --cuda-replay-tape-out /tmp/event.c8rpt \
  -f /tmp/cpu_with_tape

cuda_decision_replay \
  --tape /tmp/event.c8rpt \
  --output /tmp/cuda_replay

python validation/gpu_em/compare_exact_cuda_replay.py \
  --untaped-reference /tmp/cpu_without_tape \
  --reference /tmp/cpu_with_tape \
  --replay /tmp/cuda_replay \
  --report /tmp/exact_replay_comparison.json \
  --require-pass
```

This proves that CUDA can consume and project one *specified* legacy event.
It deliberately bypasses CUDA random sampling and is therefore a debugging
oracle, not evidence that equal seeds identify equal independently sampled
CPU/CUDA showers.  Production `--em-backend cuda` still requires ensemble
physics validation.

`run_cuda_replay.py` uses a three-arm design:

```text
legacy c8_air_shower + scalar PROPOSAL
refactored c8_air_shower + scalar PROPOSAL
refactored c8_air_shower + CUDA EM
```

All arms use IGRF13 at 2025, the 21CMA observation altitude, the same seed,
and the same external NWU antenna file. The scalar control is checked for an
identical ordered PROPOSAL process/component sequence. The CUDA comparison is
an ensemble comparison and never claims that one same-seed event must have an
identical waveform.

The applications expose:

```text
--cuda-replay-trace process_trace.csv
```

and the runner compares process counts/energy/loss/altitude distributions,
longitudinal profiles, energy deposit, \(X_{\max}\), and CoREAS/ZHS energy
fluence in 30--80 and 50--350 MHz. Radio fluence and the integrated
radiation-energy proxy use a fixed squared primary energy for fixed-energy
electron/positron/photon ensembles and squared per-shower EM deposited energy
for hadron primaries. This avoids turning a rare photonuclear transfer in a
fixed-energy electron sample into an artificial high normalized tail.

Example:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/run_cuda_replay.py \
  --legacy-executable ../corsika-21cma/corsika-build/applications/c8_air_shower \
  --cuda-executable ../corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table ../corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v10_muons_1e-3_1EeV.c8emrt \
  --antenna-file /home/yuhanglu/21CMA/data/antennas.txt \
  --output-root /tmp/c8_cuda_replay_1TeV_20 \
  --energy-gev 1000 \
  --events 20 \
  --seed 260729 \
  --max-deflection-angle 0.2 \
  --skip-scalar-control \
  --disable-process-trace
```

With the default `--minimum-showers 20`, fewer than 20 showers produces
`diagnostic_insufficient_statistics`. This replay is a capture-and-compare
debug mode; it does not yet inject the legacy CPU decisions into CUDA.

For radio-accuracy work, explicitly scan `--max-deflection-angle`. The
application default is 0.2 rad, while the CORSIKA 8 radio validation study
(arXiv:2409.15999) found CoREAS/ZHS convergence at approximately 0.001 rad.
The configured value is passed to both scalar tracking and the resident CUDA
magnetic-step limiter and is stored in `gpu_em/config.yaml`; an older CUDA
build that does not record
`environment.maximum_magnetic_deflection_rad` is not valid evidence for a
non-default scan.

Track length is not the only numerical radio setting. The historical
`c8_air_shower` observer uses 1 GHz sampling (1 ns bins). Although its
500 MHz Nyquist frequency is above 350 MHz, ZHS is written as a finite
difference of the sampled vector potential. Its transfer function therefore
contains a `sinc(pi f dt)` suppression that is already material in the
50--350 MHz band. A formalism-convergence run must use the paper setting,
10 GHz (0.1 ns), as well as small tracks:

```bash
--max-deflection-angle 0.001 \
--radio-sampling-rate-ghz 10 \
--radio-window-duration-ns 1000 \
--radio-pretrigger-ns 100
```

The unmodified legacy executable does not expose the three observer options.
For this numerical-convergence scan, pass the refactored executable as both
`--legacy-executable` and `--cuda-executable`; its default backend remains
scalar PROPOSAL for the reference arm. Keep the actual original executable
as the reference for default-configuration regression runs. The runner
rejects a mixed original/refactored run with non-default observer settings
instead of silently comparing different sampling grids.

After a run, the distributional waveform diagnostic compares independent
shower ensembles rather than raw same-index bins:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_cpu_cuda_radio.py \
  --reference-output /path/to/legacy_proposal \
  --candidate-output /path/to/cuda \
  --report /tmp/waveform_diagnostics.json \
  --csv /tmp/waveform_features.csv \
  --normalization-energy-gev 1000
```

It reports time-domain pulse widths and aligned templates, frequency-spectrum
templates, polarization/Stokes distributions, radial radiation-energy
contribution shapes, track ledgers and paired CoREAS/ZHS radiation energy for
the same underlying tracks.

To quantify whether an earlier small-sample radiation-energy deficit is a
tail fluctuation, compare it against a larger independent repeat:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_radio_bias_attribution.py \
  --signal-reference /path/to/old/legacy_proposal \
  --signal-candidate /path/to/old/cuda \
  --validation-reference /path/to/repeat/legacy_proposal \
  --validation-candidate /path/to/repeat/cuda \
  --report /tmp/radio_bias_attribution.json
```

This reports raw/median/trimmed shifts, tail influence, the probability of
reproducing the old shift by small-sample resampling, cross-channel
correlations, and radio yield per squared weighted track length.

Independent small shards (useful when a long hadronic run is vulnerable to
an external low-energy model abort) can be pooled without treating repeated
shower IDs in different directories as the same event:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/pool_radio_ensembles.py \
  --reference-output /path/to/batch_a/legacy_proposal \
  --reference-output /path/to/batch_b/legacy_proposal \
  --candidate-output /path/to/batch_a/cuda \
  --candidate-output /path/to/batch_b/cuda \
  --report /tmp/pooled_radio.json
```

The pooled report includes raw, EM-deposit-squared and weighted-track-length
squared normalizations, bootstrap intervals, median and 5% trimmed shifts,
KS/quantile-Wasserstein diagnostics, and leave-one-out sensitivity. Only pool
runs after checking that their physics configuration and relevant backend
versions are compatible; this diagnostic intentionally does not override the
stricter provenance guard in `run_physics_acceptance.py`.

## Physics ensemble acceptance

`run_physics_acceptance.py` launches two explicitly independent ensembles:
scalar PROPOSAL and CUDA EM. It then calls the same analysis implemented by
`compare_ensembles.py`.

The initial 1 TeV matrix entry can be run with:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/run_physics_acceptance.py \
  --executable ../corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table ../corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v10_muons_1e-3_1EeV.c8emrt \
  --output-root /tmp/c8_gpu_em_physics_1TeV \
  --label electron_1TeV_zenith0_cut0.5MeV \
  --energy-gev 1000 \
  --events 1000 \
  --proposal-seed 41001 \
  --cuda-seed 51001 \
  --proposal-shards 8 \
  --proposal-parallelism 8 \
  --overlap-backends \
  --gpu-min-batch 64 \
  --zenith-deg 0 \
  --em-cut-gev 0.0005 \
  --em-thinning 1e-4 \
  --maximum-weight 100 \
  --require-pass
```

The two seeds must differ. A single event is not a statistical sample, and
particle rows inside one shower are not treated as independent observations.
With sharding enabled, scalar seeds are `proposal-seed + shard-index`; none
may equal the CUDA seed. Every shard still has all numerical-library thread
counts fixed to one. `--overlap-backends` is permitted only for physics
statistics: CPU contention makes its wall times invalid for speedup claims.

An interrupted CPU-shard campaign can be continued without rerunning completed
showers:

```bash
python validation/gpu_em/run_physics_acceptance.py \
  ...the exact original physics and seed options... \
  --output-root /path/to/unfinished/root \
  --events 50 \
  --proposal-shards 50 \
  --resume-completed-proposal
```

Resume is fail-closed. For each existing shard the runner requires
`summary.yaml` and `config.yaml`, then checks the exact command, seed and event
count before reusing it. An incomplete shard must be moved aside explicitly;
it is never overwritten. Resume is refused if CUDA output, `comparison.json`
or `run_manifest.json` already exists, and it cannot be combined with
`--overlap-backends`. Reused per-shard elapsed values come from
`summary.yaml:runtime_raw`; the manifest records the reused shard indices and
timing source. After all scalar shards are valid, the CUDA ensemble is run
normally and fresh provenance sidecars and final comparison files are written.

The analysis extracts one row per shower containing:

- charged-particle \(X_{\max}\), peak size and longitudinal integrals;
- photon, electron and positron longitudinal integrals;
- energy-deposit sum, fitted \(X_{\max}\) and peak;
- weighted ground photon, electron and positron counts;
- weighted ground kinetic and total energy;
- ground radial mean/RMS and arrival-time RMS;
- deposited, ground and approximate closure energy fractions.

It also compares ensemble-mean curves for:

- charged, photon, electron, positron and total EM longitudinal profiles;
- longitudinal energy deposit;
- normalized ground radial, kinetic-energy and time-residual histograms.

The default acceptance requires every key scalar mean to satisfy both:

```text
absolute relative difference <= 1%
absolute mean difference <= 3 combined standard errors
```

The default core key list is:

```text
charged Xmax
charged peak
charged longitudinal integral
photon longitudinal integral
total deposited energy
deposit-curve Xmax
weighted ground EM count
weighted ground EM kinetic energy
approximate energy-closure fraction
```

Per-shower deposit maxima and sparse ground-tail moments remain fully reported
and statistically tested through their distributions, but they are not in the
default 1% list. UHE or detector-level matrices can replace the defaults by
repeating `--key-scalar`, for example:

```bash
--key-scalar profile_xmax_charged_gcm2 \
--key-scalar energy_deposit_sum_GeV \
--key-scalar ground_em_weighted_count \
--key-scalar ground_em_kinetic_energy_GeV
```

The selected list is always stored in the JSON report and run manifest.

To extend a completed independent ensemble without discarding its showers, add
new events and pool the old roots in the same guarded run:

```bash
python validation/gpu_em/run_physics_acceptance.py \
  ...same physics options... \
  --events 200 \
  --additional-proposal /path/to/old/proposal-shard-0 \
  --additional-proposal /path/to/old/proposal-shard-1 \
  --additional-cuda /path/to/old/cuda
```

The new and existing samples remain separate independent processes. The
manifest records both newly generated and additional sources, as well as new
and combined event counts. Pooling is refused unless the physics-bearing
command tokens agree after removing only seed, event count, output path,
verbosity and CPU/CUDA implementation controls. It is also refused unless
every input has the same SHA-256 identity for the `c8_air_shower` executable;
CUDA shards must additionally have the same complete rate-table SHA-256.
These checks happen before new showers are launched.

Every newly generated scalar or CUDA output contains
`validation_provenance.json`. It records:

- the executable path, byte size, modification time and SHA-256 captured
  before execution;
- the corresponding immutable rate-table identity for CUDA;
- the validation-runner identity;
- the exact command and its SHA-256.

The runner hashes the executable and table again after all subprocesses
finish. If either changed while showers were running, the outputs receive no
provenance sidecar and cannot be pooled. This closes the otherwise dangerous
case where a build path is reused for a physically different binary.

For a distribution, at least 95% of active bins must lie within three combined
standard errors. Its relative L1 difference is reported as a convergence
diagnostic, but the 1% gate is applied to the explicitly listed key scalar
means rather than every bin of a finite independent ensemble. Profile bins
below \(10^{-4}\) of the CPU mean peak are excluded from the shape gate but
remain in the CSV output.

Outputs are:

```text
comparison.json
curve_comparison.csv
per_shower_observables.csv
scalar_stability.json
run_manifest.json
proposal/validation_provenance.json
cuda/validation_provenance.json
proposal.log
cuda.log
proposal/
cuda/
```

`run_manifest.json` contains exact commands, independent seeds, numerical
thread settings, cache provenance and external wall times. `comparison.json`
contains every scalar and curve decision. `--require-pass` returns exit code 2
for a physics-gate failure; simulation, integrity or I/O failures return 1.
The runner also creates the bootstrap stability report described below; use
`--stability-bootstrap-repetitions 0` only when that diagnostic is explicitly
unwanted.

To reanalyse existing outputs without rerunning showers:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/compare_ensembles.py \
  --proposal /path/to/proposal-shard-0 \
  --proposal /path/to/proposal-shard-1 \
  --cuda /path/to/cuda \
  --output /tmp/c8_gpu_em_reanalysis \
  --minimum-events 100 \
  --fail-on-acceptance
```

Both `--proposal` and `--cuda` are repeatable. Shards are concatenated only
along the event axis. Coordinate grids, scalar observable sets and histogram
edges must agree exactly. Their stored primary, energy, angle, cut, thinning,
atmosphere/tracking and interaction command options must also match; changing
only the seed, event count, output path or backend implementation is allowed.
The CPU and CUDA ensembles receive the same cross-backend configuration check.
Their stored executable hashes must also match, and all CUDA shards must have
the same table hash. Any mismatch stops analysis before calculating a mean.

Outputs created before this provenance contract are rejected by default. They
can still be inspected diagnostically with
`compare_ensembles.py --allow-legacy-provenance`, but legacy and provenanced
outputs cannot be mixed and the resulting comparison is not production
evidence.

If a finite ensemble misses the 1% mean gate, quantify whether that difference
is statistically resolved before scheduling a much larger production run:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_scalar_stability.py \
  --per-shower /tmp/c8_gpu_em_reanalysis/per_shower_observables.csv \
  --output /tmp/c8_gpu_em_reanalysis/scalar_stability.json \
  --metric profile_xmax_charged_gcm2 \
  --metric ground_radius_mean_m
```

The report stores the signed relative difference, independent-sample z-score,
bootstrap confidence interval, and the estimated event count per backend
needed for a three-sigma precision band narrower than the requested relative
tolerance. Its `equivalent`, `different`, and `inconclusive` classifications
are diagnostics and do not override the strict gates in
`compare_ensembles.py`.

Before comparing observables, the loader requires every common timing record
to be closed. For CUDA it additionally requires `complete: true`, zero queue
overflow, zero cross-species host spill, zero profile fixed-point overflow and
zero invalid profile records. If a CUDA shower advertises complete strict
energy-ledger coverage, the stored ledger must also be accepted and remain
within its declared tolerance.

## Reusable-backend bitwise equivalence

`verify_reused_backend_equivalence.py` compares paired CUDA showers generated
before and after enabling cross-shower backend reuse. It validates both input
trees as complete CUDA outputs, then requires exact equality of every scalar,
every longitudinal/deposit bin and every derived ground histogram bin:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/verify_reused_backend_equivalence.py \
  --fresh /path/to/fresh-backend/cuda \
  --reused /path/to/reused-backend/cuda \
  --events 100 \
  --output /tmp/backend_reuse_equivalence.json
```

This is a paired deterministic regression, not a CPU/CUDA statistical
comparison. The two CUDA runs must use identical seeds, shower IDs and physics
configuration, executable hash and table hash. Any differing value returns
exit code 2. Historical outputs without a sidecar require the explicit
diagnostic-only `--allow-legacy-provenance` switch.

## Identical-track CoREAS/ZHS backend acceptance

`run_radio_acceptance.py` isolates radio projection from shower fluctuations.
It runs the deterministic CUDA EM transport twice with the same seed:

```text
run A: --em-backend cuda --radio-backend cpu
run B: --em-backend cuda --radio-backend cuda
```

The runner then requires the non-radio transport Parquet tables to be exactly
equal before comparing any waveform. This proves that CPU CoREAS/ZHS and the
resident CUDA implementation consumed the same shower, rather than accepting
two statistically similar but different cascades.

```bash
conda run -n corsika_venv \
  python validation/gpu_em/run_radio_acceptance.py \
  --executable ../corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --table ../corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v10_muons_1e-3_1EeV.c8emrt \
  --output-root /tmp/c8_gpu_radio_acceptance_1TeV \
  --energy-gev 1000 \
  --events 10 \
  --seed 25025 \
  --ring 1 \
  --gpu-min-batch 64 \
  --require-pass
```

For every shower, antenna, algorithm and polarization, it reports:

- maximum absolute and peak-normalized waveform difference;
- relative \(L_2\) waveform difference;
- relative summed-squared-field (fluence proxy) difference;
- whether acceptance came from the relative or absolute weak-signal bound.

Outputs are `radio_comparison.json`, `radio_component_comparison.csv`,
`run_manifest.json`, exact command logs, and both complete CORSIKA output
trees. The standalone `compare_radio_backends.py` can reanalyse an existing
same-seed pair.

## Strict electromagnetic energy closure

`validate_energy_ledger.py` independently recomputes the resident CUDA energy
balance from `gpu_em/summary.yaml`:

```text
initial total energy
+ medium electron rest-mass input
= deposited energy
+ terminated e-/e+ rest mass
+ observed total energy
+ escaped total energy
```

The medium term is required because Compton scattering, photoelectric
absorption, discrete ionization and positron annihilation each import one
atomic-electron rest mass. The ordinary `dEdX + ground` diagnostic omits this
source and also omits the rest mass of a lepton terminated by `ParticleCut`;
it is therefore not a strict conservation test.

Use an unthinned pure-EM shower with `--gpu-min-batch 1` to prevent scalar
front expansion, then validate one or more outputs:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/validate_energy_ledger.py \
  --cuda /path/to/electron-output \
  --cuda /path/to/photon-output \
  --output /tmp/c8_energy_ledger_acceptance.json \
  --tolerance 1e-4 \
  --minimum-events 2 \
  --require-complete \
  --require-pass
```

The validator does not trust the stored residual: it reconstructs source,
terminal, residual and relative error from the primitive terms, checks their
internal arithmetic at floating-point storage precision, and rejects partial
coverage when requested. In the application itself, a pure EM, unthinned
event with no scalar EM step and no CPU fallback is marked incomplete
immediately if its relative closure error exceeds \(10^{-4}\).

The statistical test itself can be checked without a shower run:

```bash
conda run -n corsika_venv \
  python -m unittest discover \
  -s validation/gpu_em/tests -p 'test_*.py'
```

## Geomagnetic pulse-amplitude and pulse-width distributions

`analyze_geomagnetic_pulse_distributions.py` reuses the polarization basis,
main-pulse selection, interpolated max-Q square-window fit, and default robust
width filter from
`python/MCMCTidyUp/pulse_analysis_modular`.  It projects each waveform onto
\(\widehat{\mathbf{v}\times\mathbf{B}}\), then aggregates the azimuth antennas
at one radius so that the statistical unit remains one shower.

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_geomagnetic_pulse_distributions.py \
  --dataset /path/to/original_vs_cuda_radio_ensemble \
  --radius-m 100 \
  --output /path/to/geomagnetic_pulse_comparison \
  --bootstrap-repetitions 20000 \
  --equivalence-relative-tolerance 0.10 \
  --minimum-ks-p-value 0.05 \
  --minimum-count-per-backend 20 \
  --minimum-width-valid-shower-fraction 0.80 \
  --require-pass
```

The outputs are a four-panel CoREAS/ZHS distribution plot, raw per-antenna
features, per-shower aggregates, a JSON statistical report, and a Markdown
summary. The formal gate requires the complete bootstrap 95% CUDA/CPU ratio
interval to lie inside the predeclared equivalence margin, an adequate KS
shape diagnostic, a minimum shower count, and sufficient fitted-width
coverage for both CoREAS and ZHS. It does not accept a broad confidence
interval merely because it contains one, and it never treats
same-integer-seed showers as paired histories.

The four simultaneous KS shape diagnostics (CoREAS/ZHS amplitude and width)
use a Holm-Bonferroni correction at the requested family-wise alpha. This
avoids treating one ordinary \(p<0.05\) fluctuation among four tests as a
backend discrepancy while retaining fail-closed rejection of genuinely small
p-values.

For a pooled full-array radio-energy diagnostic, use
`pool_radio_ensembles.py`. Its formal statistic is the geometric mean of the
positive per-shower \(E_{\rm rad}/E_{\rm dep}^{2}\) proxy, with a bootstrap
ratio interval and the same four-test Holm-Bonferroni shape correction.
Unnormalized arithmetic radiation energy is retained as a diagnostic because
it mixes radio transport with differences in the independently sampled shower
energy. Track-length normalization is excluded from acceptance: scalar
`RadioProcess` and resident CUDA radio record different track-ledger
populations.

The max-Q square-window width is a raw-monopolar-pulse observable. An ideal
rectangular FFT band-pass creates long ringing for which that estimator is not
calibrated. Band-passed width output is therefore diagnostic only, and
`--require-pass` deliberately rejects a command that also supplies
`--band-low-mhz/--band-high-mhz`.

For radial curves from one completed hadronic ensemble, use the dedicated
comparison tool rather than the multi-energy electron scaling-law matrix:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_geomagnetic_radial_comparison.py \
  --dataset /path/to/completed_cpu_cuda_ensemble \
  --output /path/to/geomagnetic_radial_validation \
  --minimum-radius-m 1 \
  --maximum-radius-m 600 \
  --bootstrap-repetitions 20000
```

It plots CoREAS/ZHS amplitude and width against every common nominal antenna
radius, plus separate CUDA/CPU ratio panels. Bands on the main curves are the
16th--84th shower percentiles; ratio bands are pointwise 95% independent
two-sample shower bootstraps. Text-rounding variants of one antenna ring are
clustered within \(10^{-4}\) m and assigned one nominal radius, so diagonal
coordinates cannot be counted twice. Width points with two or fewer distinct
values at \(10^{-6}\) ns resolution are marked `diagnostic_only`: this catches
the max-Q fitter sitting on its minimum window instead of presenting a
spurious ratio of exactly one as physics agreement. The extractor loads one
backend/algorithm stream at a time to bound memory; `--reuse-extracted`
regenerates statistics and figures from its existing CSV files without
reading waveforms again.

## Final fail-closed evidence audit

`audit_final_acceptance.py` turns the final requirement-to-evidence matrix
into one machine-readable verdict. The YAML specification names every
artifact and field used as evidence; a missing file, unresolved field,
non-finite value, unknown operator or failed required check makes the overall
result fail:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/audit_final_acceptance.py \
  --spec validation/gpu_em/final_acceptance.yaml \
  --output /path/to/final_acceptance.json \
  --require-pass
```

Diagnostic requirements remain visible in the report but cannot turn a
failed required gate into a pass. Every evidence artifact, plus the
specification itself, is recorded with its resolved path, byte size and
SHA-256.

The regression-suite evidence is generated rather than copied from terminal
scrollback:

```bash
FLUPRO=/home/yuhanglu/fluka \
conda run -n corsika_venv \
  python validation/gpu_em/run_final_test_acceptance.py \
  --source-root /home/yuhanglu/21CMA/corsika8_gpu_refactor \
  --build-root /home/yuhanglu/21CMA/corsika8_gpu_refactor_build_cuda \
  --output /path/to/final_test_acceptance.json \
  --minimum-ctest-count 32 \
  --minimum-python-count 100 \
  --require-pass
```

This records both commands and complete logs, requires zero return codes, and
fails if the reported CTest or Python test counts cannot be parsed.

## Geomagnetic scaling-law scan

`analyze_geomagnetic_scaling.py` combines independent electron-shower
ensembles to test how the geomagnetic pulse peak and width change with primary
energy, geomagnetic angle and shower-plane radius. The current production
matrix contains 100, 300, 1000 and 3000 GeV vertical showers, three azimuths
at 1 TeV and 45 degrees zenith, and ten radii from 1 to 600 m. Both CPU
PROPOSAL and CUDA EM transport are analysed for CoREAS and ZHS.

The scaling amplitude is not an average of absolute antenna peaks. At each
radius the eight signed \(E_{\mathbf v\times\mathbf B}\) waveforms are aligned
to their selected peaks with zero padding and coherently averaged. This
cancels most of the oppositely signed charge-excess projection. All
confidence intervals then resample complete showers, preserving the
within-shower antenna correlation.

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_geomagnetic_scaling.py \
  --output /path/to/geomagnetic_scaling_laws \
  --bootstrap-repetitions 10000
```

The tool fits:

- \(A\propto E^\alpha\) and \(W\propto E^\gamma\) at 100 m;
- \(A\propto\sin(\alpha_B)^\eta\) at 1 TeV and 100 m;
- separate radial laws over 50--200 m and 200--600 m.

It also reports adjacent energy slopes, bootstrap differences between the
low- and high-energy slopes, CPU/CUDA slope differences, fit \(R^2\), and
coverage gates for pulse widths. A law is marked diagnostic instead of
accepted whenever \(R^2<0.9\), fewer than 80% of showers have a valid width,
or the median valid-antenna fraction is below one half. Outputs include raw
per-antenna and per-shower-radius CSV files, `scaling_laws.json`, three
figures, and a Markdown interpretation.

`analyze_shower_scaling.py` complements that radio analysis with the
longitudinal and observation-level quantities stored by CORSIKA:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_shower_scaling.py \
  --dataset /path/to/100GeV_ensemble \
  --dataset /path/to/300GeV_ensemble \
  --dataset /path/to/1TeV_ensemble \
  --dataset /path/to/3TeV_ensemble \
  --output /path/to/geomagnetic_scaling_laws \
  --bootstrap-repetitions 10000
```

It fits the \(X_{\max}\) elongation rate in g/cm² per energy decade and
power-law exponents for \(N_{\rm charged}(X_{\max})\), charged/photon track
integrals, deposited energy, ground EM weighted count and ground EM kinetic
energy.  In addition to CPU/CUDA exponent differences, the JSON output
retains the CUDA/CPU ensemble-mean ratio at every energy.  This distinction
is important: two backends can agree in slope while retaining a
normalization offset.
