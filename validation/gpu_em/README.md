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
  --antenna-file ~/corsika-data/antennas.txt \
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

When the scalar sample was already generated elsewhere, `--skip-proposal-run`
launches only the new CUDA arm and requires one or more
`--additional-proposal` roots. Separately compiled original-CORSIKA binaries
remain rejected unless `--allow-mixed-proposal-builds` is explicit. In that
mode all per-source provenance fingerprints are retained, physics-bearing CLI
tokens must still agree, and machine-specific antenna paths are replaced by a
SHA-256 of the observer locations and sampling grids actually written to the
CoREAS and ZHS output. CUDA executable identity remains strict by default,
and rate-table identity is always strict.

For an analysis-only pool whose build strata have already passed an explicit
compatibility study, `compare_ensembles.py --allow-mixed-cuda-builds` applies
the same fail-closed rule to CUDA sources: canonical physics must match and
every executable/table provenance fingerprint remains recorded. This option
never weakens the default and must not substitute for a build-stratum test.

Two recovery/audit exceptions are explicit rather than automatic:

- `--resume-completed-cuda` (together with
  `--resume-completed-proposal`) verifies the exact stored command, seed and
  event count before reusing a closed CUDA directory after an interrupted
  post-processing stage.
- `--allow-mixed-cuda-builds` permits separately built CUDA executables only
  when the complete rate-table SHA-256 and canonical physics configuration
  remain identical. Every executable fingerprint is retained, and
  `compare_cuda_build_strata.py` should be run before treating the pooled
  sample as one physics ensemble.

When two builds have deliberately reused the same seed range, use the stricter
fixed-seed audit instead of an ensemble compatibility test:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/compare_fixed_seed_cuda_outputs.py \
  --reference /path/to/reference/batch_000 \
  --candidate /path/to/candidate/batch_000 \
  --output /path/to/fixed_seed_build_audit
```

The tool normalizes only the executable and output paths, requires every other
command/configuration field and the rate-table hash to agree, and streams the
longitudinal, production, energy-loss, interaction, ground-particle, CoREAS and
ZHS Parquet files in bounded batches. Primary, first-interaction, ground,
energy-deposition and radio summary values are checked as well. It hashes the
exact logical numeric values, including floating-point bit patterns,
independently of Parquet compression. This is the appropriate regression test
for a non-physics change such as the CUDA-parent FLUKA timer fix; failed builds
must not be pooled merely because their small samples look statistically
compatible.

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
  --manifest /path/to/external_pooled_manifest.json \
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

`--manifest` is optional. It lets a read-only CPU shard directory be combined
with explicitly listed CUDA roots without copying a `run_manifest.json` into
the simulation output. `analyze_shower_feature_distributions.py`,
`plot_single_event_runtime_histograms.py`, and the radial tool accept the same
external-manifest pattern.

The robust width filter treats a log-MAD at floating-point roundoff scale as
zero before applying its factor gate. This prevents a discretized fitter bin
from turning neighbouring physical width bins into enormous robust-z outliers
when the ensemble becomes large.

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
  --manifest /path/to/external_pooled_manifest.json \
  --output /path/to/geomagnetic_radial_validation \
  --minimum-r-perp-m 1 \
  --maximum-r-perp-m 600 \
  --bootstrap-repetitions 20000
```

The horizontal coordinate is the perpendicular shower-axis distance used by
`pulse_analysis_modular`, not the horizontal ground radius:

\[
r_\perp=\left\|\mathbf d-(\mathbf d\!\cdot\!\hat{\mathbf n})
\hat{\mathbf n}\right\|,
\]

where \(\mathbf d=(x-x_{\rm core},y-y_{\rm core},0)\) is the antenna
displacement in local NWU coordinates and \(\hat{\mathbf n}\) is constructed
from the configured zenith and azimuth. The exported per-antenna CSV retains
both `r_perp_m` and `ground_radius_m`; `radius_m` remains only as a backward-
compatible internal alias of `r_perp_m`. A legacy extraction containing only
ground radius is rejected by `--reuse-extracted` and must be regenerated.

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
reapplies the current robust width filter, rebuilds the per-shower aggregates,
and regenerates statistics and figures from the existing per-antenna CSV
without reading waveforms again.

## Completion-conditioned milestone diagnostics

`watch_interim_campaign_milestones.py` can repeat the complete shower,
post-Xmax, CoREAS/ZHS, radial, and runtime plotting chain while a distributed
campaign is still running. It only consumes closed CPU sources and CUDA
batches that the independent production watcher has already strictly audited.
For example:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/watch_interim_campaign_milestones.py \
  --distributed-status /path/to/distributed_campaign_status.json \
  --automatic-status /path/to/automatic_completion_watcher_status.json \
  --local-campaign /path/to/cuda_shard_a \
  --local-campaign /path/to/cuda_shard_b \
  --final-root /path/to/final_campaign_root \
  --status-json /path/to/interim_milestone_watcher_status.json \
  --pulse-analysis-root /path/to/pulse_analysis_modular \
  --milestone 50 --milestone 100 --milestone 250
```

These products are deliberately not acceptance evidence. Until the remote
partition is terminal, the CPU subset is conditioned on which jobs happened
to finish first. Every generated `interim_manifest.json` records the exact
source roots and seed sets, carries this selection warning, and marks the
sample ineligible for final acceptance. The watcher never controls simulation
processes and is independent of the exact-seed 500+500 finalizer.

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
FLUPRO=~/fluka \
conda run -n corsika_venv \
  python validation/gpu_em/run_final_test_acceptance.py \
  --source-root ~/corsika-21cma-cuda/corsika8_gpu_refactor \
  --build-root ~/corsika-21cma-cuda/corsika8_gpu_refactor_build_cuda \
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

## Post-Xmax electromagnetic profile analysis

`analyze_post_xmax_em_profiles.py` removes the artificial broadening caused
by averaging independent showers at fixed atmospheric depth. It locates the
quadratic (e^-+e^+) maximum of each shower, interpolates all EM components
onto a common \(\Delta X=X-X_{\max}^{e^\pm}\) grid, and writes absolute,
per-shower-normalized and CUDA/CPU-ratio figures:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_post_xmax_em_profiles.py \
  --ensemble-root /path/to/completed_cpu_cuda_ensemble \
  --output-dir /path/to/post_xmax_em_profile_analysis \
  --resamples 10000
```

For a pooled analysis assembled from several immutable campaigns, keep the
simulation `run_manifest.json` files unchanged and pass a separate manifest:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/analyze_post_xmax_em_profiles.py \
  --ensemble-root /path/to/cpu_profile_shards \
  --manifest /path/to/final_analysis_manifest.json \
  --output-dir /path/to/post_xmax_em_profile_analysis_500
```

The separate JSON uses the same `additional_sources.proposal` and
`additional_sources.cuda` lists understood by the normal validation
manifest. This is useful when an acceptance run spans more than one verified
executable hash: provenance stays immutable while the pooled source list is
explicit.

The longitudinal writer has a common depth grid whose bins beyond the fixed
observation surface are zero-filled. The analysis detects the last observed
bin per shower and converts later bins to missing values. Ratio curves and
the common tail integral require 90% coverage by default; every point in the
CSV retains the actual CPU and CUDA shower counts. Bootstrap resampling uses
complete showers, not longitudinal bins. The normalized curves divide every
shower by its own value at \(\Delta X=0\), separating post-maximum shape from
normalization and (X_{\max})-distribution effects.

## Resumable local CUDA ensemble supplements

`run_local_cuda_ensemble.py` extends existing completed CUDA ensembles to a
target count using sequential fixed-size batches. It hashes the executable,
rate table, original reference executable, FLUKA library, antenna file and
physics runner before starting. A completed batch is reused only after its
event count, seed, output set and CUDA provenance have been checked; an
incomplete existing batch or a mid-campaign hash change stops the run.

The following pattern counts two existing 50-shower outputs and adds sixteen
25-shower batches to reach 500 total showers:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/run_local_cuda_ensemble.py \
  --executable ../corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --proposal-executable ../corsika-21cma/corsika-build/applications/c8_air_shower \
  --table ../corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v10_muons_1e-3_1EeV.c8emrt \
  --physics-runner validation/gpu_em/run_physics_acceptance.py \
  --output-root /path/to/local_cuda500 \
  --existing-cuda /path/to/first_cuda50 \
  --existing-cuda /path/to/second_cuda50 \
  --reference-proposal-root /path/to/one_event_cpu_shards \
  --antenna-file /path/to/antennas.txt \
  --flupro /path/to/fluka \
  --target-events 500 \
  --batch-events 25 \
  --cuda-seed-start 10200101
```

Each batch is a normal `run_physics_acceptance.py` result with its own exact
command, provenance and CPU-reference diagnostic. `campaign_manifest.json`
records the immutable configuration, completed batches, runtime and running
total. Re-running the same command resumes only fully closed batches; it
never overwrites a partial batch automatically.

The runner no longer fixes the original 100 TeV vertical configuration in
its source. The primary energy and PDG ID, zenith/azimuth, IGRF coefficient
model and epoch, EM cut and thinning,
automatic or explicit maximum weight, hadron/muon/tau cuts, core, ring, GPU
batch/memory/table controls and FLUKA-process scheduling controls are command
line options. All values are copied into the immutable campaign manifest and
the exact child command. `--existing-cuda` is optional, so a new campaign can
start from zero. If CPU references were linked on another machine, use
`--allow-mixed-proposal-builds`: executable hashes remain in provenance while
the later ensemble comparison still requires an identical canonical physics
configuration and observer layout.

For long GPU campaigns whose remote CPU references are not ready yet, select
`--defer-reference-comparison`. This runs the same CUDA executable directly
in resumable batches, writes normal CORSIKA output plus schema-1 CUDA
provenance for every closed batch (including executable, table, antenna and
FLUKA-library hashes), and records `comparison_status: deferred` in the
campaign manifest. It does not weaken the final comparison; it moves that
comparison to the later pooled-analysis step. For example, the 100 PeV
inclined proton campaign uses:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/run_local_cuda_ensemble.py \
  --executable ../corsika8_gpu_refactor_build_cuda/applications/c8_air_shower \
  --proposal-executable ../corsika-21cma/corsika-build/applications/c8_air_shower \
  --table ../corsika8_gpu_refactor_build_cuda/gpu_em_tables/production_v10_muons_1e-3_1EeV.c8emrt \
  --physics-runner validation/gpu_em/run_physics_acceptance.py \
  --output-root /path/to/proton_100PeV_theta47_phi180_cuda500 \
  --defer-reference-comparison \
  --antenna-file /path/to/antennas.txt \
  --flupro /path/to/fluka \
  --target-events 500 --batch-events 5 --cuda-seed-start 10400001 \
  --primary-pdg 2212 --energy-gev 1e8 \
  --zenith-deg 47 --azimuth-deg 180 \
  --geomagnetic-model IGRF13 --geomagnetic-year 2025 \
  --em-cut-gev 0.0005 --em-thinning 1e-4 --maximum-weight 0 \
  --had-cut-gev 0.3 --mu-cut-gev 0.3 --tau-cut-gev 0.3 \
  --gpu-min-batch 4096 --gpu-memory-fraction 0.70 \
  --cuda-hadronic-workers 4 --cuda-hadronic-min-batch 64 \
  --cuda-hadronic-target-batch-ms 5 --cuda-hadronic-max-batch 256
```

With `--maximum-weight 0`, the child command deliberately omits
`--max-weight` and preserves `c8_air_shower`'s automatic Kobal maximum
weight. A direct-production batch that exits before writing all summaries and
provenance is not silently reused; inspect or quarantine that batch before
resuming.

When CPU-heavy hadronic phases leave one GPU process intermittently idle,
separate the seed interval into two independent runner roots and reduce the
memory fraction of each process. For example, two 250-event processes can use
disjoint ranges `10400001--10400250` and `10400251--10400500`, each with
`--gpu-memory-fraction 0.35`. Run them from different current working
directories because FLUKA creates `fort.11` and `.timer.out` there. Pool the
outputs only after both provenance records show the same executable, table,
physics, antenna layout and no fatal overflow. This is an ensemble-throughput
optimization, not a single-shower benchmark.

## Exact remote CPU failure reruns

`run_remote_cpu_ensemble.py` normally assigns the contiguous schedule
`seed-start + index`. If a production shard fails because of an output-layer
bug, replacing it with an arbitrary fresh seed can censor showers correlated
with that failure. After fixing the output code, write the exact failed seeds
to a UTF-8 file and rerun them with `--seed-list-file`:

```text
# failed_seeds.txt
10300030
10300040
10300047
10300067
```

```bash
python validation/gpu_em/run_remote_cpu_ensemble.py \
  --executable /path/to/fixed_original_c8_air_shower \
  --output-root /path/to/exact_failed_seed_reruns \
  --antenna-file /path/to/antennas_nwu.txt \
  --seed-list-file failed_seeds.txt \
  --jobs 4 --cpu-list 120-123 \
  --energy-gev 1e8 --primary-pdg 2212 \
  --zenith-deg 47 --azimuth-deg 180 \
  --em-cut-gev 0.0005 --em-thinning 1e-4 \
  --had-cut-gev 0.3 --mu-cut-gev 0.3 --tau-cut-gev 0.3
```

The list replaces `--events` and `--seed-start`; blank lines and `#` comments
are ignored. Empty, duplicate, negative or malformed seeds are rejected, and
the seed-list content hash is stored in immutable provenance.

## Distributed campaign staging monitor

`monitor_distributed_campaign.py` reads only the `completed_indices` declared
by one or more remote CPU manifests. It incrementally `rsync`s those closed
shards, then checks the seed, scalar executable SHA-256, runner provenance,
closed shower timing and required profile/particle/radio files. Incomplete
remote directories are never selected merely because they exist. It also
records the completed/target counts of local CUDA campaign roots in one atomic
status JSON.

```bash
python validation/gpu_em/monitor_distributed_campaign.py \
  --remote-host SERVER_ALIAS \
  --ssh-control-path /tmp/c8-server.sock \
  --remote-campaign preflight=/absolute/remote/preflight \
  --remote-campaign main=/absolute/remote/main \
  --local-campaign /absolute/local/cuda-shard-a \
  --local-campaign /absolute/local/cuda-shard-b \
  --staging-root ~/corsika-data/cpu-staging \
  --status-json ~/corsika-data/final/distributed_status.json \
  --poll-seconds 60
```

Use `--once` for a single validation/synchronization pass. Continuous mode
limits its sleep interval to at most 60 seconds, overwrites the atomic status
snapshot, and prints only a compact heartbeat; it does not rebuild binaries,
rerun failed seeds or start final statistical analysis automatically.

## Fail-closed distributed campaign finalization

`finalize_distributed_campaign.py` is the single entry point used after all
remote CPU shards, exact failed-seed reruns and local CUDA batches have been
staged.  Its default mode is a read-only readiness audit.  Before any plot is
made it requires the exact requested CPU and CUDA seed intervals, closed
per-shower timing records, zero fatal CUDA integrity counters, complete
profile/particle/CoREAS/ZHS files, matching antenna contents and observer
layouts, and one canonical physics configuration.  Independent scalar builds
from the output-writer repair are retained as explicit provenance strata;
CUDA executable and table hashes may not vary.

Use `--maximum-weight VALUE` to state the thinning contract explicitly. A
positive value requires every source command to contain the same
`--max-weight`; zero requires the option to be absent and therefore audits the
application's automatic Kobal value. If the run command omitted the
geomagnetic CLI options because they were application defaults, the
finalizer verifies IGRF model and epoch against `gpu_em/config.yaml` and adds
the same documented implicit values to both canonical configurations. It
never edits the original `config.yaml` to manufacture agreement.

For the 100 PeV, 47-degree campaign, first run without `--execute`:

```bash
conda run -n corsika_venv \
  python validation/gpu_em/finalize_distributed_campaign.py \
  --final-root ~/corsika-data/corsika_validation_results/final_inclined_proton_100PeV_theta47_phi180_emthin1e-4_cpu500_cuda500_v1 \
  --proposal-root ~/corsika-data/corsika_validation_results/remote_inclined_proton_100PeV_theta47_phi180_emthin1e-4_cpu500_staging_v1 \
  --cuda-root ~/corsika-data/corsika_validation_results/local_proton_100PeV_theta47_phi180_emthin1e-4_cuda250_fullaccel_shardA_igrf13_2025 \
  --cuda-root ~/corsika-data/corsika_validation_results/local_proton_100PeV_theta47_phi180_emthin1e-4_cuda250_fullaccel_shardB_igrf13_2025 \
  --expected-events 500 \
  --proposal-seed-start 10300001 --cuda-seed-start 10400001 \
  --pulse-analysis-root /path/to/pulse_analysis_modular \
  --antenna-sha256 238a481851b4d39e9fcc18ed5afefd5ea90a806e235ad7aa6e3bddae0e138668
```

The audit writes `finalization_readiness.json` atomically and exits before
analysis if a seed, file or configuration is missing.  Once its status is
`audit_complete`, repeat the same command with `--execute`.  The finalizer
then creates an immutable pooled source manifest and runs, in order:

- the 500-versus-500 ensemble comparison;
- scalar-feature and all-component longitudinal plots;
- per-shower-aligned post-\(X_{\max}\) EM profiles and fixed-depth diagnosis;
- CoREAS/ZHS geomagnetic amplitude and pulse-width distributions using
  `pulse_analysis_modular`;
- amplitude and width versus shower-plane radius; and
- CPU/GPU single-shower runtime histograms.

Every child command and combined log is retained beside the plots.  The
readiness status becomes `complete` only after all seven analysis steps exit
successfully; a statistical discrepancy is still plotted and reported rather
than being hidden by the orchestration layer.
