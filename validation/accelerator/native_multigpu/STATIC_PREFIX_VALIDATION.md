# Static CPU-prefix validation

This option parallelizes independent CPU subshowers before the existing GPU
handoff. It does not change cross sections, CDF solvers, cut/thinning policies,
environment transport or radio algorithms. It is disabled by default.

## Regression coverage

- `test_native_multigpu`: serial/four-worker coordination, sixteen CPU workers
  followed by four GPU-worker fixtures, two CPU workers followed by one GPU
  fixture, exact-once output merges, input consumption, failures and timeouts.
  These are C++ fixtures, not GPU performance measurements.
- `test_multigpu_frontier`: preservation of fourteen particle species, nuclear
  identity, kinetic energy, weights, logical environment nodes and transport
  histories; disjoint CPU/GPU history ranges; duplicate detection; bounded
  import without skipping pending work; static import-chunk validation.
- `test_small_stack_digest`: standard SHA256 results for empty/short files,
  either side of the 64 KiB boundary and 1 MiB, on a guarded 64 KiB thread
  stack. It checks the buffer-allocation fix, not a different hash algorithm.

The standalone coordinator tests can be built as described in
[the usage guide](../../../documentation/native_multigpu_CN.md#验证).
`test_multigpu_frontier` is an application build target with
`CORSIKA_BUILD_MULTIGPU_APPLICATION=ON`; run it directly after building.

## Integrated-source checks (2026-10-19)

Fresh CUDA and OpenMP application builds passed, as did the frontier tests in
both builds and both standalone CTest targets. The coordinator fixtures also
cover more CPU workers than roots (empty shards), incorrect consumption
counts, unsupported parallel-prefix EGS4, namespace exhaustion and the
single-GPU public CLI. Failed/incomplete workers must not publish `COMPLETE`.

Local physics regression used a 10 TeV proton, seed `26101991`, zenith 47
degrees, azimuth 180 degrees, 81 antennas, a 0.2 rad deflection limit and one
RTX 4060 Laptop GPU with a 70% memory budget. The CPU split and EM handoff
caps were both 312.5 GeV. The unchanged stopped-positron photon default was
off. Five full GPU runs and two CPU-prefix captures checked:

- Old versus integrated serial-prefix capture: byte-identical frontier and
  identical numerical tables.
- Old versus integrated ordinary single-GPU shower (no parallel-prefix
  option): identical numerical tables.
- Validated candidate versus integrated two-CPU-prefix/one-GPU shower,
  with import chunk 32768: byte-identical CPU and EM frontiers and identical
  merged numerical tables.
- Integrated four-CPU-prefix/one-GPU shower with the environment override
  unset: configured import chunk 4096, complete input consumption and valid
  output merging.

The seven tables compared were longitudinal profile, production profile,
energy deposition, ground particles, first interaction and CoREAS/ZHS
observers. Numerical equality ignores Parquet metadata, not any columns or
rows. Independent checks confirmed exact-once CPU/GPU root partitioning,
complete consumption, compensated profile/radio sums, preservation of the
first interaction, finite outputs and balanced CPU fallback accounting.
Queue/profile/radio overflow, native-CDF failure, generic fallback and
rejection-sampling counters were zero in the parallel-prefix runs.

These checks establish integration consistency on the tested inputs; they
are not a new large-sample acceptance or a timing benchmark. The candidate
binary used for the two-CPU comparison had SHA256
`6d822cb377abab1472743261138d57942ac42a43dabe269790d3929893e37e2a`.

## Completed ensemble checks (2026-10-18)

The recorded candidate was based on `a491dda1`; the current production base
`2f1bc707` differs from it in documentation only. The candidate also provided
the optional static import-chunk control now included here. No candidate
CDF-derivative, inverse-loss or rejection-sampling optimization is included.

| Sample | Completed checks | Important limitations |
|---|---|---|
| 10 TeV proton, 1000 Cp8 and 1000 C8 events | First interactions matched for all paired primary seeds; no Holm-adjusted significant mean/shape differences in the defined tests | Cp8 includes 146 CPU16/four-GPU and 854 CPU2/single-GPU events; pooled and separate topology strata were checked |
| 100 TeV Fe, 500 Cp8 and 500 C8 events | EM, muon, hadron and deposit profiles; Xmax and sampled-antenna CoREAS/ZHS fluence; no Holm-adjusted significant differences | Cp8 includes 283 CPU16/four-GPU and 217 CPU4/single-GPU events; timings must stay separated by topology |
| 1e8 GeV proton, 100 parallel-prefix events versus 100 historical serial-prefix events | Profile/deposit shapes, mean radio waveforms and ground observables passed the defined screen | Independent seeds, historical versions and two import chunks; not a controlled isolated speedup measurement |

Three old/new **serial-prefix** seed regressions also gave byte-identical
frontiers and identical numeric profiles, deposits, ground records, first
interactions and CoREAS/ZHS tables. Fourteen-species handoff tests include
protons, neutrons, charged pions/kaons, Fe-56, electrons/positrons, photons,
both muon charges and electron/muon neutrinos.

For the 10 TeV sample, the ZHS mean-waveform test had nominal p=0.00950 and
Holm-adjusted p=0.09495 across ten shape tests; it must not be described as
perfect agreement. The 100 TeV Fe relative mean differences included muons
-0.00438, hadrons -0.000235 and deposited energy -0.000966; their 95% intervals
included zero. Lack of significance is **not** equivalence at arbitrary
precision, and not every low-energy radio confidence interval fits inside 5%.

The validation records used the fixed, predeclared seeds, without replacing
failed events to obtain a passing sample. Source-side accounting checks covered
unique/disjoint roots, complete consumption, drained fallback, finite outputs
and compensated merges. Results were reviewed before raw-data archiving.

The 1e9 GeV Fe 5+5 timing study is still in progress at this release preparation;
it is not claimed as completed validation. Enabling CPU-prefix parallelism
changes later random histories and can slow low-energy showers through process
startup overhead. Do not enable it automatically or advertise one universal
speedup from these mixed/historical cohorts.
