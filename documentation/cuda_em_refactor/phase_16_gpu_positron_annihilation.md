# Phase 16: GPU positron annihilation

## Scope

This phase moves the PROPOSAL 7.6.2
`AnnihilationHeitler + HeitlerAnnihilation` path from an explicit CPU fallback
to the charged-electromagnetic GPU wavefront. It preserves the existing
bremsstrahlung path and does not change the scalar CPU backend.

The supported process is

```text
e+ + e- (medium) -> gamma + gamma
```

for an in-flight positron above the configured charged-particle transport cut.
An electron carrying the same process ID remains invalid and is returned as an
explicit `GpuProcessNotImplemented` fallback.

## PROPOSAL formulas reproduced on the device

The direct annihilation cross section always reports a stochastic loss
fraction

\[
v = 1.
\]

It does not use that number to divide the final-state energy. The final-state
generator consumes two additional independent random numbers:

1. \(u_\rho\), used to sample the photon energy split \(\rho\);
2. \(u_\phi\), used to sample the common azimuth.

For positron total energy \(E\), electron mass \(m_e\), and
\(\gamma=E/m_e\),

\[
\rho_\min = \frac{1}{2}
  \left(1-\sqrt{\frac{\gamma-1}{\gamma+1}}\right),\qquad
\rho_\max = \frac{1}{2}
  \left(1+\sqrt{\frac{\gamma-1}{\gamma+1}}\right).
\]

The device solves the same implicit distribution as PROPOSAL:

\[
F(\rho)=a_1\log\rho+\frac{a_2}{\rho}-\rho,
\]

\[
F(\rho)-F(\rho_\min)
-u_\rho\left[F(\rho_\max)-F(\rho_\min)\right]=0,
\]

where

\[
a_2=(\gamma+1)^{-2},\qquad
a_1=1+2\gamma a_2.
\]

The implementation deliberately reproduces PROPOSAL's bisection details:

- at most 101 iterations;
- stopping width
  \(\rho_\min E[\mathrm{MeV}]\,10^{-5}\);
- return the lower endpoint of the final bracket, not its midpoint.

The photon energies are

\[
E_{\gamma,0}=(E+m_e)(1-\rho),\qquad
E_{\gamma,1}=(E+m_e)\rho.
\]

The additional \(m_e\) is the rest energy of the medium electron. Therefore
the correct event-level closure is

\[
E_{\gamma,0}+E_{\gamma,1}=E_{e^+}+m_e,
\]

not merely \(E_{e^+}\).

The two polar cosines use the same two-body kinematics as PROPOSAL, while the
second photon receives the opposite azimuth \(\phi+\pi\).

## Deterministic random-number mapping

The counter-based Philox keys are

```text
(seed, shower_id, history_id, step_id,
 AnnihilationProcessId, AnnihilationRhoDrawId=1)

(seed, shower_id, history_id, step_id,
 AnnihilationProcessId, AnnihilationAzimuthDrawId=2)
```

The inverse-CDF interaction selection has already used draw zero. Thus no
random-number key is reused and wavefront ordering cannot alter an individual
history.

## Data-path changes

`GpuProcessCapability` now has a dedicated `Annihilation` value, accepted only
for positrons.

The former bremsstrahlung-only final-state batch is now a mixed charged-lepton
batch while retaining source-compatible names:

- `BremsFinalStateRecord` stores `process_id`;
- `final_state_uniform` and `final_state_draw_id` preserve the annihilation
  \(\rho\) draw;
- `photon_energy_fraction` stores bremsstrahlung \(v\) or annihilation
  \(\rho\), selected by `process_id`;
- device and host results expose separate `brems_interactions` and
  `annihilation_interactions` counts;
- `GpuEmStatistics::annihilation_final_states` records accepted events;
- bremsstrahlung LPM counters no longer include non-bremsstrahlung events.

Both accepted processes produce two children, so the existing stable CUB
exclusive-scan allocation remains valid. No global atomic append is used.

The full charged pipeline now supports

```text
rate selection
  -> continuous transport
  -> interaction-vertex rate reselection
  -> mixed Brems / Annihilation GPU final state
  -> stable gamma/e-/e+ routing
```

## Validation

`testGpuAnnihilationFinalState` performs 4,096 events over the production
domain from the 0.5 MeV kinetic cut to 100 GeV. For every event it compares
against a real
`PROPOSAL::secondaries::HeitlerAnnihilation` instance:

- Philox random values and draw IDs;
- process and history identity;
- multiplicity and both photon PIDs;
- energy split and energy closure;
- both three-dimensional directions;
- repeated-run deterministic equality.

It also checks:

- an electron with the annihilation process ID is an explicit fallback;
- a non-unit stochastic loss is rejected;
- a below-rest-energy positron is rejected;
- no-interaction continuations remain separate;
- annihilation never increments bremsstrahlung LPM statistics.

The synthetic full-pipeline table now includes an annihilation rate only for
positrons. `testGpuLeptonTransport` verifies that the device-chained pipeline
actually selects and executes at least one annihilation event and remains
identical to the staged reference path.

After full relinking, all 19 CUDA-specific tests pass.

## Remaining work

The charged wavefront still returns its produced particles to the host after
each validation call. The next architectural step is a resident mixed
electromagnetic scheduler so annihilation photons can enter the resident
photon queue and bremsstrahlung leptons can re-enter the resident charged
queue without a host round trip.

Electron/positron pair production and discrete ionization final states are
still explicit fallbacks and must be implemented before the first complete
production physics backend.
