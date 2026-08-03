# Phase 17: GPU discrete ionization

## Scope

This phase moves electron and positron discrete ionization final states from
the explicit CPU fallback path to the mixed charged-electromagnetic CUDA
wavefront.

Continuous ionization loss was already handled during GPU transport. This
phase implements the stochastic delta-ray branch selected from the PROPOSAL
inverse-CDF table:

```text
e-/e+ + e-(medium) -> e-/e+ + e-
```

The scalar CPU backend remains unchanged.

## Exact PROPOSAL 7.6.2 final state

The device reproduces `PROPOSAL::secondaries::NaivIonization`. For projectile
total energy \(E_i\), projectile/electron mass \(m_e\), and selected loss
fraction \(v\),

\[
E_f = E_i(1-v),
\qquad
E_\delta = E_i v + m_e.
\]

The second term includes the rest energy of the struck medium electron, so

\[
E_f + E_\delta = E_i + m_e.
\]

The three momentum magnitudes are

\[
p_i=\sqrt{(E_i+m_e)(E_i-m_e)},
\]

\[
p_f=\sqrt{(E_f+m_e)(E_f-m_e)},
\qquad
p_\delta=\sqrt{(E_\delta+m_e)(E_\delta-m_e)}.
\]

The outgoing-primary and delta-electron polar cosines are

\[
\cos\phi =
\frac{(E_i+m_e)E_f-E_i m_e-m_e^2}{p_i p_f},
\]

\[
\cos\bar{\phi} =
\frac{(E_i+m_e)E_\delta-E_i m_e-m_e^2}{p_i p_\delta}.
\]

One Philox value chooses the azimuth. The delta electron uses the opposite
azimuth:

```text
(seed, shower_id, history_id, step_id,
 IonizationProcessId, IonizationAzimuthDrawId=1)
```

The implementation also preserves a subtle PROPOSAL rounding order. CORSIKA's
specified final-state bridge first computes

```text
loss_MeV = v * projectile_energy_MeV
```

and `NaivIonization` then recomputes `loss_MeV/projectile_energy_MeV`.
The device does the same before evaluating energies and angles. This removed a
roughly \(3\times10^{-14}\) CPU/GPU discrepancy in the strict event oracle.

## Mixed charged-final-state batch

`GpuProcessCapability::Ionization` accepts the process only for electrons and
positrons.

The existing stable scan-based charged batch now distinguishes:

- bremsstrahlung;
- positron annihilation;
- discrete ionization.

All currently supported charged final states have two children, so one CUB
exclusive scan still allocates deterministic secondary offsets without an
atomic append. `BremsFinalStateRecord::process_id` defines the interpretation
of its generic fields.

New accounting:

- `ionization_interaction_count` in the device batch;
- `ionization_interactions` in the host batch;
- `GpuEmStatistics::ionization_final_states`.

Ionization does not enter either bremsstrahlung LPM counter.

## Validation

`testGpuIonizationFinalState` runs 4,096 events across:

- electron and positron projectiles;
- total energies from the configured 0.5 MeV kinetic cut to 100 GeV;
- loss fractions spanning the allowed outgoing-primary energy range;
- axial and non-axial incoming directions.

Every accepted event is compared to a real
`PROPOSAL::secondaries::NaivIonization` instance for:

- the Philox azimuth and draw ID;
- multiplicity, PIDs, history IDs and parent IDs;
- both energies and target-electron energy closure;
- both three-dimensional directions;
- repeated-run deterministic equality.

The test also verifies explicit handling of:

- a photon carrying the ionization process ID;
- zero loss;
- excessive loss;
- no-interaction continuation;
- statistics isolation from annihilation and bremsstrahlung LPM.

The result is 73,738 successful checks.

`testGpuLeptonTransport` was updated so its staged and device-chained paths
compare per-process counts and records. Its explicit CPU fallback test now
uses the still-unimplemented electron-pair process, proving that enabling
ionization did not remove fallback coverage.

## Remaining work

The main unsupported charged final state is electron/positron pair production.
Unlike ionization, its PROPOSAL final state samples a conditional asymmetry
\(\rho(E,v,u)\) by numerical integration and then applies an LPM acceptance
factor depending on \(\rho\). Implementing it requires a versioned conditional
final-state table (or an equivalent device sampler) plus new Epair LPM
metadata.

The charged wavefront also still round-trips its produced particles through
the host. A resident mixed photon/lepton scheduler remains the next major
performance architecture milestone.
