#include "Egs4Tables.hpp"
#ifdef EGS4_STEP_REFERENCE_FILE
#include EGS4_STEP_REFERENCE_FILE
#endif
#include <algorithm>
#include <iostream>
#include <limits>

namespace {
void check(bool b, char const* message) { if (!b) throw std::runtime_error(message); }
void close(double a, double b, char const* message) {
  check(std::isfinite(a) && std::isfinite(b) &&
        std::abs(a - b) <= 3.e-13 * std::max({1.e-20, std::abs(a), std::abs(b)}), message);
}
struct Input { double energy, density, z, sterncor, stepfc, demfp; int charge; };
struct Output { c7_egs4::ElectronQuery electron; c7_egs4::PhotonQuery photon; c7_egs4::LocalStep step; };
KOKKOS_INLINE_FUNCTION Output evaluate(c7_egs4::TableView t, Input a) {
  auto e = c7_egs4::electronQuery(t, a.energy, .51099895, a.charge);
  return {e, c7_egs4::photonQuery(t, a.energy),
          c7_egs4::proposeLocalStep(t, e, a.energy, .51099895, 1.01099895,
                                  a.density, a.z, a.sterncor, a.stepfc, a.demfp, 1.e20)};
}
template<class Exec> struct BatchEvaluator {
  c7_egs4::TableView table;
  Kokkos::View<Input*, typename Exec::memory_space> input;
  Kokkos::View<Output*, typename Exec::memory_space> output;
  KOKKOS_INLINE_FUNCTION void operator()(int i) const { output(i) = evaluate(table, input(i)); }
};
void run(std::string const& path) {
  using namespace c7_egs4;
  auto t = readAirTables(path);
  check(t.electrons.size() == 500 && t.photons.size() == 500, "Table sizes");
  check(t.brems_pair_coefficients.size() == 43, "Brem/pair coefficient count");
#ifdef EGS4_STEP_REFERENCE_FILE
  check(t.medium.rho == c7_step::Rho && t.medium.radiation_length_cm == c7_step::Rlc &&
        t.medium.teff_cm == c7_step::Teff && t.medium.eke0 == c7_step::Eke0 &&
        t.medium.eke1 == c7_step::Eke1, "Frozen HATCH constants differ");
  auto reference = c7_step::rows();
  for (int i = 0; i < 500; ++i) {
    auto const& r = reference[i]; auto c = t.electrons[i].c;
    check(c[18] == r.t0 && c[19] == r.t1 && c[4] == r.e0 && c[5] == r.e1 &&
          c[6] == r.p0 && c[7] == r.p1, "Frozen HATCH REAL table conversion differs");
  }
#endif
  auto view = hostView(t);
  check(electronQuery(view, 1., .51099895, 0).status == Status::invalid_input, "Unsupported particle");
  check(electronQuery(view, .7, .51099895, -1).status == Status::outside_table, "AE bound");
  check(electronQuery(view, 2. * t.medium.ue, .51099895, -1).status == Status::outside_table, "UE bound");
  check(electronQuery(view, std::numeric_limits<double>::quiet_NaN(), .51099895, -1).status == Status::invalid_input, "NaN energy");
  check(photonQuery(view, .1).status == Status::outside_table, "AP bound");
  check(photonQuery(view, 2. * t.medium.up).status == Status::outside_table, "UP bound");
  check(photonQuery(view, -1.).status == Status::invalid_input, "Negative photon energy");

  // Each logarithmic table interval is exercised using interior coordinates.
  // These are synthetic kernel inputs, not independent shower events.
  std::vector<Input> inputs;
  for (int row = 1; row < 480; ++row) {
    double energy = .51099895 + std::exp((row + 1.5 - t.medium.eke0) / t.medium.eke1);
    if (energy <= 1.01099895) continue;
    for (int charge : {-1, 1}) for (double fc : {1., .0625})
      for (double rho : {t.medium.rho, 1.225e-3})
        inputs.push_back({energy, rho, -1.e6, 10., fc, .7, charge});
  }
  using Exec = Kokkos::DefaultExecutionSpace;
  DeviceTables<Exec> device(t);
  Kokkos::View<Input*, typename Exec::memory_space> in("inputs", inputs.size());
  Kokkos::View<Output*, typename Exec::memory_space> out("outputs", inputs.size());
  auto h = Kokkos::create_mirror_view(in);
  for (std::size_t i = 0; i < inputs.size(); ++i) h(i) = inputs[i];
  Kokkos::deep_copy(in, h);
  auto v = device.view();
  Kokkos::parallel_for("native_cpp_egs4_local_queries", Kokkos::RangePolicy<Exec>(0, inputs.size()),
                      BatchEvaluator<Exec>{v, in, out});
  Exec().fence();
  auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  double max_loss_relative_difference = 0.;
  double max_step_backend_relative_difference = 0.;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto a = inputs[i]; auto r = evaluate(view, a); auto actual = result(i);
    check(r.electron.status == Status::success && r.photon.status == Status::success && r.step.status == Status::success, "Valid input rejected");
    check(actual.electron.status == r.electron.status && actual.photon.status == r.photon.status &&
          actual.step.status == r.step.status && actual.electron.index == r.electron.index &&
          actual.photon.index == r.photon.index, "Host/device state mismatch");
    close(actual.electron.rate_per_cm, r.electron.rate_per_cm, "Electron rate mismatch");
    close(actual.electron.loss_MeV_per_cm, r.electron.loss_MeV_per_cm, "Electron loss mismatch");
    close(actual.electron.maximum_step_cm, r.electron.maximum_step_cm, "TMXS mismatch");
    close(actual.photon.mean_free_path_cm, r.photon.mean_free_path_cm, "Photon mfp mismatch");
    for (int j = 0; j < 3; ++j) close(actual.electron.branches[j], r.electron.branches[j], "Electron branch mismatch");
    for (int j = 0; j < 4; ++j) close(actual.photon.branches[j], r.photon.branches[j], "Photon branch mismatch");
    close(actual.step.loss_MeV_per_cm, r.step.loss_MeV_per_cm, "Density-corrected loss mismatch");
    close(actual.step.rate_per_cm, r.step.rate_per_cm, "Density-corrected rate mismatch");
    close(actual.step.true_step_cm, r.step.true_step_cm, "Local step mismatch");
    close(actual.step.projected_step_cm, r.step.projected_step_cm, "Projected step mismatch");
    max_step_backend_relative_difference = std::max(max_step_backend_relative_difference,
        std::abs(actual.step.true_step_cm - r.step.true_step_cm) / r.step.true_step_cm);
#ifdef EGS4_STEP_REFERENCE_FILE
    auto f = reference[r.electron.index];
    double x = std::log(a.energy - .51099895);
    double frozen_loss = a.charge < 0 ? f.e1 * x + f.e0 : f.p1 * x + f.p0;
    close(r.electron.loss_MeV_per_cm, frozen_loss, "Frozen loss interpolation mismatch");
    max_loss_relative_difference = std::max(max_loss_relative_difference,
        std::abs(r.electron.loss_MeV_per_cm - frozen_loss) / std::abs(frozen_loss));
#endif
    check(r.step.projected_step_cm <= r.step.true_step_cm, "Path projection ordering");
    check(r.step.true_step_cm <= (a.stepfc * 200. * t.medium.teff_cm) * (t.medium.rho / a.density) * (1. + 1.e-14), "STEPFC cap");
    close(continuousLoss(r.step.loss_MeV_per_cm, .25 * r.step.true_step_cm),
          r.step.loss_MeV_per_cm * (.25 * r.step.true_step_cm), "Local loss multiplication");
  }
  // Explicitly check the height/density correction and its 3 MeV switch.
  for (double energy : {2.999, 3., 100000.}) {
    auto q = electronQuery(view, energy, .51099895, -1);
    auto s = proposeLocalStep(view, q, energy, .51099895, 1.01099895, t.medium.rho, 0., 10., .0625, .7, 1.e20);
    double expected = energy < 3. ? q.loss_MeV_per_cm : std::min(q.loss_MeV_per_cm, 76.65 / t.medium.radiation_length_cm);
    close(s.loss_MeV_per_cm, expected, "Sternheimer correction");
  }
  auto q = electronQuery(view, 10., .51099895, -1);
  check(proposeLocalStep(view, q, 10., .51099895, 1.01099895, 0., 0., 10., .0625, .7, 1.e20).status == Status::invalid_input, "Invalid region density");
  check(proposeLocalStep(view, q, 10., .51099895, 1.01099895, t.medium.rho, 0., 10., 0., .7, 1.e20).status == Status::invalid_input, "Invalid STEPFC");
  std::cout << "execution_space=" << Exec::name() << '\n'
#ifdef EGS4_STEP_REFERENCE_FILE
            << "frozen_electron_coefficients_checked=3000\n"
#else
            << "frozen_electron_coefficients_checked=0 (external fixture not supplied)\n"
#endif
            << "batch_queries_checked=" << inputs.size() << '\n'
#ifdef EGS4_STEP_REFERENCE_FILE
            << "max_loss_relative_difference_vs_frozen=" << max_loss_relative_difference << '\n'
#endif
            << "max_local_step_relative_difference_cpu_kokkos=" << max_step_backend_relative_difference << '\n'
            << "runtime_physics_dependencies=none (EGSDAT file only)\n"
            << "scope=HATCH tables and ELECTR local proposal; NOT a complete shower backend\n";
}
}
int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "Usage: test_egs4_tables EGSDAT6_file\n"; return 2; }
  try {
    Kokkos::ScopeGuard guard(argc, argv);
    run(argv[1]); return 0;
  } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
