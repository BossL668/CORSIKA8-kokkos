#include "ScalarConstantsDriver.hpp"
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <corsika/accelerator/em/common/PhotonPairKinematics.hpp>
#include <corsika/accelerator/em/RandomDomains.hpp>
#include <corsika/accelerator/em/common/SphericalAtmosphere.hpp>

namespace scalar_constants_audit {
struct Evaluate {
  using Space = Kokkos::DefaultExecutionSpace;
  Kokkos::View<Input*, Space> input;
  Kokkos::View<Output*, Space> output;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
    namespace em = corsika::gpu::em;
    namespace c = corsika::accelerator::scalar_constants;
    auto const& x = input(i);
    auto& y = output(i);
    y.advance = em::advanceUniformMagneticField(
        x.particle, x.mass_GeV, x.charge, x.field_T, x.distance_m);
    y.limit = em::maximumUniformMagneticStep(
        x.particle, x.mass_GeV, x.charge, x.field_T);
    y.mass_GeV = em::transportMassGeV(x.particle.pid);
    y.c = c::SpeedOfLightMPerS;
    y.e = c::ElementaryChargeC;
    y.epsilon = c::VacuumPermittivityFPerM;
    y.magnetic_factor = em::GeVPerCToTeslaMeter;
    y.proposal_electron_GeV=em::ElectronMassGeV;
    y.proposal_muon_GeV=em::MuonMassGeV;
    y.muon_lifetime_s=em::MuonMeanLifetimeS;
    y.maximum_time_s=em::ParticleCutMaximumTimeS;
    // Production grammage integrator: 1 kg/m^3 = 0.001 g/cm^3.
    em::EnvironmentSnapshot environment{};
    environment.number_of_layers=1;
    environment.observation_radius_m=6371000.;
    environment.observation_plane_point_m[2]=6371000.;
    environment.observation_plane_normal[2]=1.;
    auto& layer=environment.atmosphere_layers[0];
    layer.inner_radius_m=6371000.;
    layer.outer_radius_m=6471000.;
    layer.density_model = em::DensityModel::Homogeneous;
    layer.density_parameter_a = 0.001;
    double position[3]={0.,0.,6371100.};
    y.grammage = em::atmosphereGrammage(
        environment, 0, position, x.particle.direction, x.distance_m).value;
  }
};

std::vector<Output> evaluate(std::vector<Input> const& input) {
  using Space = Kokkos::DefaultExecutionSpace;
  Kokkos::View<Input*, Space> in("constants_input", input.size());
  Kokkos::View<Output*, Space> out("constants_output", input.size());
  auto host = Kokkos::create_mirror_view(in);
  for (std::size_t i=0; i<input.size(); ++i) host(i)=input[i];
  Kokkos::deep_copy(in, host);
  Kokkos::parallel_for("scalar_constants_audit",
      Kokkos::RangePolicy<Space>(0, input.size()), Evaluate{in, out});
  auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  std::vector<Output> values(input.size());
  for (std::size_t i=0; i<input.size(); ++i) values[i]=result(i);
  return values;
}
}
