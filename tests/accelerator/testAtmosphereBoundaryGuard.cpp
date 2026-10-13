/* (c) Copyright 2026 CORSIKA Project; BSD-3-Clause. */
#include <Kokkos_Core.hpp>
#include <corsika/accelerator/em/detail/PhotonTransportStep.hpp>
#include "AtmosphereBoundaryNativeReference.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace em = corsika::gpu::em;
namespace physics = corsika::accelerator::em::detail;
using namespace corsika;

namespace {
  void require(bool ok, char const* message) {
    if (!ok) throw std::runtime_error(message);
  }
  struct Result {
    unsigned errors{};
    physics::PhotonTransportOutcome first{};
  };
  std::vector<em::EmParticleState> fixtures() {
    // Archived Fe100 production event_0011, 0250, 0331, 0451.
    double const positions[4][3]{
      {85.92128448106335,-217.9654766378676,6377999.995764578},
      {86.71071541967895,-118.16838088708272,6377999.998259872},
      {337.14181418655215,52.430073799867024,6377999.990968304},
      {1700.5324059152017,2803.16191536913,6382399.15795008}};
    double const directions[4][3]{
      {.19338171453490643,-.8358968928578908,-.5136924147717814},
      {-.1922440628321444,-.9792533836503119,.06407051517824482},
      {.331972766700332,.19582486806251398,-.9227386971497542},
      {.5536822909620622,.8279550130711574,-.08903042741300364}};
    std::vector<em::EmParticleState> inputs;
    for (int i=0;i<4;++i) {
      em::EmParticleState p{};
      p.pid=22; p.energy_GeV=1.; p.weight=1.;
      for (int j=0;j<3;++j) { p.position_m[j]=positions[i][j]; p.direction[j]=directions[i][j]; }
      inputs.push_back(p);
    }
    // Both sides of every internal layer, both directions, tangents and
    // grazing rays; resolve the guard edge as well as adjacent floating values.
    for (double height : {7000.,11400.,37000.,100000.})
      for (double offset : {-1.e-3,-2.e-4,-1.001e-4,-1.e-4,-.999e-4,
                            -8.e-5,-2.e-5,-1.e-6,-1.e-7,-1.e-8,0.,
                            1.e-8,1.e-7,1.e-6,2.e-5,8.e-5,.999e-4,
                            1.e-4,1.001e-4,2.e-4,1.e-3})
        for (double cosine : {-1.,-.9,-.5,-.1,-.01,-1.e-4,0.,
                              1.e-4,.01,.1,.5,.9,1.})
          for (int azimuth=0;azimuth<4;++azimuth) {
            em::EmParticleState p{};
            p.pid=22; p.energy_GeV=1.; p.weight=1.;
            p.position_m[2]=6371000.+height+offset;
            p.direction[2]=cosine;
            auto const phi=azimuth*1.57079632679489661923;
            p.direction[0]=std::sqrt(1.-cosine*cosine)*std::cos(phi);
            p.direction[1]=std::sqrt(1.-cosine*cosine)*std::sin(phi);
            inputs.push_back(p);
          }
    return inputs;
  }

  template<class Ex> void run() {
    auto const env=makeAtmosphereBoundaryTestEnvironment();
    auto const inputs=fixtures();
    Kokkos::View<em::EmParticleState*,typename Ex::memory_space> states("guard_states",inputs.size());
    Kokkos::View<Result*,typename Ex::memory_space> output("guard_results",inputs.size());
    Kokkos::View<em::EnvironmentSnapshot,typename Ex::memory_space> environment("guard_environment");
    auto staging=Kokkos::create_mirror_view(states);
    for (std::size_t i=0;i<inputs.size();++i) staging(i)=inputs[i];
    Kokkos::deep_copy(states,staging);
    Kokkos::deep_copy(environment,env);
    Kokkos::parallel_for("atmosphere_guard",Kokkos::RangePolicy<Ex>(0,inputs.size()),
        KOKKOS_LAMBDA(std::size_t i) {
          Result result{};
          em::EmInteractionRecord interaction{};
          interaction.status=em::EmInteractionStatus::NoDiscreteInteraction;
          interaction.particle=states(i);
          // Follow repeated layer crossings to catch boundary ping-pong.
          for (int step=0;step<6;++step) {
            auto out=physics::transportPhoton(environment(),interaction);
            if (step==0) result.first=out;
            if (out.fallback_flag) { ++result.errors; break; }
            auto const& record=out.record;
            if (!(record.distance_m>0.) ||
                !physics::transportFinite(record.distance_m) ||
                record.end.energy_GeV!=record.start.energy_GeV ||
                record.end.time_s<=record.start.time_s ||
                record.end.step_id!=record.start.step_id+1)
              ++result.errors;
            if (record.limit!=em::PhotonTransportLimit::LayerBoundary) break;
            auto const change=record.end_layer_index-record.start_layer_index;
            if (change!=1 && change!=-1) ++result.errors;
            interaction.particle=record.end;
          }
          if (i<4 && !result.first.fallback_flag) {
            // A discrete vertex can precede the near shell. Do not skip it,
            // replay it twice, or sample its density from the wrong layer.
            interaction.particle=states(i);
            interaction.status=em::EmInteractionStatus::Selected;
            interaction.interaction_grammage_g_per_cm2=
                .01*result.first.record.traversed_grammage_g_per_cm2;
            auto const vertex=physics::transportPhoton(environment(),interaction);
            auto const& r=vertex.record;
            if (vertex.fallback_flag || r.limit!=em::PhotonTransportLimit::Interaction ||
                r.interaction.interaction_vertex_reached!=1 ||
                !(r.distance_m>0. && r.distance_m<result.first.record.distance_m) ||
                r.start_layer_index!=r.end_layer_index ||
                r.interaction.mass_density_g_per_cm3!=r.end_density_g_per_cm3)
              ++result.errors;
          }
          output(i)=result;
        });
    auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},output);
    std::vector<Result> results(inputs.size());
    unsigned errors=0;
    for (std::size_t i=0;i<results.size();++i) {
      results[i]=host(i);
      if (results[i].errors && errors<15)
        std::cerr << "failed fixture=" << i << " z=" << inputs[i].position_m[2]
                  << " nz=" << inputs[i].direction[2] << " reason="
                  << static_cast<int>(results[i].first.fallback.reason) << '\n';
      errors+=results[i].errors;
    }
    std::cout << Ex::name() << " boundary fixtures=" << inputs.size()
              << " errors=" << errors << '\n';
    require(errors==0,"boundary guard regression failed");
    std::vector<em::PhotonTransportRecord> records;
    for (auto const& result : results) records.push_back(result.first.record);
    compareNativeAtmosphereBoundary(inputs,records,env);
    // Preserve the old axial guard and the explicit outer-environment fallback.
    em::EmParticleState p{};p.pid=22;p.energy_GeV=1.;p.weight=1.;
    p.position_m[2]=6378000.+5.e-5;p.direction[2]=-1.;
    require(em::queryAtmosphereLayer(env,p.position_m,p.direction).layer_index==0,
            "Phase 57 inward guard regressed");
    p.direction[2]=1.;
    require(em::queryAtmosphereLayer(env,p.position_m,p.direction).layer_index==1,
            "Phase 57 outward guard regressed");
    p.position_m[2]=env.atmosphere_layers[4].outer_radius_m;
    em::EmInteractionRecord interaction{};interaction.particle=p;
    interaction.status=em::EmInteractionStatus::NoDiscreteInteraction;
    auto out=physics::transportPhoton(env,interaction);
    require(out.fallback_flag && out.fallback.reason==em::ProposalFallbackReason::UnsupportedGeometry,
            "Phase 73 outer-boundary safeguard removed");
  }
}

int main(int argc,char** argv) {
  try {
    std::cout << std::setprecision(17);
    Kokkos::ScopeGuard guard(Kokkos::InitializationSettings{}.set_num_threads(2));
#ifdef KOKKOS_ENABLE_OPENMP
    if (argc>1 && std::string(argv[1])=="openmp") {run<Kokkos::OpenMP>();return 0;}
#else
    (void)argc;(void)argv;
#endif
    run<Kokkos::DefaultExecutionSpace>();
    return 0;
  } catch (std::exception const& e) {std::cerr << e.what() << '\n';return 1;}
}
