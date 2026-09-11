// Independent live PROPOSAL oracle. No table creation, LPM, thinning, or shower.
// This isolates fixed-process/v kinematics, including annihilation rho inversion.
#include "LiveFinalStateDriver.hpp"
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <corsika/accelerator/em/common/ProcessCapabilities.hpp>
#include <PROPOSAL/PROPOSAL.h>
#include <PROPOSAL/secondaries/SecondariesCalculator.h>
#include <PROPOSAL/secondaries/parametrization/epairproduction/KelnerKokoulinPetrukhinEpairProduction.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// Distribution diagnostic, NOT a same-random/tree test. Check |rho| CDF at
// 63 live PROPOSAL inverse quantiles. The sampling band uses a DKW/union bound
// for the CDF and sign tests (alpha=0.01 for this finite matrix). PROPOSAL's
// numerical integration error is not included in that sampling-only bound.
int auditEpairDistribution() {
  namespace em=corsika::gpu::em;
  PROPOSAL::Air medium;
  constexpr unsigned samples=65536;
  unsigned const groups=2*2*2*medium.GetComponents().size();
  double const band=std::sqrt(std::log(4.*groups/.01)/(2.*samples));
  unsigned group=0,failed=0;
  std::cout<<std::setprecision(17)<<"epair samples_per_cell="<<samples
    <<" groups="<<groups<<" sampling_band="<<band<<'\n';
  for(int pid:{11,13}) for(auto const& component:medium.GetComponents()) {
    auto const def=PROPOSAL::ParticleDef::GetParticleDefForType(pid);
    PROPOSAL::secondaries::KelnerKokoulinPetrukhinEpairProduction oracle(def,medium);
    em::BremsLpmSnapshot snapshot{};
    snapshot.lepton_mass_MeV=def.mass;
    snapshot.electron_mass_MeV=PROPOSAL::ME;
    snapshot.component_count=1;
    snapshot.components[0]={component.GetHash(),component.GetNucCharge(),
      component.GetAtomicNum(),component.GetLogConstant()};
    for(double energy:{1e3,1e7}) for(double v:{.01,.5}) {
      auto const out=migration_audit::evaluateRho(snapshot,energy,v,samples,++group);
      std::vector<double> absolute; absolute.reserve(samples);
      unsigned fallback=0,negative=0,range_errors=0;
      unsigned long long trials=0;
      double const rho_max=std::sqrt(1.-4.*PROPOSAL::ME/(energy*v))*
        (1.-6.*def.mass*def.mass/(energy*energy*(1.-v)));
      for(auto const& item:out) {
        trials+=item.trials;
        if(item.status) {++fallback;continue;}
        if(!std::isfinite(item.rho)||std::abs(item.rho)>rho_max) ++range_errors;
        negative+=(item.rho<0.);
        absolute.push_back(std::abs(item.rho));
      }
      std::sort(absolute.begin(),absolute.end());
      double max_diff=0.;
      bool oracle_valid=true;
      double last=-1.;
      for(unsigned q=1;q<64;++q) {
        double const u=double(q)/64.;
        double const x=oracle.CalculateRho(energy,v,component,u,.75);
        oracle_valid &= std::isfinite(x)&&x>=last&&x>=0.&&x<=rho_max;
        last=x;
        auto const count=std::upper_bound(absolute.begin(),absolute.end(),x)-absolute.begin();
        max_diff=std::max(max_diff,std::abs(double(count)/samples-u));
      }
      double const sign_diff=std::abs(double(negative)/samples-.5);
      bool const pass=!fallback&&!range_errors&&oracle_valid&&max_diff<=band&&sign_diff<=band;
      failed+=!pass;
      std::cout<<"epair pid="<<pid<<" Z="<<component.GetNucCharge()<<" E_MeV="<<energy
        <<" v="<<v<<" samples="<<samples<<" fallback="<<fallback
        <<" range_errors="<<range_errors<<" oracle_valid="<<oracle_valid
        <<" max_quantile_cdf_diff="<<max_diff<<" sign_diff="<<sign_diff
        <<" mean_trials="<<double(trials)/samples<<" pass="<<pass<<'\n';
    }
  }
  std::cout<<"epair cells="<<group<<" total_samples="<<group*samples
    <<" failed_cells="<<failed<<'\n';
  return failed?1:0;
}

int main(int argc,char** argv) {
  try {
    namespace em=corsika::gpu::em;
    corsika::accelerator::em::KokkosRuntimeConfig config{};
#if defined(CORSIKA8_KOKKOS_BACKEND_OPENMP)
    config.threads=2;
#else
    config.threads=1;
#endif
    corsika::accelerator::em::KokkosRuntime runtime(config);
    if(argc==2&&std::string(argv[1])=="--epair-cdf") return auditEpairDistribution();
    if(argc!=1) throw std::invalid_argument("usage: testKokkosLiveFinalStateOracle [--epair-cdf]");
    PROPOSAL::Air medium;
    auto const component=medium.GetComponents().front();
    std::vector<migration_audit::Input> inputs;
    std::vector<std::vector<PROPOSAL::ParticleState>> expected;
    using Pair=std::pair<int,int>;
    std::vector<Pair> groups{{22,em::ComptonProcessId},{11,em::BremsProcessId},
      {-11,em::BremsProcessId},{11,em::IonizationProcessId},
      {-11,em::IonizationProcessId},{13,em::IonizationProcessId},
      {-13,em::IonizationProcessId},{-11,em::AnnihilationProcessId}};
    std::array<std::array<double,3>,4> directions{{{{0,0,1}},{{0,0,-1}},
       {{.6,0,.8}},{{-.36,.48,.8}}}};
    for(auto [pid,process]:groups) {
      auto const def=PROPOSAL::ParticleDef::GetParticleDefForType(pid);
      std::vector<PROPOSAL::InteractionType> types{static_cast<PROPOSAL::InteractionType>(process)};
      auto calculator=PROPOSAL::make_secondaries(types,def,medium);
      for(double energy:{.002, .01, .1, 1., 100., 1e4, 1e6, 1e8}) {
        if(energy*1000.<=def.mass*1.01) continue;
        for(auto direction:directions) for(unsigned j=0;j<32;++j) {
          migration_audit::Input x{};
          x.rho_uniform=(j+.5)/32.; x.azimuth_uniform=(31-j+.5)/32.;
          auto& record=x.interaction;
          record.status=em::EmInteractionStatus::Selected;
          record.process_id=process; record.component_hash=component.GetHash();
          record.particle_mass_GeV=def.mass/1000.;
          record.particle.pid=pid; record.particle.energy_GeV=energy;
          record.particle.weight=1.; record.particle.history_id=inputs.size()+1;
          for(int axis=0;axis<3;++axis) record.particle.direction[axis]=direction[axis];
          double const E=energy*1000., m=def.mass;
          double v=.1*(1.-m/E);
          if(process==em::ComptonProcessId) v=(1.-1./(1.+2.*E/PROPOSAL::ME))*x.rho_uniform;
          if(std::abs(pid)==13) {
            double const gamma=E/m;
            double const tmax=2.*PROPOSAL::ME*(gamma*gamma-1.)/
              (1.+2.*gamma*PROPOSAL::ME/m+std::pow(PROPOSAL::ME/m,2));
            v=std::min(v,.2*tmax/E);
          }
          if(process==em::AnnihilationProcessId) v=1.;
          record.energy_fraction=v;
          PROPOSAL::StochasticLoss loss(process,v*E,PROPOSAL::Cartesian3D(0,0,0),
            PROPOSAL::Cartesian3D(direction[0],direction[1],direction[2]),0.,0.,E,component.GetHash());
          std::vector<double> draws;
          if(process==em::AnnihilationProcessId) draws={x.rho_uniform,x.azimuth_uniform};
          else draws={x.azimuth_uniform};
          expected.push_back(calculator->CalculateSecondaries(loss,component,draws));
          inputs.push_back(x);
        }
      }
    }
    auto const actual=migration_audit::evaluate(inputs);
    struct Metric { unsigned cases{},identity_fail{},energy_fail{},direction_fail{},nonfinite{};
      double max_energy_relative{},max_direction_absolute{},max_direction_energy_GeV{};
      double max_norm_error{}; };
    std::map<Pair,Metric> metrics;
    bool printed_first=false;
    std::cout<<std::setprecision(17);
    for(std::size_t i=0;i<inputs.size();++i) {
      auto const& x=inputs[i]; auto const& y=actual[i]; auto const& ref=expected[i];
      auto& m=metrics[{x.interaction.particle.pid,x.interaction.process_id}]; ++m.cases;
      if(y.error||y.count!=ref.size()){++m.identity_fail;continue;}
      for(std::size_t j=0;j<ref.size();++j) {
        auto const pid=ref[j].type;
        if(y.children[j].pid!=pid) ++m.identity_fail;
        double const mass=PROPOSAL::ParticleDef::GetParticleDefForType(pid).mass;
        // Scalar adapter: MeV -> HEP, subtract native mass, then stack adds C8 mass.
        double const expected_energy=(ref[j].energy-mass)/1000.+em::transportMassGeV(pid);
        if(!std::isfinite(expected_energy)||!std::isfinite(y.children[j].energy_GeV)) {
          ++m.nonfinite;
          continue;
        }
        double const delta=std::abs(expected_energy-y.children[j].energy_GeV);
        m.max_energy_relative=std::max(m.max_energy_relative,delta/std::max(std::abs(expected_energy),1e-300));
        if(delta>1e-12+1e-10*std::abs(expected_energy)) ++m.energy_fail;
        auto const d=ref[j].direction.GetCartesianCoordinates();
        double norm2=0.;
        for(int axis=0;axis<3;++axis) {
          norm2+=y.children[j].direction[axis]*y.children[j].direction[axis];
          double const diff=std::abs(d[axis]-y.children[j].direction[axis]);
          if(!std::isfinite(diff)) {++m.nonfinite;continue;}
          if(diff>m.max_direction_absolute) {
            m.max_direction_absolute=diff;
            m.max_direction_energy_GeV=x.interaction.particle.energy_GeV;
          }
          if(diff>1e-12+1e-10*std::max(std::abs(d[axis]),std::abs(y.children[j].direction[axis]))) {
            ++m.direction_fail;
            if(!printed_first) {
              std::cout<<"FIRST_DIRECTION_DIVERGENCE input="<<i<<" pid="<<x.interaction.particle.pid
                <<" process="<<x.interaction.process_id<<" E_GeV="<<x.interaction.particle.energy_GeV
                <<" v="<<x.interaction.energy_fraction<<" azimuth_uniform="<<x.azimuth_uniform
                <<" child="<<j<<" axis="<<axis<<" scalar="<<d[axis]<<" device="
                <<y.children[j].direction[axis]<<" absolute="<<diff<<'\n'; printed_first=true;
            }
          }
        }
        m.max_norm_error=std::max(m.max_norm_error,std::abs(norm2-1.));
      }
    }
    unsigned failed=0;
    for(auto const& [key,m]:metrics) {
      std::cout<<"pid="<<key.first<<" process="<<key.second<<" cases="<<m.cases
        <<" identity_fail="<<m.identity_fail<<" energy_fail="<<m.energy_fail
        <<" direction_fail="<<m.direction_fail<<" nonfinite="<<m.nonfinite
        <<" max_energy_relative="<<m.max_energy_relative
        <<" max_direction_absolute="<<m.max_direction_absolute
        <<" max_direction_energy_GeV="<<m.max_direction_energy_GeV
        <<" max_norm_squared_error="<<m.max_norm_error<<'\n';
      failed+=m.identity_fail+m.energy_fail+m.direction_fail+m.nonfinite;
    }
    std::cout<<"cases="<<inputs.size()<<" strict_diagnostic_failures="<<failed
      <<" backend="<<runtime.info().backend
      <<" (strict differences are not automatically distribution/physics failures)\n";
    return failed?1:0;
  } catch(std::exception const& error) {std::cerr<<error.what()<<'\n'; return 2;}
}
