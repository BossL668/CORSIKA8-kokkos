// Independent scalar unit/LeapFrog oracle; device side invokes production code.
#include "ScalarConstantsDriver.hpp"
#include <corsika/accelerator/em/KokkosRuntime.hpp>
#include <corsika/framework/geometry/LeapFrogTrajectory.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <PROPOSAL/Constants.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    using namespace corsika;
    using namespace corsika::units::si;
    namespace em = corsika::gpu::em;
    accelerator::em::KokkosRuntimeConfig config;
#ifdef CORSIKA8_KOKKOS_BACKEND_OPENMP
    config.threads=2;
#else
    config.threads=1;
#endif
    accelerator::em::KokkosRuntime runtime(config);
    auto cs=get_root_CoordinateSystem();
    auto const momentumUnit=constants::c*convert_HEP_to_SI<MassType::dimension_type>(1_GeV);
    double const factor=constants::e*tesla/momentumUnit*meter;
    std::vector<scalar_constants_audit::Input> input;
    std::vector<Code> codes;
    for (auto code : {Code::Electron,Code::Positron,Code::MuMinus,Code::MuPlus})
      for (double kinetic : {0.0005001,0.01,1.,1.e5})
        for (double field : {0.,5.e-5})
          for (auto direction : {std::array<double,3>{.6,0.,.8},
                                 std::array<double,3>{0.,1.,0.},
                                 std::array<double,3>{0.,0.,-1.}})
            for (double length : {1.e-4,0.01,1.}) {
              scalar_constants_audit::Input x{};
              x.particle.pid=static_cast<int>(get_PDG(code));
              x.mass_GeV=get_mass(code)/1_GeV;
              x.particle.energy_GeV=kinetic+x.mass_GeV;
              x.charge=static_cast<double>(get_charge_number(code));
              for (int a=0; a<3; ++a) x.particle.direction[a]=direction[a];
              x.field_T[1]=field;
              x.distance_m=length;
              input.push_back(x); codes.push_back(code);
            }
    auto output=scalar_constants_audit::evaluate(input);
    double maxPosition=0.,maxDirection=0.,maxTime=0.,maxGrammage=0.,maxLimit=0.;
    std::size_t failures=0;
    for (std::size_t i=0; i<input.size(); ++i) {
      auto const& x=input[i]; auto const& y=output[i];
      double p=std::sqrt((x.particle.energy_GeV-x.mass_GeV)*(x.particle.energy_GeV+x.mass_GeV));
      auto speed=constants::c*(p/x.particle.energy_GeV);
      DirectionVector dir(cs,{x.particle.direction[0],x.particle.direction[1],x.particle.direction[2]});
      MagneticFieldVector b(cs,0_T,x.field_T[1]*1_T,0_T);
      auto pSI=constants::c*convert_HEP_to_SI<MassType::dimension_type>(p*1_GeV);
      auto k=x.charge*constants::e/pSI*speed;
      auto dt=x.distance_m*1_m/speed;
      LeapFrogTrajectory scalar(Point(cs,0_m,0_m,0_m),dir*speed,b,k,dt);
      auto position=scalar.getPosition(1.); auto finalDir=scalar.getDirection(1.);
      double pos[3]={position.getX(cs)/1_m,position.getY(cs)/1_m,position.getZ(cs)/1_m};
      double d[3]={finalDir.getX(cs),finalDir.getY(cs),finalDir.getZ(cs)};
      double dp=0.,dd=0.;
      for (int a=0; a<3; ++a) {
        dp=std::max(dp,std::abs(pos[a]-y.advance.particle.position_m[a]));
        dd=std::max(dd,std::abs(d[a]-y.advance.particle.direction[a]));
      }
      double timeError=std::abs(y.advance.particle.time_s-dt/1_s)/(dt/1_s);
      // Independent dimensional reference, not another portable integrator.
      double grammage=((1_kg/(meter*meter*meter))*(x.distance_m*meter))/(1_g/(1_cm*1_cm));
      double grammageError=std::abs(y.grammage-grammage)/grammage;
      bool constantsOK=y.c==constants::c/(meter/second) &&
          y.e==constants::e/coulomb && y.epsilon==constants::epsilonZero/(farad/meter) &&
          y.mass_GeV==get_mass(codes[i])/1_GeV && std::abs(y.magnetic_factor/factor-1.)<2.e-15 &&
          std::abs(y.proposal_electron_GeV/(PROPOSAL::ME/1000.)-1.)<2.e-15 &&
          std::abs(y.proposal_muon_GeV/(PROPOSAL::MMU/1000.)-1.)<2.e-15 &&
          y.muon_lifetime_s==get_lifetime(Code::MuMinus)/1_s && y.maximum_time_s==10_ms/1_s;
      double limitError=0.;
      bool limitOK=true;
      if (x.field_T[1]==0. || std::abs(x.particle.direction[1])==1.) {
        limitOK=y.limit.status==em::MagneticStepStatus::Linear;
      } else {
        // These directions are perpendicular to B; scalar tracking's radius.
        double radius=(pSI/(constants::e*(x.field_T[1]*tesla)))/meter;
        if (radius>1.e9) limitOK=y.limit.status==em::MagneticStepStatus::Linear;
        else {
          double expected=2.*std::cos(.2)*std::sin(.2)*radius;
          limitError=std::abs(y.limit.distance_m/expected-1.);
          limitOK=y.limit.status==em::MagneticStepStatus::Success && limitError<3.e-14;
        }
      }
      bool ok=constantsOK && limitOK && timeError<3.e-14 && grammageError<3.e-14 &&
          dp<64.*std::numeric_limits<double>::epsilon()*std::max(1.,x.distance_m) &&
          dd<64.*std::numeric_limits<double>::epsilon() &&
          (y.advance.status==em::MagneticStepStatus::Success || y.advance.status==em::MagneticStepStatus::Linear);
      failures+=!ok;
      if (!ok) std::cerr<<"failed case "<<i<<" constants="<<constantsOK<<" limit="<<limitOK<<'\n';
      maxPosition=std::max(maxPosition,dp);maxDirection=std::max(maxDirection,dd);
      maxTime=std::max(maxTime,timeError);maxGrammage=std::max(maxGrammage,grammageError);
      maxLimit=std::max(maxLimit,limitError);
    }
    std::cout<<std::setprecision(17)<<"{\"cases\":"<<input.size()<<",\"failures\":"<<failures
        <<",\"scalar_factor\":"<<factor<<",\"position_abs_m\":"<<maxPosition
        <<",\"direction_abs\":"<<maxDirection<<",\"flight_time_relative\":"<<maxTime
        <<",\"grammage_relative\":"<<maxGrammage<<",\"step_limit_relative\":"<<maxLimit<<"}\n";
    return failures?1:0;
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
