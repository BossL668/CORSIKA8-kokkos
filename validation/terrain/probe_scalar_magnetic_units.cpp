// Isolated scalar LeapFrogTrajectory vs shared transport formula. No RNG,
// terrain relocation, production parameter changes, or GPU initialization.
#include <corsika/framework/geometry/LeapFrogTrajectory.hpp>
#include <corsika/accelerator/em/common/UniformMagneticField.hpp>
#include <corsika/accelerator/em/common/TransportMass.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>

int main(int argc, char** argv) {
  using namespace corsika;
  using namespace corsika::units::si;
  namespace em=corsika::gpu::em;
  auto cs=get_root_CoordinateSystem();
  auto unitMomentum=constants::c*convert_HEP_to_SI<MassType::dimension_type>(1_GeV);
  double cpuFactor=constants::e*(1_T)/unitMomentum*1_m;
  if(argc==2 && std::string(argv[1])=="--linear-gate") {
    // A separate, deliberately unresolved orchestration diagnostic. The
    // magnetic polynomial itself is not supposed to choose tracking policy.
    // Scalar getTrack() explicitly returns a zero-field trajectory at R>1e9 m.
    em::EmParticleState p{}; p.pid=11;p.energy_GeV=1.e5;p.direction[2]=-1.;
    double mass=em::TransportElectronMassGeV, b[3]={0.,5.e-5,0.};
    double momentum=std::sqrt((p.energy_GeV-mass)*(p.energy_GeV+mass));
    auto pSI=constants::c*convert_HEP_to_SI<MassType::dimension_type>(momentum*1_GeV);
    double radius=(pSI/(constants::e*(5.e-5*tesla)))/meter;
    auto limit=em::maximumUniformMagneticStep(p,mass,-1.,b);
    auto advance=em::advanceUniformMagneticField(p,mass,-1.,b,1000.);
    bool mismatch=radius>1.e9 && limit.status==em::MagneticStepStatus::Linear &&
                  advance.particle.position_m[0]!=0.;
    std::cout<<std::setprecision(17)<<"{\"scope\":\"linear-policy diagnostic; not a full shower\","
        <<"\"scalar_radius_m\":"<<radius<<",\"portable_limit_is_linear\":"
        <<(limit.status==em::MagneticStepStatus::Linear?"true":"false")
        <<",\"position_delta_from_scalar_straight_m\":"<<std::abs(advance.particle.position_m[0])
        <<",\"direction_delta_from_scalar_straight\":"<<std::abs(advance.particle.direction[0])
        <<",\"unresolved_mismatch\":"<<(mismatch?"true":"false")<<"}\n";
    return mismatch?1:0;
  }
  std::cout<<std::setprecision(17)<<"{\"scalar_factor\":"<<cpuFactor
           <<",\"portable_factor\":"<<em::GeVPerCToTeslaMeter<<",\"cases\":[";
  bool first=true;int failures=0,stockFailures=0;
  for(double field:{0.,5.e-5})for(double length:{.001,1.,1000.})for(double charge:{-1.,1.}) {
    double mass=.00051099895, energy=1.;
    double momentum=std::sqrt((energy-mass)*(energy+mass));
    auto speed=(momentum/energy)*constants::c;
    DirectionVector direction(cs,{.6,0.,.8});
    VelocityVector velocity=direction*speed;
    MagneticFieldVector magnetic(cs,0_T,field*1_T,0_T);
    auto momentumSI=constants::c*convert_HEP_to_SI<MassType::dimension_type>(momentum*1_GeV);
    auto k=charge*constants::e/momentumSI*speed;
    auto sameK=(charge*em::GeVPerCToTeslaMeter/momentum)*speed/(tesla*meter);
    Point origin(cs,0_m,0_m,0_m);
    LeapFrogTrajectory scalar(origin,velocity,magnetic,k,length*1_m/speed);
    // Control experiment only: use the device constant in the stock trajectory.
    LeapFrogTrajectory aligned(origin,velocity,magnetic,sameK,length*1_m/speed);
    em::EmParticleState start{};start.pid=charge<0?11:-11;start.energy_GeV=energy;
    start.direction[0]=.6;start.direction[2]=.8;double b[3]={0.,field,0.};
    auto device=em::advanceUniformMagneticField(start,mass,charge,b,length);
    auto delta=[&](LeapFrogTrajectory const& t) {
      auto pos=t.getPosition(1.);auto dir=t.getDirection(1.);
      double p[3]={pos.getX(cs)/1_m,pos.getY(cs)/1_m,pos.getZ(cs)/1_m};
      double d[3]={dir.getX(cs),dir.getY(cs),dir.getZ(cs)};
      std::array<double,2> r{};
      for(int a=0;a<3;++a){r[0]=std::max(r[0],std::abs(p[a]-device.particle.position_m[a]));
        r[1]=std::max(r[1],std::abs(d[a]-device.particle.direction[a]));}
      return r;
    };
    auto raw=delta(scalar),control=delta(aligned);
    stockFailures+=(raw[0]>64.*std::numeric_limits<double>::epsilon()*std::max(1.,length)
               ||raw[1]>64.*std::numeric_limits<double>::epsilon());
    failures+=(control[0]>64.*std::numeric_limits<double>::epsilon()*std::max(1.,length)
               ||control[1]>64.*std::numeric_limits<double>::epsilon());
    if(!first)std::cout<<',';
    first=false;
    std::cout<<"{\"B_T\":"<<field<<",\"length_m\":"<<length<<",\"charge\":"<<charge
             <<",\"stock_position_delta_m\":"<<raw[0]<<",\"stock_direction_delta\":"<<raw[1]
             <<",\"aligned_position_delta_m\":"<<control[0]<<",\"aligned_direction_delta\":"<<control[1]<<'}';
  }
  std::cout<<"],\"aligned_control_failures\":"<<failures
           <<",\"stock_failures\":"<<stockFailures<<"}\n";
  return (failures||stockFailures)?1:0;
}
