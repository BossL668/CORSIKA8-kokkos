#include "GeometryDriver.hpp"
#include <corsika/accelerator/em/common/FullEnvironmentTreeBuilder.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <corsika/modules/tracking/TrackingLeapFrogCurved.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace corsika {
  template <typename TDerived> struct OriginalIntersect {
    template <typename TParticle> auto nextIntersect(TParticle const&,TimeType) const;
  };
}
#include "OriginalIntersect.inl"
using namespace corsika;
using namespace corsika::units::si;
namespace et=corsika::gpu::em::environment_tree;
using Interface=IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
template<typename T> using Extra=GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<T>>>;
using NativeNode=VolumeTreeNode<Interface>;
struct Particle {
  NativeNode const* node;Point position;Vector<hepmomentum_d> momentum;
  HEPEnergyType energy;int sign;
  auto getNode() const {return node;}
  auto const& getPosition() const {return position;}
  auto const& getMomentum() const {return momentum;}
  auto getEnergy() const {return energy;}
  auto getVelocity() const {return momentum/energy*constants::c;}
  auto getCharge() const {return sign*constants::e;}
  auto getPID() const {return sign<0?Code::Electron:Code::Positron;}
};
struct NativeCurvedSearch:Intersect<tracking_leapfrog_curved::Tracking> {
  using Intersect<tracking_leapfrog_curved::Tracking>::nextIntersect;
};
static void require(bool ok,char const* message) {if (!ok) throw std::runtime_error(message);}
static double median(std::vector<double> v) {std::sort(v.begin(),v.end());return v[v.size()/2];}
static nlohmann::json exercise(Environment<Interface>& env,std::string const& name,
                             double scale,double offset,bool atmosphere) {
  auto cs=env.getCoordinateSystem();auto host=et::exportFullEnvironmentTree(*env.getUniverse(),cs);
  std::mt19937_64 random(73421);
  std::uniform_real_distribution<double> uniform(-1.,1.),positive(0.,1.);
  std::vector<CurvedQuery> queries;std::vector<Particle> particles;std::vector<Result> expected;
  OriginalIntersect<tracking_leapfrog_curved::Tracking> native;
  OriginalIntersect<tracking_line::Tracking> straight;
  unsigned linear=0,transitions=0,exact_refactored=0;
  constexpr unsigned count=16384;
  for (unsigned i=0;i<count;++i) {
    double xyz[3]{scale*uniform(random),scale*uniform(random),offset+scale*uniform(random)};
    if (atmosphere) {xyz[0]*=.1;xyz[1]*=.1;}
    Point position(cs,xyz[0]*1_m,xyz[1]*1_m,xyz[2]*1_m);
    auto node=env.getUniverse()->getContainingNode(position);
    require(node && node->hasModelProperties(),"curved test point lacks material");
    double dir[3]{uniform(random),uniform(random),uniform(random)};
    auto norm=std::sqrt(et::dot(dir,dir));for (auto& d:dir) d/=norm;
    auto const field=node->getModelProperties().getMagneticField(position);
    if (i%31==0 && field.getNorm()>0_T) {
      auto unit=field.normalized().getComponents(cs);
      for (int a=0;a<3;++a) dir[a]=unit[a].magnitude();
    }
    double p=std::pow(10.,-3.+5.*positive(random));
    if (i%23==0) p=1.e12; // C8's high-gyroradius straight fallback.
    Particle particle{node,position,{cs,dir[0]*p*1_GeV,dir[1]*p*1_GeV,dir[2]*p*1_GeV},
                      std::hypot(p,0.00051099895)*1_GeV,i%43==0?0:(i%2?1:-1)};
    auto const velocity=particle.getVelocity();auto const speed=velocity.getNorm();
    auto const direction=velocity.normalized();
    auto const p_perp=field.getNorm()==0_T?particle.getMomentum().getNorm():
      (particle.getMomentum()-particle.getMomentum().getParallelProjectionOnto(field)).getNorm();
    auto const p_si=constants::c*convert_HEP_to_SI<MassType::dimension_type>(particle.getMomentum().getNorm());
    auto const rg=particle.sign==0 || field.getNorm()==0_T?et::infinity()*1_m:
      convert_HEP_to_SI<MassType::dimension_type>(p_perp)*constants::c/(abs(particle.getCharge())*field.getNorm());
    CurvedQuery q{};q.node=host.topology.id(node);auto& a=q.track;
    auto v=velocity.getComponents(cs);
    auto d=direction.getComponents(cs);
    auto b=field.getComponents(cs);
    for (int j=0;j<3;++j) {a.position[j]=xyz[j];a.velocity[j]=v[j]/(1_m/1_s);a.direction[j]=d[j].magnitude();a.field[j]=b[j]/1_T;}
    a.speed=speed/(1_m/1_s);a.charge_SI=particle.getCharge()/1_C;
    a.momentum_SI=p_si/(1_kg*1_m/1_s);a.gyroradius_m=rg/1_m;
    a.linear=particle.sign==0 || field.getNorm()==0_T || p_perp<1_eV || rg>1.e9_m;
    q.limit=a.linear?et::infinity():2*std::cos(.2)*std::sin(.2)*a.gyroradius_m/a.speed;
    TimeType t;NativeNode const* next;
    if (a.linear) {std::tie(t,next)=straight.nextIntersect(particle,q.limit*1_s);++linear;}
    else {
      std::tie(t,next)=native.nextIntersect(particle,q.limit*1_s);
      auto [t2,n2]=NativeCurvedSearch().nextIntersect(particle,q.limit*1_s);
      require(t==t2 && next==n2,"native tree refactoring changed curved search");++exact_refactored;
    }
    if (next!=node) ++transitions;
    queries.push_back(q);particles.push_back(particle);
    expected.push_back({t/1_s,host.topology.id(next),q.node});
  }
  std::vector<CurvedResult> host_results;
  for (auto const& q:queries) {
    auto a=q.track;static_cast<et::TreeView&>(a)=host.topology.view();
    auto r=geometry_detail::nextEnvironmentBoundary(a,q.node,q.limit);
    host_results.push_back({{r.distance,r.node,et::locate(host.topology.view(),a.position)},a.unsupported_shape,a.incomplete_roots});
  }
  auto device=runDeviceCurved(host.topology.nodes,host.topology.children,host.topology.exclusions,queries);
  auto audit=[&](auto const& values) {
    unsigned nodes=0,distances=0,unsupported=0,incomplete=0;
    double max_abs=0.,max_scaled=0.;nlohmann::json failures=nlohmann::json::array();
    for (unsigned i=0;i<count;++i) {
      auto const& actual=values[i];auto const& ref=expected[i];
      unsupported+=actual.unsupported;incomplete+=actual.incomplete;
      bool node_bad=actual.result.node!=ref.node || actual.result.located!=ref.located;
      nodes+=node_bad;
      double abs_error=0.,rel=0.;bool distance_bad=false;
      if (actual.result.distance!=ref.distance) {
        abs_error=std::abs(actual.result.distance-ref.distance)*queries[i].track.speed;
        rel=std::abs(actual.result.distance-ref.distance)/std::max(1.e-12,std::abs(ref.distance));
        max_abs=std::max(max_abs,abs_error);max_scaled=std::max(max_scaled,rel);
        // Native curved intersections already classify roots with a 1e-4 m
        // margin. Use a substantially stricter 1e-7 m numerical test, without
        // relaxing exact logical-node agreement.
        distance_bad=!std::isfinite(abs_error) || (abs_error>1.e-7 && rel>3.e-10);
      }
      distances+=distance_bad;
      if ((node_bad || distance_bad || actual.incomplete) && failures.size()<12)
        failures.push_back({{"i",i},{"current",queries[i].node},{"node",actual.result.node},{"expected_node",ref.node},
          {"distance_m",actual.result.distance*queries[i].track.speed},{"expected_m",ref.distance*queries[i].track.speed},
          {"absolute_error_m",abs_error},{"scaled_error",rel},{"incomplete",actual.incomplete}});
    }
    return nlohmann::json{{"node_mismatches",nodes},{"distance_mismatches",distances},{"unsupported",unsupported},
      {"incomplete_roots",incomplete},{"max_absolute_error_m",max_abs},{"max_scaled_error",max_scaled},{"examples",failures},
      {"passed",nodes==0 && distances==0 && unsupported==0 && incomplete==0}};
  };
  std::vector<double> native_ms,portable_ms;volatile double checksum=0.;
  for (int repeat=0;repeat<7;++repeat) {
    auto t=std::chrono::steady_clock::now();
    for (unsigned i=0;i<count;++i) {
      auto r=queries[i].track.linear?straight.nextIntersect(particles[i],queries[i].limit*1_s):native.nextIntersect(particles[i],queries[i].limit*1_s);
      auto value=std::get<0>(r)/1_s;
      checksum+=std::isfinite(value)?value:0.;
    }
    native_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
    t=std::chrono::steady_clock::now();
    for (auto const& q:queries) {
      auto a=q.track;static_cast<et::TreeView&>(a)=host.topology.view();
      auto value=geometry_detail::nextEnvironmentBoundary(a,q.node,q.limit).distance;
      checksum+=std::isfinite(value)?value:0.;
    }
    portable_ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
  }
  return {{"scenario",name},{"queries",count},{"linear_fallbacks",linear},{"transitions",transitions},
    {"native_refactored_exact",exact_refactored},{"host",audit(host_results)},{"device",audit(device.results)},
    {"native_one_core_ms",median(native_ms)},{"portable_one_core_ms",median(portable_ms)},
    {"kokkos_kernel_ms",median(device.kernel_ms)},{"checksum",double(checksum)}};
}
int main(int argc,char** argv) {
  initializeDevice(argc,argv);int code=0;
  try {
    Environment<Interface> env;auto cs=env.getCoordinateSystem();Point origin(cs,0_m,0_m,0_m);
    MagneticFieldVector b(cs,.003_T,.004_T,.005_T);
    NuclearComposition composition({Code::Nitrogen,Code::Oxygen},{.8,.2});
    auto outer=env.createNode<Sphere>(origin,400_m);
    auto model=outer->setModelProperties<Extra<HomogeneousMedium<Interface>>>(
      1.0003,origin,Medium::AirDry1Atm,b,1.2_kg/(1_m*1_m*1_m),composition);
    auto inner=env.createNode<Sphere>(origin,100_m);inner->setModelProperties(model);
    auto center=env.createNode<Sphere>(origin,30_m);center->setModelProperties(model);
    auto excluded=env.createNode<Sphere>(Point(cs,80_m,0_m,0_m),25_m);excluded->setModelProperties(model);
    inner->excludeOverlapWith(excluded);
    inner->addChild(std::move(center));outer->addChild(std::move(inner));outer->addChild(std::move(excluded));
    env.getUniverse()->addChild(std::move(outer));
    nlohmann::json result{{"backend",deviceName()},{"scope","curved boundary microbenchmark, not full transport/shower"}};
    result["tests"].push_back(exercise(env,"nested_spheres_exclusion",120.,0.,false));
    Environment<Interface> atmosphere;auto acs=atmosphere.getCoordinateSystem();
    Point earth(acs,0_m,0_m,0_m),surface(acs,0_m,0_m,constants::EarthRadius::Mean);
    MagneticFieldVector field(acs,20.e-6_T,0_T,-40.e-6_T);
    create_5layer_atmosphere<Interface,Extra>(atmosphere,AtmosphereId::USStdBK,earth,
        1.000327,surface,Medium::AirDry1Atm,field);
    result["tests"].push_back(exercise(atmosphere,"actual_C8_USStdBK",50000.,constants::EarthRadius::Mean/1_m+51000.,true));
    Environment<Interface> planes;auto pcs=planes.getCoordinateSystem();
    Point po(pcs,0_m,0_m,0_m);MagneticFieldVector pb(pcs,.003_T,.004_T,.005_T);
    auto upper=planes.createNode<SeparationPlane>(Plane(Point(pcs,0_m,0_m,200_m),DirectionVector(pcs,{0.,0.,1.})));
    auto pm=upper->setModelProperties<Extra<HomogeneousMedium<Interface>>>(
      1.0003,po,Medium::AirDry1Atm,pb,1.2_kg/(1_m*1_m*1_m),composition);
    auto lower=planes.createNode<SeparationPlane>(Plane(Point(pcs,0_m,0_m,0_m),DirectionVector(pcs,{0.,0.,1.})));
    lower->setModelProperties(pm);upper->addChild(std::move(lower));planes.getUniverse()->addChild(std::move(upper));
    result["tests"].push_back(exercise(planes,"nested_half_spaces",120.,0.,false));
    Environment<Interface> zero;auto zcs=zero.getCoordinateSystem();Point zo(zcs,0_m,0_m,0_m);
    auto ball=zero.createNode<Sphere>(zo,400_m);
    ball->setModelProperties<Extra<HomogeneousMedium<Interface>>>(
      1.0003,zo,Medium::AirDry1Atm,MagneticFieldVector(zcs,0_T,0_T,0_T),1.2_kg/(1_m*1_m*1_m),composition);
    zero.getUniverse()->addChild(std::move(ball));
    result["tests"].push_back(exercise(zero,"zero_field_linear_fallback",120.,0.,false));
    result["passed"]=true;
    for (auto const& test:result["tests"])
      if (!test["host"]["passed"].get<bool>() || !test["device"]["passed"].get<bool>()) result["passed"]=false;
    std::cout<<result.dump(2)<<'\n';if (!result["passed"].get<bool>()) code=2;
  } catch (std::exception const& e) {std::cerr<<e.what()<<'\n';code=1;}
  finalizeDevice();return code;
}
