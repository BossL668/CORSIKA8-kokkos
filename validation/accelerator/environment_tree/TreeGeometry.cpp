#include "GeometryDriver.hpp"
#include <corsika/accelerator/em/common/EnvironmentTreeBuilder.hpp>
#include <corsika/media/Environment.hpp>
#include <corsika/modules/tracking/TrackingStraight.hpp>
#include <corsika/accelerator/em/common/FullEnvironmentTreeBuilder.hpp>
#include <corsika/accelerator/em/common/EnvironmentSnapshotStorage.hpp>
#include <corsika/media/FlatExponential.hpp>
#include <corsika/media/SlidingPlanarExponential.hpp>
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/GladstoneDaleRefractiveIndex.hpp>
#include <corsika/media/CORSIKA7Atmospheres.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>

// A frozen copy of the unmodified C8 search is the independent oracle. Only
// its class name differs; do not compare two calls to the new shared helper.
namespace corsika {
  template <typename TDerived> struct OriginalIntersect {
    template <typename TParticle> auto nextIntersect(TParticle const&,TimeType) const;
  };
}
#include "OriginalIntersect.inl"

using namespace corsika;
using namespace corsika::units::si;
namespace et=corsika::gpu::em::environment_tree;
using NativeNode=VolumeTreeNode<IEmpty>;
template <typename NativeNode> struct Particle {
  NativeNode const* node;
  Point position;
  Vector<hepmomentum_d> momentum;
  auto getNode() const { return node; }
  auto const& getPosition() const { return position; }
  auto const& getMomentum() const { return momentum; }
  auto getEnergy() const { return 1_GeV; }
};
struct NativeSearch:Intersect<tracking_line::Tracking> {
  using Intersect<tracking_line::Tracking>::nextIntersect;
};
static void require(bool condition,char const* message) {
  if (!condition) throw std::runtime_error(message);
}
static bool same(double a,double b,double& maximum) {
  if (a==b) return true;
  if (!std::isfinite(a) || !std::isfinite(b)) return false;
  double rel=std::abs(a-b)/std::max(1.e-12,std::max(std::abs(a),std::abs(b)));
  maximum=std::max(maximum,rel);
  return rel<=3.e-11;
}
static bool sameBoundary(double a,double b,double& maximum,double& max_abs_m,double tolerance) {
  auto relative_ok=same(a,b,maximum);
  if (a==b) return true;
  if (!std::isfinite(a) || !std::isfinite(b)) return false;
  auto abs_m=std::abs(a-b)*299792458.;max_abs_m=std::max(max_abs_m,abs_m);
  return relative_ok || abs_m<=tolerance;
}
static double elapsed(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}

template <typename I>
static nlohmann::json exercise(Environment<I>& env,std::string const& name,
    double scale=120.,double offset=0.,double step_limit=200.) {
  using NativeNode=typename Environment<I>::BaseNodeType;
  auto cs=env.getCoordinateSystem();
  auto host=et::exportEnvironmentTree(*env.getUniverse(),cs,
      [](auto const&,unsigned i){return static_cast<int>(i);});
  require(host.nodes.size()==host.native_nodes.size(),"node loss in export");
  for (unsigned i=0;i<host.nodes.size();++i) {
    auto const& node=host.nodes[i]; auto const& original=*host.native_nodes[i];
    require(node.parent==host.id(original.getParent()),"parent ID changed");
    require(node.child_count==original.getChildNodes().size(),"child loss");
    require(node.excluded_count==original.getExcludedNodes().size(),"exclusion loss");
    require(node.model_id==static_cast<int>(i),"model payload ID changed");
  }
  std::vector<Query> queries;
  std::vector<Particle<NativeNode>> particles;
  std::vector<Result> reference;
  OriginalIntersect<tracking_line::Tracking> original;
  NativeSearch refactored;
  std::mt19937_64 rng(719235);
  std::uniform_real_distribution<double> uniform(-1.,1.);
  constexpr unsigned samples=65536;
  for (unsigned i=0;i<samples;++i) {
    Query q{};
    for (int a=0;a<3;++a) {q.p[a]=uniform(rng)*scale;q.v[a]=uniform(rng);}
    q.p[2]+=offset;
    double norm=std::sqrt(et::dot(q.v,q.v));
    for (auto& v:q.v) v/=norm;
    // Test axis-parallel rays, exact boundaries and offsets on both sides.
    if (i%64<6) {
      for (auto& v:q.v) v=0.;
      q.v[(i%64)/2]=i%2?-1.:1.;
      if (i%2==0) q.p[0]=100.+(static_cast<int>(i%5)-2)*1.e-10;
    }
    Point p(cs,q.p[0]*1_m,q.p[1]*1_m,q.p[2]*1_m);
    auto node=env.getUniverse()->getContainingNode(p);
    require(node!=nullptr,"universe must contain test point");
    q.node=host.id(node);
    q.limit=i%7==0?std::numeric_limits<double>::infinity():step_limit;
    particles.push_back({node,p,Vector<hepmomentum_d>(cs,q.v[0]*1_GeV,q.v[1]*1_GeV,q.v[2]*1_GeV)});
    auto const& particle=particles.back();
    // Use exactly the native velocity after its unit arithmetic.
    auto velocity=(particle.getMomentum()/particle.getEnergy()*constants::c).getComponents(cs);
    for (int a=0;a<3;++a) q.v[a]=velocity[a]/(1_m/1_s);
    q.limit/=constants::c/(1_m/1_s);
    auto [t,next]=original.nextIntersect(particle,q.limit*1_s);
    auto [tr,nr]=refactored.nextIntersect(particle,q.limit*1_s);
    require(t==tr && next==nr,"refactoring changed native search decisions");
    reference.push_back({t/1_s,host.id(next),q.node}); queries.push_back(q);
  }

  double max_host=0.,max_device=0.,max_host_abs_m=0.,max_device_abs_m=0.;
  double const abs_tolerance=std::max(1.e-10,64.*std::numeric_limits<double>::epsilon()*(scale+std::abs(offset)));
  for (unsigned i=0;i<queries.size();++i) {
    auto const& q=queries[i];et::StraightAccess access{};
    static_cast<et::TreeView&>(access)=host.view();
    std::copy(q.p,q.p+3,access.position);std::copy(q.v,q.v+3,access.velocity);
    auto r=geometry_detail::nextEnvironmentBoundary(access,q.node,q.limit);
    if (r.node!=reference[i].node || !sameBoundary(r.distance,reference[i].distance,max_host,max_host_abs_m,abs_tolerance)) {
      std::cerr<<name<<" host mismatch i="<<i<<" current="<<q.node<<" node="<<r.node
               <<" expected="<<reference[i].node<<" distance="<<r.distance
               <<" expected="<<reference[i].distance<<'\n';
      throw std::runtime_error("portable/native search mismatch");
    }
    require(et::locate(host.view(),q.p)==q.node,"portable/native containment mismatch");
  }
  auto device=runDeviceGeometry(host.nodes,host.children,host.exclusions,queries);
  auto const& ho=device.results;
  for (unsigned i=0;i<queries.size();++i) {
    if (ho[i].node!=reference[i].node || ho[i].located!=reference[i].located ||
        !sameBoundary(ho[i].distance,reference[i].distance,max_device,max_device_abs_m,abs_tolerance)) {
      std::cerr<<std::setprecision(17)<<name<<" device mismatch "<<i<<" current="<<queries[i].node
          <<" node="<<ho[i].node<<" expected="<<reference[i].node
          <<" located="<<ho[i].located<<" expected="<<reference[i].located
          <<" distance_s="<<ho[i].distance<<" expected_s="<<reference[i].distance
          <<" abs_error_m="<<std::abs(ho[i].distance-reference[i].distance)*299792458.<<'\n';
      throw std::runtime_error("device/native tree mismatch");
    }
  }
  // Include locate+search in all three timings. Allocations/copies are outside
  // kernel timing; no claim that this is full-shower speed or end-to-end GPU cost.
  std::vector<double> native_ms,portable_ms;
  volatile double checksum=0.;
  for (int rep=0;rep<7;++rep) {
    auto start=std::chrono::steady_clock::now();
    for (unsigned i=0;i<queries.size();++i) {
      auto const& q=queries[i];auto const& particle=particles[i];
      auto node=env.getUniverse()->getContainingNode(particle.getPosition());
      auto [t,next]=original.nextIntersect(particle,q.limit*1_s);
      checksum+=(std::isfinite(t/1_s)?t/1_s:0.)+(node!=nullptr)+(next!=nullptr);
    }
    native_ms.push_back(elapsed(start));start=std::chrono::steady_clock::now();
    for (auto const& q:queries) {
      et::StraightAccess access{};static_cast<et::TreeView&>(access)=host.view();
      std::copy(q.p,q.p+3,access.position);std::copy(q.v,q.v+3,access.velocity);
      auto r=geometry_detail::nextEnvironmentBoundary(access,q.node,q.limit);
      checksum+=(std::isfinite(r.distance)?r.distance:0.)+(et::locate(host.view(),q.p)>=0)+(r.node>=0);
    }
    portable_ms.push_back(elapsed(start));
  }
  auto median=[](auto v){std::sort(v.begin(),v.end());return v[v.size()/2];};
  return { {"scenario",name},{"samples",samples},{"nodes",host.nodes.size()},
    {"children",host.children.size()},{"exclusions",host.exclusions.size()},
    {"native_refactor_exact",true},{"host_max_scaled_error",max_host},
    {"device_max_scaled_error",max_device},{"node_mismatches",0},
    {"host_max_absolute_error_m",max_host_abs_m},{"device_max_absolute_error_m",max_device_abs_m},
    {"absolute_tolerance_m",abs_tolerance},{"relative_tolerance",3.e-11},
    {"acceptance","exact node IDs; relative OR absolute distance tolerance (64 epsilon position scale, minimum 1e-10 m)"},
    {"native_one_core_ms",median(native_ms)},{"portable_one_core_ms",median(portable_ms)},
    {"kokkos_kernel_ms",median(device.kernel_ms)},{"native_repetitions_ms",native_ms},
    {"portable_repetitions_ms",portable_ms},{"kokkos_repetitions_ms",device.kernel_ms},
    {"upload_including_allocation_ms",device.upload_ms},{"download_ms",device.download_ms},
    {"checksum",static_cast<double>(checksum)} };
}

using Interface=IRefractiveIndexModel<IMediumPropertyModel<IMagneticFieldModel<IMediumModel>>>;
template<typename Density>
using Extra=GladstoneDaleRefractiveIndex<MediumPropertyModel<UniformMagneticField<Density>>>;
static nlohmann::json modelTest() {
  Environment<Interface> env;auto cs=env.getCoordinateSystem();
  Point origin(cs,0_m,0_m,0_m),reference(cs,0_m,0_m,100_m);
  MagneticFieldVector b(cs,2.e-5_T,-1.e-5_T,3.e-5_T);
  NuclearComposition comp({Code::Nitrogen,Code::Oxygen},{.8,.2});
  auto one=env.createNode<Sphere>(origin,100000_m);
  auto two=env.createNode<Sphere>(origin,90000_m);
  auto three=env.createNode<Sphere>(origin,80000_m);
  auto shared=env.createNode<Sphere>(origin,70000_m);
  auto same_model=one->setModelProperties<Extra<HomogeneousMedium<Interface>>>(
      1.0003,reference,Medium::AirDry1Atm,b,1.2_kg/(1_m*1_m*1_m),comp);
  two->setModelProperties<Extra<SlidingPlanarExponential<Interface>>>(
      1.0003,reference,Medium::AirDry1Atm,b,origin,1.2_kg/(1_m*1_m*1_m),-8000_m,comp,100_m);
  three->setModelProperties<Extra<FlatExponential<Interface>>>(
      1.0003,reference,Medium::AirDry1Atm,b,origin,DirectionVector(cs,{0.,0.,1.}),
      1.2_kg/(1_m*1_m*1_m),-8000_m,comp);
  shared->setModelProperties(same_model);
  three->addChild(std::move(shared));two->addChild(std::move(three));
  one->addChild(std::move(two));env.getUniverse()->addChild(std::move(one));
  auto exported=et::exportFullEnvironmentTree(*env.getUniverse(),cs);
  require(exported.topology.nodes.size()==5 && exported.models.size()==3,"full snapshot node/model loss");
  require(exported.topology.nodes[0].model_id==-1,"model-free universe was invented");
  require(exported.topology.nodes[1].model_id==exported.topology.nodes[4].model_id,"shared model identity lost");
  std::vector<ModelQuery> queries;
  std::vector<ModelResult> expected;
  double max_host=0.;
  for (int id=1;id<=3;++id) {
    auto const& native=exported.topology.native_nodes[id]->getModelProperties();
    auto model_id=exported.topology.nodes[id].model_id;
    auto const& model=exported.models[model_id];
    require(model.medium_id==static_cast<int>(Medium::AirDry1Atm) && model.composition_count==2,"medium/composition lost");
    for (unsigned j=0;j<model.composition_count;++j) {
      auto const& c=exported.composition[model.composition_begin+j];
      require(c.code==static_cast<int>(comp.getComponents()[j]) && c.fraction==comp.getFractions()[j],"composition changed");
    }
    auto bf=native.getMagneticField(reference).getComponents(cs);
    for (int j=0;j<3;++j) require(model.magnetic_field_T[j]==bf[j]/1_T,"field changed");
    for (int i=0;i<4096;++i) {
      ModelQuery q{};q.id=model_id;
      q.p[0]=10.*std::sin(i);q.p[1]=20.*std::cos(i);q.p[2]=100.+i*15.;
      Point p(cs,q.p[0]*1_m,q.p[1]*1_m,q.p[2]*1_m);
      VelocityVector v(cs,0_m/1_s,0_m/1_s,(i%2?-1.:1.)*constants::c);
      StraightTrajectory track(Line(p,v),(i%32)*10_m/constants::c);
      auto direction=track.getDirection(0).getComponents(cs);
      for (int a=0;a<3;++a) q.direction[a]=direction[a].magnitude();
      q.length=track.getLength()/1_m;
      auto x=native.getIntegratedGrammage(track);q.grammage=x/(1_g/(1_cm*1_cm));
      ModelResult oracle{native.getMassDensity(p)/(1_g/(1_cm*1_cm*1_cm)),native.getRefractiveIndex(p),
        q.grammage,native.getArclengthFromGrammage(track,x)/1_m};
      require(same(et::density(model,q.p),oracle.rho,max_host),"density export changed physics");
      require(same(et::refractiveIndex(model,q.p),oracle.index,max_host),"refractivity export changed physics");
      require(same(et::grammage(model,q.p,q.direction,q.length),oracle.grammage,max_host),"grammage export changed physics");
      require(same(et::arclength(model,q.p,q.direction,q.grammage),oracle.length,max_host),"inverse grammage export changed physics");
      queries.push_back(q);expected.push_back(oracle);
    }
  }
  et::SnapshotStorage storage(exported);
  require(testNodeQueue(),"logical-node identity lost in device queue");
  auto output=runDeviceModels(exported.models,queries,storage.view());double max_device=0.;
  for (unsigned i=0;i<output.size();++i) {
    require(same(output[i].rho,expected[i].rho,max_device),"device density error");
    require(same(output[i].index,expected[i].index,max_device),"device refractivity error");
    require(same(output[i].grammage,expected[i].grammage,max_device),"device grammage error");
    require(same(output[i].length,expected[i].length,max_device),"device inverse grammage error");
  }
  return {{"queries",queries.size()},{"models",exported.models.size()},{"host_max_relative_error",max_host},
          {"device_max_relative_error",max_device},{"shared_models_preserved",true},{"composition_and_field_exact",true},
          {"full_device_owner_used",true},{"logical_node_queue_roundtrip",true}};
}

static void invalidExportTests() {
  // Unsupported geometry is an explicit error, never dropped from the tree.
  struct Unsupported:IVolume {bool contains(Point const&) const override {return true;}};
  Environment<IEmpty> env;
  env.getUniverse()->addChild(env.createNode<Unsupported>());
  bool rejected=false;
  try {(void)et::exportEnvironmentTree(*env.getUniverse(),env.getCoordinateSystem(),[](auto const&,unsigned){return -1;});}
  catch (std::invalid_argument const&) {rejected=true;}
  require(rejected,"unknown shape must not be silently skipped");
  Environment<IEmpty> broken;
  auto external=broken.createNode<Sphere>(Point(broken.getCoordinateSystem(),0_m,0_m,0_m),1_m);
  broken.getUniverse()->excludeOverlapWith(external);
  rejected=false;
  try {(void)et::exportEnvironmentTree(*broken.getUniverse(),broken.getCoordinateSystem(),[](auto const&,unsigned){return -1;});}
  catch (std::out_of_range const&) {rejected=true;}
  require(rejected,"external exclusion pointer must not be silently skipped");
}

static nlohmann::json photonTransportTest() {
  namespace em=corsika::gpu::em;
  using MaterialNode=typename Environment<Interface>::BaseNodeType;
  Environment<Interface> env;auto cs=env.getCoordinateSystem();Point origin(cs,0_m,0_m,0_m);
  NuclearComposition composition({Code::Nitrogen,Code::Oxygen},{.8,.2});
  auto outer=env.createNode<Sphere>(origin,200_m);
  auto middle=env.createNode<Sphere>(origin,100_m);
  auto inner=env.createNode<Sphere>(origin,50_m);
  unsigned density_index=0;
  for (auto node:{outer.get(),middle.get(),inner.get()}) {
    ++density_index;
    node->setModelProperties<Extra<HomogeneousMedium<Interface>>>(
      1.0003,origin,Medium::AirDry1Atm,MagneticFieldVector(cs,0_T,0_T,0_T),
      density_index*1.2_kg/(1_m*1_m*1_m),composition);
  }
  middle->addChild(std::move(inner));outer->addChild(std::move(middle));env.getUniverse()->addChild(std::move(outer));
  auto full=et::exportFullEnvironmentTree(*env.getUniverse(),cs);et::SnapshotStorage storage(full);
  std::vector<em::EmParticleState> input;
  for (unsigned i=0;i<256;++i) {
    em::EmParticleState p;p.pid=22;p.energy_GeV=1.;p.medium_id=17;
    p.position_m[0]=(i%16)-7.5;p.position_m[1]=(i/16)-7.5;p.position_m[2]=150.;
    auto angle=.001*(static_cast<int>(i%5)-2);p.direction[0]=std::sin(angle);p.direction[2]=-std::cos(angle);
    p.history_id=i+1;p.environment_node=full.topology.id(env.getUniverse()->getContainingNode(
        Point(cs,p.position_m[0]*1_m,p.position_m[1]*1_m,p.position_m[2]*1_m)));
    input.push_back(p);
  }
  auto output=runPhotonSequences(storage.view(),input);
  OriginalIntersect<tracking_line::Tracking> native;
  unsigned steps=0;double max_position=0.,max_column=0.;
  for (unsigned i=0;i<input.size();++i) {
    auto const& q=input[i];auto const& sequence=output[i];
    require(sequence.fallback==0,"full-tree photon transport fell back");
    Particle<MaterialNode> particle{full.topology.native_nodes[q.environment_node],
      Point(cs,q.position_m[0]*1_m,q.position_m[1]*1_m,q.position_m[2]*1_m),
      Vector<hepmomentum_d>(cs,q.direction[0]*1_GeV,q.direction[1]*1_GeV,q.direction[2]*1_GeV)};
    for (unsigned j=0;j<sequence.count;++j) {
      auto const& record=sequence.records[j];auto const current=particle.getNode();
      auto [time,next]=native.nextIntersect(particle,std::numeric_limits<double>::infinity()*1_s);
      auto velocity=particle.getMomentum()/particle.getEnergy()*constants::c;
      StraightTrajectory track(Line(particle.position,velocity),time);
      auto column=current->getModelProperties().getIntegratedGrammage(track)/(1_g/(1_cm*1_cm));
      auto endpoint=track.getPosition(1).getCoordinates(cs);
      require(record.start.environment_node==full.topology.id(current),"start logical node lost");
      require(record.end.environment_node==full.topology.id(next),"photon logical transition differs from C8");
      require(record.end.step_id==j+1,"transport identity not advanced once");
      for (int a=0;a<3;++a) {
        auto error=std::abs(record.end.position_m[a]-endpoint[a]/1_m);
        max_position=std::max(max_position,error);require(error<1.e-10,"photon endpoint differs from native C8");
      }
      auto error=std::abs(record.traversed_grammage_g_per_cm2-column);
      max_column=std::max(max_column,error);require(error<1.e-11,"node-specific grammage changed");
      particle.position=track.getPosition(1);particle.node=next;++steps;
      if (!next || !next->hasModelProperties()) {
        require(j+1==sequence.count && record.limit==em::PhotonTransportLimit::EscapedEnvironment,"escape was not terminal");
        break;
      }
      require(record.end.medium_id==static_cast<int>(next->getModelProperties().getMedium()),"next medium ID lost");
    }
    require(sequence.count==5,"expected all nested inward and outward boundaries");
  }
  return {{"particles",input.size()},{"steps",steps},{"node_mismatches",0},
    {"max_position_error_m",max_position},{"max_grammage_error_g_cm2",max_column},
    {"scope","actual photon transport, five consecutive logical-node crossings"}};
}

int main(int argc,char** argv) {
  initializeDevice(argc,argv);
  struct Cleanup {~Cleanup(){finalizeDevice();}} cleanup;
  nlohmann::json report; report["backend"]=deviceName();
  Environment<IEmpty> nested;auto cs=nested.getCoordinateSystem();
  auto outer=nested.createNode<Sphere>(Point(cs,0_m,0_m,0_m),100_m);
  auto left=nested.createNode<Sphere>(Point(cs,-15_m,0_m,0_m),40_m);
  auto right=nested.createNode<Sphere>(Point(cs,15_m,0_m,0_m),40_m);
  left->excludeOverlapWith(right);
  auto rotated=make_rotation(make_translation(cs,{0_m,0_m,20_m}),
                             QuantityVector<dimensionless_d>(Eigen::Vector3d(1.,1.,0.)),.41);
  auto box=nested.createNode<Box>(rotated,12_m,14_m,16_m);
  right->addChild(std::move(box));
  outer->addChild(std::move(left));outer->addChild(std::move(right));
  nested.getUniverse()->addChild(std::move(outer));
  report["tests"].push_back(exercise(nested,"nested_spheres_rotated_box_exclusion"));

  Environment<IEmpty> slabs;
  auto slab=slabs.createNode<SeparationPlane>(Plane(Point(cs,0_m,0_m,20_m),DirectionVector(cs,{0.,0.,1.})));
  slab->addChild(slabs.createNode<SeparationPlane>(Plane(Point(cs,0_m,0_m,-20_m),DirectionVector(cs,{0.,0.,1.}))));
  slabs.getUniverse()->addChild(std::move(slab));
  report["tests"].push_back(exercise(slabs,"nested_half_spaces"));

  Environment<IEmpty> deep;
  auto* parent=deep.getUniverse().get();
  for (int i=0;i<128;++i) {
    auto child=deep.createNode<Sphere>(Point(cs,0_m,0_m,0_m),(200.-i)*1_m);
    auto* next=child.get();parent->addChild(std::move(child));parent=next;
  }
  report["tests"].push_back(exercise(deep,"128_level_sphere_tree"));
  report["models"]=modelTest();
  report["photon_transport"]=photonTransportTest();
  invalidExportTests();report["invalid_export_rejected"]=true;
  Environment<Interface> atmosphere;
  Point center(cs,0_m,0_m,0_m),sea_level(cs,0_m,0_m,constants::EarthRadius::Mean);
  MagneticFieldVector field(cs,2.e-5_T,0_T,3.e-5_T);
  create_5layer_atmosphere<Interface,Extra>(atmosphere,AtmosphereId::USStdBK,center,
      1.000327,sea_level,Medium::AirDry1Atm,field);
  auto full=et::exportFullEnvironmentTree(*atmosphere.getUniverse(),cs);
  require(full.topology.nodes.size()==6 && full.models.size()==5,"native atmosphere export incomplete");
  report["tests"].push_back(exercise(atmosphere,"native_C8_USStdBK_atmosphere",100000.,
      constants::EarthRadius::Mean/1_m+50000.,200000.));
  report["native_atmosphere_export"]={{"nodes",full.topology.nodes.size()},{"models",full.models.size()},
      {"composition_entries",full.composition.size()}};
  report["scope"]="straight geometry, full model ownership, logical-node queue and multi-step photon transport; full application/shower integration pending";
  std::cout<<report.dump(2)<<'\n';
}
