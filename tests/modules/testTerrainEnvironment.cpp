#include <catch2/catch_all.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/modules/terrain/TerrainBoundaryTracking.hpp>
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>
#include <corsika/geometry/terrain/TerrainBoundary.hpp>
#include <corsika/modules/terrain/TerrainMagneticTracking.hpp>
#include <corsika/modules/terrain/TerrainMagneticField.hpp>
#include <corsika/modules/terrain/TerrainParticleAdmission.hpp>
#include <corsika/geometry/terrain/TerrainTrajectoryAudit.hpp>
#include <corsika/modules/transport/InterfaceRegionMap.hpp>
#include <iostream>
#include <iomanip>
using namespace corsika;
using Catch::Approx;

TEST_CASE("Interface material definitions do not depend on geometry names", "[terrain][interface]") {
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};terrain::buildAtmosphere(env,cfg);
  interfaces::validateCalculatorMaterialKeys(env);
  auto targets=interfaces::collectNuclearTargets(env);
  CHECK(targets.count(Code::Nitrogen)==1);CHECK(targets.count(Code::Oxygen)==1);
  interfaces::HomogeneousMaterial water{Medium::WaterLiquid,
      NuclearComposition({Code::Hydrogen,Code::Oxygen},{2./3.,1./3.}),1.,1.33};
  auto waterNode=env.createNode<Sphere>(Point(cs,0_m,0_m,0_m),1_m);
  waterNode->setModelProperties(water.makeModel<terrain::Interface>(cs));
  env.getUniverse()->addChild(std::move(waterNode));
  interfaces::validateCalculatorMaterialKeys(env);
  CHECK(interfaces::collectNuclearTargets(env).count(Code::Hydrogen)==1);
  interfaces::HomogeneousMaterial ice{Medium::WaterIce,water.composition,.919,1.78};
  auto iceNode=env.createNode<Sphere>(Point(cs,5_m,0_m,0_m),1_m);
  iceNode->setModelProperties(ice.makeModel<terrain::Interface>(cs));
  env.getUniverse()->addChild(std::move(iceNode));
  CHECK_THROWS_AS(interfaces::validateCalculatorMaterialKeys(env),std::invalid_argument);
  auto bad=water;bad.density_g_cm3=-1.;
  CHECK_THROWS_AS(bad.makeModel<terrain::Interface>(cs),std::invalid_argument);
}

namespace {
terrain::MeshData cube() {
  terrain::MeshData data;
  data.vertices={{{-1.,-1.,-1.}},{{1.,-1.,-1.}},{{1.,1.,-1.}},{{-1.,1.,-1.}},
                 {{-1.,-1.,1.}},{{1.,-1.,1.}},{{1.,1.,1.}},{{-1.,1.,1.}}};
  data.faces={{{0,2,1}},{{0,3,2}},{{4,5,6}},{{4,6,7}},{{0,1,5}},{{0,5,4}},
              {{1,2,6}},{{1,6,5}},{{2,3,7}},{{2,7,6}},{{3,0,4}},{{3,4,7}}};
  return data;
}
struct Particle {
  Point position;
  DirectionVector direction;
  terrain::Environment::BaseNodeType const* node;
  auto const& getPosition()const{return position;}
  auto const& getDirection()const{return direction;}
  MomentumVector getMomentum()const{return direction*1_GeV;}
  auto getEnergy()const{return 1_GeV;}
  auto getPID()const{return Code::Photon;}
  auto getCharge()const{return 0*constants::e;}
  auto getVelocity()const{return direction*constants::c;}
  auto* getNode()const{return node;}
};
struct ChargedParticle:Particle {
  double charge_number=1.;
  auto getCharge()const{return charge_number*constants::e;}
  auto getPID()const{return Code::Positron;}
};
}

TEST_CASE("Terrain magnetic field is ENU in air and zero inside rock", "[terrain][magnetic]") {
  CHECK(terrain::magneticNwuToEnu({1.,2.,3.})==std::array<double,3>{-2.,1.,3.});
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};cfg.air_magnetic_field_enu_T={1.e-5,2.e-5,-3.e-5};
  terrain::buildAtmosphere(env,cfg);auto data=cube();
  terrain::embedTerrain(env,cfg,terrain::ClosedMesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces),Point(cs,0_m,0_m,0_m));
  for(double h:{2802.,7001.,11401.,37001.,100001.}) {
    Point p(cs,2_m,0_m,(h-2802.)*1_m);
    auto b=env.getUniverse()->getContainingNode(p)->getModelProperties().getMagneticField(p);
    CHECK(b.getX(cs)/1_T==Approx(1.e-5));CHECK(b.getY(cs)/1_T==Approx(2.e-5));CHECK(b.getZ(cs)/1_T==Approx(-3.e-5));
  }
  Point p(cs,0_m,0_m,0_m);
  CHECK(env.getUniverse()->getContainingNode(p)->getModelProperties().getMagneticField(p).getNorm()==0_T);
}

TEST_CASE("Curved DEM query includes both roots and interior extrema", "[terrain][magnetic]") {
  auto cs=get_root_CoordinateSystem();auto data=cube();
  terrain::ClosedMesh mesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces);
  auto f=terrain::exportFlatTerrain(mesh);
  // Endpoints lie above the box; the parabola crosses z=1 at x=-1 and x=1.
  terrain::flat::QuadraticPath p{{{-2.,0.,2.5},{1.,0.,-2.},false},{0.,0.,.5},4.};
  auto hit=terrain::flat::nextCurvedBoundary(f.view(),p);
  REQUIRE(hit.hit.found());CHECK(hit.hit.distance==Approx(1.).margin(1.e-12));
  // Endpoints both above the box: the interior minimum must keep the BVH node.
  CHECK(terrain::flat::curveBox(f.nodes[0],p,4.));
  p.start.logically_inside=true;
  hit=terrain::flat::nextCurvedBoundary(f.view(),p);
  REQUIRE(hit.hit.found());CHECK(hit.hit.distance==Approx(3.).margin(1.e-12));
  p.maximum_length_m=.99;CHECK_FALSE(terrain::flat::nextCurvedBoundary(f.view(),p).hit.found());
  // Tangential contact must not manufacture a material transition.
  p={{{-2.,0.,3.},{1.,0.,-2.},false},{0.,0.,.5},4.};
  CHECK_FALSE(terrain::flat::nextCurvedBoundary(f.view(),p).hit.found());
}

TEST_CASE("Terrain leapfrog reproduces the stock air trajectory and zero-field reference", "[terrain][magnetic]") {
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};cfg.air_magnetic_field_enu_T={0.,0.,5.e-5};
  terrain::buildAtmosphere(env,cfg);auto data=cube();
  auto info=terrain::embedTerrain(env,cfg,terrain::ClosedMesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces),Point(cs,0_m,0_m,0_m));
  auto* rock=env.getUniverse()->getContainingNode(Point(cs,0_m,0_m,0_m));auto* air=rock->getParent();
  terrain::MagneticTracking tracking(*info.mesh);
  ChargedParticle p{{Point(cs,-2_m,.1_m,.2_m),DirectionVector(cs,{1.,0.,0.}),air},1.};
  auto [track,next]=tracking.getTrack(p);REQUIRE(next==rock);
  CHECK(track.getPosition(1.).getX(cs)/1_m==Approx(-1.).margin(1.e-12));
  CHECK(track.getPosition(1.).getY(cs)/1_m<.1);
  p.position=track.getPosition(1.);p.node=rock;p.direction=track.getDirection(1.);
  auto [straight,outside]=tracking.getTrack(p);CHECK(outside==air);
  CHECK((straight.getDirection(1.)-p.direction).getNorm()<1.e-14);
  // Same polynomial as stock air, evaluated at the same physical times. The
  // terrain radio accuracy cap introduced in the prior magnetic audit makes
  // its step shorter; identical full-step durations are no longer the contract.
  terrain::Environment reference;terrain::buildAtmosphere(reference,cfg);
  p.position=Point(cs,10000_m,0_m,10000_m);p.node=reference.getUniverse()->getContainingNode(p.position);
  tracking_leapfrog_curved::Tracking original;
  auto [expected,originalNext]=original.getTrack(p);auto [actual,actualNext]=tracking.getTrack(p);
  CHECK(actualNext==originalNext);
  double fraction=actual.getDuration()/expected.getDuration();
  REQUIRE(fraction>0.); REQUIRE(fraction<1.);
  for(double u:{0.,.25,.5,1.}) {
    CHECK((actual.getPosition(u)-expected.getPosition(u*fraction)).getNorm()/1_m<1.e-8);
    CHECK((actual.getDirection(u)-expected.getDirection(u*fraction)).getNorm()<2.e-14);
  }
  auto a=actual.getPosition(0.),b=actual.getPosition(1.);
  auto delta=p.getVelocity()*actual.getDuration();
  auto m=terrain::flat::quadraticMidpoint(
      {a.getX(cs)/1_m,a.getY(cs)/1_m,a.getZ(cs)/1_m},
      {b.getX(cs)/1_m,b.getY(cs)/1_m,b.getZ(cs)/1_m},
      {delta.getX(cs)/1_m,delta.getY(cs)/1_m,delta.getZ(cs)/1_m});
  CHECK((Point(cs,m.x*1_m,m.y*1_m,m.z*1_m)-actual.getPosition(.5)).getNorm()/1_m<1.e-8);
}

TEST_CASE("Trajectory midpoint is not the chord midpoint near terrain", "[terrain][magnetic]") {
  auto cs=get_root_CoordinateSystem();auto data=cube();
  terrain::ClosedMesh mesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces);
  auto m=terrain::flat::quadraticMidpoint({-2,0,0},{2,0,0},{4,0,8});
  CHECK(m.x==0.);CHECK(m.y==0.);CHECK(m.z==2.);
  CHECK(mesh.contains(Point(cs,0_m,0_m,0_m))); // chord: wrongly classified as rock
  CHECK_FALSE(mesh.contains(Point(cs,m.x*1_m,m.y*1_m,m.z*1_m))); // true curve: air
  auto straight=terrain::flat::quadraticMidpoint({-2,0,0},{2,0,0},{4,0,0});
  CHECK(straight.x==0.);CHECK(straight.z==0.);
}

TEST_CASE("SI scalar tracking and portable magnetic DEM length agree", "[terrain][magnetic]") {
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};cfg.air_magnetic_field_enu_T={2.e-5,-3.e-5,4.e-5};
  terrain::buildAtmosphere(env,cfg);auto data=cube();
  auto info=terrain::embedTerrain(env,cfg,terrain::ClosedMesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces),Point(cs,0_m,0_m,0_m));
  auto* rock=env.getUniverse()->getContainingNode(Point(cs,0_m,0_m,0_m));
  terrain::MagneticTracking tracking(*info.mesh);
  double stockCoupling=constants::e/(constants::c*convert_HEP_to_SI<MassType::dimension_type>(1_GeV))*1_T*1_m;
  std::cout<<std::setprecision(17)<<"stock magnetic GeV conversion="<<stockCoupling
           <<"; device conversion=0.299792458\n";
  for(int i=0;i<2000;++i) {
    double y=.6*(double(i%97)/97.-.5),z=.6*(double(i%89)/89.-.5);
    double charge=i%2?1.:-1.;
    auto direction=DirectionVector(cs,{1.,.1*(double(i%31)/31.-.5),.1*(double(i%23)/23.-.5)}).normalized();
    ChargedParticle particle{{Point(cs,-2_m,y*1_m,z*1_m),direction,rock->getParent()},charge};
    auto [track,node]=tracking.getTrack(particle);REQUIRE(node==rock);
    terrain::flat::Vec3 d{direction.getX(cs),direction.getY(cs),direction.getZ(cs)};
    auto const& b=cfg.air_magnetic_field_enu_T;
    auto q=terrain::flat::cross(d,{b[0],b[1],b[2]});
    // The oracle particle has momentum 1 GeV/c. Compare SI conversion in the
    // real scalar tracker with the portable GeV/c -> T m expression.
    double factor=.5*charge*stockCoupling;
    terrain::flat::QuadraticPath path{{{-2.,y,z},d,false},
        {factor*q.x,factor*q.y,factor*q.z},10.};
    auto hit=terrain::flat::nextCurvedBoundary(tracking.flatTerrain().view(),path);
    REQUIRE(hit.hit.found());
    CHECK(track.getDuration()*constants::c/1_m==Approx(hit.hit.distance).margin(2.e-13));
    auto point=terrain::flat::curvePosition(path,hit.hit.distance);
    auto end=track.getPosition(1.);
    CHECK(std::abs(end.getX(cs)/1_m-point.x)<2.e-13);
    CHECK(std::abs(end.getY(cs)/1_m-point.y)<2.e-13);
    CHECK(std::abs(end.getZ(cs)/1_m-point.z)<2.e-13);
    // The air device implementation uses 0.299792458, while stock units use
    // legacy e, hbar and hbar*c constants. Bound that known conversion offset
    // analytically, separately from the strict SAME-coefficient geometry test.
    double ratio=.299792458/stockCoupling;
    path.quadratic={path.quadratic.x*ratio,path.quadratic.y*ratio,path.quadratic.z*ratio};
    auto portable=terrain::flat::nextCurvedBoundary(tracking.flatTerrain().view(),path);
    REQUIRE(portable.hit.found());
    auto devicePoint=terrain::flat::curvePosition(path,portable.hit.distance);
    double fieldNorm=std::sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
    double bound=4.*std::abs(stockCoupling-.299792458)*fieldNorm+2.e-13;
    CHECK(std::abs(devicePoint.y-end.getY(cs)/1_m)<=bound);
    CHECK(std::abs(devicePoint.z-end.getZ(cs)/1_m)<=bound);
  }
}

TEST_CASE("CPU terrain direction handoff preserves angles and rejects real defects", "[terrain][admission]") {
  gpu::em::EmParticleState p;p.pid=11;p.energy_GeV=.0020629586206524823;
  p.direction[0]=.12262319820935977;p.direction[1]=.61800669947661124;p.direction[2]=.7765508809221483;
  p.history_id=223910;p.weight=7.;auto original=p;
  CHECK(terrain::canonicalizeCpuDirection(p));
  double norm=0.;for(double d:p.direction)norm+=d*d;
  CHECK(std::abs(norm-1.)<1.e-15);
  CHECK(p.energy_GeV==original.energy_GeV);CHECK(p.weight==original.weight);CHECK(p.history_id==original.history_id);
  CHECK(p.direction[0]/p.direction[1]==Approx(original.direction[0]/original.direction[1]).epsilon(1.e-15));
  auto valid=p;CHECK_FALSE(terrain::canonicalizeCpuDirection(p));
  CHECK(p.direction[0]==valid.direction[0]);
  p.direction[0]=2.;CHECK_THROWS(terrain::canonicalizeCpuDirection(p));
  p.direction[0]=NAN;CHECK_THROWS(terrain::canonicalizeCpuDirection(p));
}

TEST_CASE("Terrain admission rejects bad surfaces before BVH", "[terrain]") {
  auto data=cube();CHECK_NOTHROW(terrain::validateTerrainData(data));
  auto bad=data;bad.faces.pop_back();CHECK_THROWS(terrain::validateTerrainData(bad));
  bad=data;bad.faces[0][0]=999;CHECK_THROWS(terrain::validateTerrainData(bad));
  bad=data;std::swap(bad.faces[0][0],bad.faces[0][1]);CHECK_THROWS(terrain::validateTerrainData(bad));
  bad=data;for(auto& f:bad.faces)std::swap(f[0],f[1]);CHECK_THROWS(terrain::validateTerrainData(bad));
  bad=data;bad.vertices[0][0]=std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(terrain::validateTerrainData(bad));
}

TEST_CASE("Flat terrain matches CPU mesh and oriented face semantics", "[terrain]") {
  auto cs=get_root_CoordinateSystem();auto data=cube();
  terrain::ClosedMesh mesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces);
  auto flat=terrain::exportFlatTerrain(mesh);
  REQUIRE(flat.indices.size()==12);
  REQUIRE(flat.vertices.size()==data.vertices.size());
  REQUIRE(flat.indexed_triangles.size()==data.faces.size());
  for(std::size_t i=0;i<data.faces.size();++i)
    for(int k=0;k<3;++k)CHECK(flat.indexed_triangles[i].vertex[k]==data.faces[i][k]);
  for(double y:{-1.,-.5,0.,.5,1.})for(double z:{-1.,-.5,0.,.5,1.}) {
    terrain::flat::Ray ray{{-2.,y,z},{1.,0.,0.}};
    auto cpu=mesh.intersectRay(Point(cs,-2_m,y*1_m,z*1_m),DirectionVector(cs,{1.,0.,0.}));
    auto actual=terrain::flat::intersect(flat.view(),ray);
    // Independent inclusive cube oracle, including corner/grazing rays. The
    // old CPU mesh can discard a shared feature; that bug is not a contract.
    REQUIRE(actual.found());
    CHECK(actual.distance==Approx(1.).margin(1.e-12));
    if(cpu.hit)CHECK(actual.distance==Approx(cpu.distance/1_m).margin(1.e-12));
    else CHECK((std::abs(y)==1.||std::abs(z)==1.));
  }
  terrain::flat::Ray ray{{-2.,.1,.2},{1.,0.,0.}};
  CHECK(terrain::flat::intersect(flat.view(),ray,-1).distance==Approx(1.));
  CHECK(terrain::flat::intersect(flat.view(),ray,+1).distance==Approx(3.));
  ray.origin={1.,.1,.2};
  CHECK_FALSE(terrain::flat::intersect(flat.view(),ray,-1).found());
  ray.direction={-1.,0.,0.};
  CHECK(terrain::flat::intersect(flat.view(),ray,-1).distance==Approx(0.));
}

TEST_CASE("Terrain tracking changes material rather than absorbs", "[terrain]") {
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};terrain::buildAtmosphere(env,cfg);
  auto data=cube();terrain::validateTerrainData(data);
  auto info=terrain::embedTerrain(env,cfg,
      terrain::ClosedMesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces),
      Point(cs,0_m,0_m,0_m));
  REQUIRE(info.triangles==12);
  auto* rock=env.getUniverse()->getContainingNode(Point(cs,0_m,0_m,0_m));
  auto* air=env.getUniverse()->getContainingNode(Point(cs,2_m,0_m,0_m));
  REQUIRE(rock!=air);
  CHECK(rock->getParent()==air);
  terrain::BoundaryTracking tracking;
  Particle p{Point(cs,-2_m,.1_m,.2_m),DirectionVector(cs,{1.,0.,0.}),air};
  auto [entry,entryNode]=tracking.getTrack(p);
  CHECK(entryNode==rock);
  CHECK(entry.getDuration()*constants::c/1_m==Approx(1.).margin(1.e-9));
  p.position=entry.getPosition(1.);p.node=entryNode;
  auto [exit,exitNode]=tracking.getTrack(p);
  CHECK(exitNode==air);
  CHECK(exit.getDuration()*constants::c/1_m==Approx(2.).margin(1.e-9));
  p.position=exit.getPosition(1.);p.node=exitNode;
  auto [continuation,airNode]=tracking.getTrack(p);
  CHECK(continuation.getDuration()*constants::c/1_m>1000.);
  CHECK(airNode!=rock); // no spurious immediate re-entry at the exit face
  p.direction=-p.direction; // scattering back into the rock at its surface
  auto [reentry,reentryNode]=tracking.getTrack(p);
  CHECK(reentryNode==rock);
  CHECK(reentry.getDuration()*constants::c/1_m<=2.e-9);
  CHECK(rock->getModelProperties().getMassDensity(Point(cs,0_m,0_m,0_m))/(1_kg/(1_m*1_m*1_m))==Approx(2650.));
}

TEST_CASE("Terrain atmosphere keeps native density and rejects layer spanning mesh", "[terrain]") {
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};terrain::buildAtmosphere(env,cfg);
  for(double height:{0.,2000.,6999.9,7000.1,11399.9,11400.1,36999.9,37000.1,99999.9,100000.1,112799.}) {
    Point p(cs,0_m,0_m,(height-2802.)*1_m);
    auto* node=env.getUniverse()->getContainingNode(p);
    REQUIRE(node->hasModelProperties());
    double const b=height<7000.?1183.6071:height<11400.?1143.0425:height<37000.?1322.9748:655.67307;
    double const H=height<7000.?9542.4834:height<11400.?8000.0534:height<37000.?6295.6893:7375.2177;
    double expected=height>=100000.?1.e-6:10.*b/H*std::exp(-height/H);
    CHECK(node->getModelProperties().getMassDensity(p)/(1_kg/(1_m*1_m*1_m))==Approx(expected).epsilon(2.e-12));
  }
  auto data=cube();data.vertices[6][2]=5000.;
  CHECK_THROWS(terrain::embedTerrain(env,cfg,
      terrain::ClosedMesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces),Point(cs,0_m,0_m,0_m)));
}

TEST_CASE("Portable logical boundary matches actual CPU tracking intersection", "[terrain]") {
  terrain::Environment env;auto cs=env.getCoordinateSystem();
  terrain::TerrainAtmosphereConfig cfg{2802.};terrain::buildAtmosphere(env,cfg);
  auto data=cube();
  auto info=terrain::embedTerrain(env,cfg,
      terrain::ClosedMesh(terrain::MeshLoader::toPoints(data,cs,1_m),data.faces),
      Point(cs,0_m,0_m,0_m));
  auto flat=terrain::exportFlatTerrain(*info.mesh);
  auto* rock=env.getUniverse()->getContainingNode(Point(cs,0_m,0_m,0_m));
  auto* air=rock->getParent();
  terrain::BoundaryTracking tracking;
  for(bool inside:{false,true})for(double x:{-2.,-1.,-1.+2.e-9,0.,1.-2.e-9,1.,2.})
    for(double sign:{-1.,1.}) {
      Particle p{Point(cs,x*1_m,.1_m,.2_m),DirectionVector(cs,{sign,0.,0.}),inside?rock:air};
      auto candidate=terrain::flat::nextBoundary(flat.view(),{{x,.1,.2},{sign,0.,0.},inside});
      auto cpu=terrain::BoundaryTracking::intersect(p,*rock);
      CHECK(candidate.hit.found()==cpu.hasIntersections());
      if(candidate.hit.found()) {
        auto const duration=inside?cpu.getExit():cpu.getEntry();
        CHECK(duration*constants::c/1_m==Approx(candidate.hit.distance).margin(2.e-14));
        auto [track,next]=tracking.getTrack(p);
        CHECK(next==(inside?air:rock));
        CHECK(track.getDuration()*constants::c/1_m==Approx(candidate.hit.distance).margin(2.e-14));
        CHECK(candidate.crossing==(inside?terrain::flat::Crossing::ExitRock:terrain::flat::Crossing::EnterRock));
      } else {
        CHECK_FALSE(cpu.hasIntersections());
      }
    }
}
