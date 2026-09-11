/* Terrain extension of stock LeapFrogCurved. The atmospheric sphere solver,
 * maximum deflection, trajectory and time conventions are unchanged. Only
 * finite DEM faces add a quadratic/BVH intersection; the mesh is cached once. */
#pragma once
#include <corsika/modules/terrain/TerrainBoundaryTracking.hpp>
#include <corsika/modules/tracking/TrackingLeapFrogCurved.hpp>
#include <corsika/geometry/terrain/FlatTerrainExport.hpp>
#include <corsika/geometry/terrain/TerrainCurvedBoundary.hpp>

namespace corsika::terrain {
// Borrow the indexed mesh while retaining StraightTrajectory (zero-field mode).
// Only DEM intersections change; atmospheric sphere tracking and RNG do not.
class StraightTrackingView : public BoundaryTracking {
 public:
  explicit StraightTrackingView(FlatTerrainData const& mesh):BoundaryTracking(mesh),mesh_(mesh){}
  FlatTerrainData const& flatTerrain()const noexcept{return mesh_;}
 private:
  FlatTerrainData const& mesh_;
};
class MagneticTracking : public Intersect<MagneticTracking> {
  // Intersection context belongs to this call, never a global or thread-local
  // particle stack. CRTP still performs the original environment tree walk.
  template<class P> struct Query {
    P const& particle;
    flat::View mesh;
    TriangularMesh const* source;
    flat::QuadraticPath path;
    bool linear;
    auto getPosition()const{return particle.getPosition();}
    auto getDirection()const{return particle.getDirection();}
    auto getVelocity()const{return particle.getVelocity();}
    auto getMomentum()const{return particle.getMomentum();}
    auto getEnergy()const{return particle.getEnergy();}
    auto getCharge()const{return particle.getCharge();}
    auto getPID()const{return particle.getPID();}
    auto getNode()const{return particle.getNode();}
  };
 public:
  explicit MagneticTracking(TriangularMesh const& mesh,double maxDeflection=.2)
      :mesh_(exportFlatTerrain(mesh)),source_(&mesh),angle_(maxDeflection) {
    if(!std::isfinite(angle_)||angle_<=0.||angle_>.2)
      throw std::invalid_argument("terrain deflection must be in (0,0.2] rad");
  }
  FlatTerrainData const& flatTerrain()const noexcept{return mesh_;}
  template<class P> auto getTrack(P const& p)const {
    auto const velocity=p.getVelocity();auto const speed=velocity.getNorm();
    auto const position=p.getPosition();auto const cs=source_->getCoordinateSystem();
    auto field=p.getNode()->getModelProperties().getMagneticField(position);
    auto const charge=p.getCharge();
    bool linear=charge==0*constants::e||field.getNorm()==0_T;
    LengthType limit=std::numeric_limits<double>::infinity()*1_m;
    if(!linear) {
      auto perpendicular=(p.getMomentum()-p.getMomentum().getParallelProjectionOnto(field)).getNorm();
      auto radius=convert_HEP_to_SI<MassType::dimension_type>(perpendicular)*constants::c/(abs(charge)*field.getNorm());
      linear=perpendicular<1_eV||radius>1.e9_m;
      if(!linear)limit=2.*std::cos(angle_)*std::sin(angle_)*radius;
    }
    auto momentumSI=constants::c*convert_HEP_to_SI<MassType::dimension_type>(p.getMomentum().getNorm());
    auto k=charge/momentumSI*speed;
    if(linear){field=MagneticFieldVector(cs,0_T,0_T,0_T);k=0./(tesla*second);}
    auto direction=velocity.normalized();
    // q has dimension 1/m. Evaluate using the stock SI conversion, so this is
    // an independent CPU oracle for the portable GeV/T expression on device.
    auto q=.5*direction.cross(field)*k/speed;
    Query<P> query{p,mesh_.view(),source_,
        {{{position.getX(cs)/1_m,position.getY(cs)/1_m,position.getZ(cs)/1_m},
          {direction.getX(cs),direction.getY(cs),direction.getZ(cs)},
          &p.getNode()->getVolume()==source_},
         {q.getX(cs)*1_m,q.getY(cs)*1_m,q.getZ(cs)*1_m},limit/1_m},linear};
    auto [duration,node]=this->nextIntersect(query,limit/speed);
    return std::make_tuple(LeapFrogTrajectory(position,velocity,field,k,duration),node);
  }
  template<class P,class Node> static Intersections intersect(Query<P> const& p,Node const& node) {
    auto const* mesh=dynamic_cast<TriangularMesh const*>(&node.getVolume());
    if(!mesh) {
      if(p.linear)return tracking_line::Tracking::intersect(p,node);
      return tracking_leapfrog_curved::Tracking::intersect(p,node);
    }
    if(mesh!=p.source)throw std::logic_error("tracker was not configured for this terrain mesh");
    // Includes the zero-curvature branch, using exactly the same indexed
    // straight-ray query as Kokkos, instead of the old edge-vector oracle.
    auto h=flat::nextCurvedBoundary(p.mesh,p.path);
    if(!h.hit.found())return {};
    TimeType time=h.hit.distance*1_m/p.getVelocity().getNorm();
    return p.path.start.logically_inside?Intersections(0_s,TimeType(time)):Intersections(TimeType(time));
  }
  static std::string getName(){return "Terrain-LeapFrog-QuadraticBVH";}
  static std::string getVersion(){return "1.0";}
 private:
  FlatTerrainData mesh_;
  TriangularMesh const* source_;
  double angle_;
};
} // namespace corsika::terrain
