// Target-local terrain transport: logical-side, oriented boundary intersections.
// No stock tracking/radio or beta4 header is replaced by this file.
#pragma once
#include <corsika/geometry/terrain/ValidatedTerrain.hpp>
#include <corsika/modules/tracking/TrackingStraight.hpp>
#include <corsika/geometry/terrain/FlatTerrainData.hpp>
#include <corsika/geometry/terrain/TerrainBoundary.hpp>

namespace corsika::terrain {
using namespace corsika;
using namespace corsika::units::si;

class BoundaryTracking : public Intersect<BoundaryTracking> {
  template<class P> struct IndexedParticle {
    P const& particle;
    FlatTerrainData const& terrain;
    auto getPosition()const{return particle.getPosition();}
    auto getDirection()const{return particle.getDirection();}
    auto getMomentum()const{return particle.getMomentum();}
    auto getEnergy()const{return particle.getEnergy();}
    auto getNode()const{return particle.getNode();}
  };
 public:
  explicit BoundaryTracking(bool legacy = false) : legacy_(legacy) {}
  explicit BoundaryTracking(FlatTerrainData const& terrain):legacy_(false),terrain_(&terrain){}
  template<class P> auto getTrack(P const& p) {
    if (legacy_) return legacyTracking_.getTrack(p);
    auto [time, node] = terrain_ ? this->nextIntersect(IndexedParticle<P>{p,*terrain_}) : this->nextIntersect(p);
    VelocityVector const velocity = p.getMomentum()/p.getEnergy()*constants::c;
    return std::make_tuple(StraightTrajectory(Line(p.getPosition(), velocity), time), node);
  }
  template<class P,class Node>
  static Intersections intersect(IndexedParticle<P> const& p,Node const& node) {
    auto const* mesh=dynamic_cast<TriangularMesh const*>(&node.getVolume());
    if(!mesh)return tracking_line::Tracking::intersect(p,node);
    auto const cs=mesh->getCoordinateSystem();
    auto const pos=p.getPosition();auto const dir=p.getDirection();
    bool const inside=p.getNode()==&node;
    auto hit=flat::nextBoundary(p.terrain.view(),
        {{pos.getX(cs)/1_m,pos.getY(cs)/1_m,pos.getZ(cs)/1_m},
         {dir.getX(cs),dir.getY(cs),dir.getZ(cs)},inside});
    if(!hit.hit.found())return {};
    auto speed=p.getMomentum().getNorm()/p.getEnergy()*constants::c;
    TimeType time=hit.hit.distance*1_m/speed;
    return inside?Intersections(0_s,TimeType(time)):Intersections(TimeType(time));
  }
  template<class P, class Node>
  static Intersections intersect(P const& p, Node const& node) {
    auto const* mesh = dynamic_cast<TriangularMesh const*>(&node.getVolume());
    if (!mesh) return tracking_line::Tracking::intersect(p, node);
    bool const logicallyInside = p.getNode() == &node;
    DirectionVector const direction = p.getDirection();
    auto const speed = p.getMomentum().getNorm()/p.getEnergy()*constants::c;
    // Search a tiny distance backwards to include an intersection rounded a
    // few ulps behind the endpoint. This does NOT move the particle, and is
    // independent of the user padding (which is not the geometry resolution).
    LengthType const tolerance = 1e-8 * 1_m;
    auto const origin = p.getPosition()-tolerance*direction;
    auto hits = mesh->intersectRayAll(origin, direction);
    for (auto const& hit : hits) {
      auto const distance = hit.distance-tolerance;
      if (distance < -tolerance) continue;
      double const orientation = hit.normal.dot(direction);
      // An outward surface hit is an exit, not a fresh entry for a particle
      // already assigned to the enclosing air node. Conversely an inward
      // hit must limit an air step even if parity at its start is ambiguous.
      if (logicallyInside ? orientation <= 1e-12 : orientation >= -1e-12) continue;
      TimeType const time = std::max(distance, 1e-9*1_m)/speed;
      if (logicallyInside) return Intersections(0_s, TimeType(time));
      return Intersections(TimeType(time));
    }
    return Intersections();
  }
  static std::string getName() { return "Terrain-LogicalSide-OrientedTracking"; }
  static std::string getVersion() { return "1.0"; }
 private:
  bool legacy_;
  FlatTerrainData const* terrain_{};
  tracking_line::Tracking legacyTracking_;
};


} // namespace corsika::terrain
