#pragma once
#include <corsika/detail/modules/egs4/Egs4Transport.hpp>

namespace c7_egs4 {
// C7 CURVED normal-step rebasing, after the magnetic displacement and BEFORE
// the MSCAT azimuth rotation. Coordinates are the original EGS surface-arc
// coordinates (cm), not Cartesian C8 positions. Observer positions are kept
// separately; using surface arcs as Cartesian radio endpoints is incorrect.
enum class CurvedStatus { invalid, advanced, angle_cut, observation };
struct CurvedInput {
  bool photon{},detector_local{},flat_output{true};
  Point previous_local,previous_observer,advanced_local;
  Direction advanced_direction;
  double earth_radius_cm{},observation_height_cm{},minimum_cosine{-1.};
  double geometric_step_cm{},time_path_cm{},total_energy_MeV{},mass_MeV{.51099895};
  double speed_cm_s{2.99792458e10};
  bool direct_azimuth{};
};
struct CurvedResult {
  CurvedStatus status{CurvedStatus::invalid};
  Point local,observer;
  Direction direction;
  double wa{1.},wap{},time_increment_s{},observer_to_local_distance_ratio{};
};
KOKKOS_INLINE_FUNCTION bool validPoint(Point p){return finite(p.x)&&finite(p.y)&&finite(p.z);}
KOKKOS_INLINE_FUNCTION CurvedResult rebaseCurvedStep(CurvedInput q) {
  CurvedResult r;r.local=q.advanced_local;r.direction=q.advanced_direction;
  if(!validPoint(q.previous_local)||!validPoint(q.previous_observer)||!validPoint(q.advanced_local)||
     !validDirection(q.advanced_direction)||!finite(q.earth_radius_cm)||q.earth_radius_cm<=0.||
     q.advanced_local.z>=q.earth_radius_cm||!finite(q.observation_height_cm)||
     !finite(q.minimum_cosine)||q.minimum_cosine < -1.||q.minimum_cosine>1.||
     !finite(q.geometric_step_cm)||q.geometric_step_cm<=0.||!finite(q.time_path_cm)||q.time_path_cm<0.||
     !finite(q.speed_cm_s)||q.speed_cm_s<=0.||!finite(q.total_energy_MeV)||q.total_energy_MeV<=0.||
     (!q.photon&&(!finite(q.mass_MeV)||q.mass_MeV<=0.||q.total_energy_MeV<=q.mass_MeV)))return r;
  if(!q.detector_local||!q.flat_output) {
    // ELECTR clamps W before the rebase; PHOTON does not.
    if(!q.photon)r.direction.z=larger(-1.,smaller(1.,r.direction.z));
    double dx=r.local.x-q.previous_local.x,dy=r.local.y-q.previous_local.y;
    double trans2=larger(dx*dx+dy*dy,q.photon?.00001:.0001);
    double radial_z=q.earth_radius_cm-r.local.z;
    double aux=::sqrt(trans2+radial_z*radial_z);
    double znew=q.earth_radius_cm-aux;
    double sindif=::sqrt(trans2)/aux,cosdif=smaller(1.,radial_z/aux);
    double corr=q.earth_radius_cm*::asin(sindif)/((q.earth_radius_cm-znew)*sindif);
    r.local.x=q.previous_local.x+dx*corr;r.local.y=q.previous_local.y+dy*corr;r.local.z=znew;
    double w=r.direction.z*cosdif;
    if(::fabs(r.direction.z)<1.)w-=sindif*::sqrt((1.-r.direction.z)*(1.+r.direction.z));
    r.direction.z=larger(-1.,smaller(1.,w));
    if(r.direction.z<q.minimum_cosine){r.status=CurvedStatus::angle_cut;return r;}
    double lateral=::sqrt(r.local.x*r.local.x+r.local.y*r.local.y);
    r.wa=smaller(1.,::cos(lateral/q.earth_radius_cm));
    double radial=q.earth_radius_cm-znew;
    r.observer.z=q.earth_radius_cm-radial*r.wa;
    double aux2=::sqrt(radial*radial*(1.-r.wa)*(1.+r.wa)+
      (-r.observer.z-q.observation_height_cm)*(-r.observer.z-q.observation_height_cm));
    r.wap=aux2>0.?-(q.observation_height_cm+r.observer.z)/aux2:0.;
    // Original ELECTR branches to its observation endpoint before normal
    // time/rotation handling. Caller must handle this disposition explicitly.
    if(!q.photon&&q.flat_output&&-r.observer.z<=q.observation_height_cm) {
      r.status=CurvedStatus::observation;return r;
    }
    r.wap=smaller(1.,r.wap);
    if(r.direction.x!=0.) {
      double tanphi=r.direction.y/r.direction.x;
      r.direction.x=::fabs(r.direction.z)<1.?::copysign(1.,r.direction.x)*
        ::sqrt((1.-r.direction.z)*(1.+r.direction.z)/(1.+tanphi*tanphi)):0.;
      r.direction.y=tanphi*r.direction.x;
    } else if(r.direction.y!=0.&&::fabs(r.direction.z)<1.) {
      r.direction.y=::copysign(1.,r.direction.y)*::sqrt((1.-r.direction.z)*(1.+r.direction.z));
    } else r.direction.y=0.;
    if(r.wa!=1.) {
      // The local-to-observer patch is only valid in the detector hemisphere.
      if(r.wa<=0.)return r;
      double cp,sp;
      if(q.direct_azimuth&&lateral>0.) {cp=r.local.x/lateral;sp=r.local.y/lateral;}
      else {double phi=(r.local.x!=0.||r.local.y!=0.)?::atan2(r.local.y,r.local.x):0.;cp=::cos(phi);sp=::sin(phi);}
      double radius=::sqrt((1.-r.wa)*(1.+r.wa))*(q.earth_radius_cm-r.observer.z)/r.wa;
      r.observer.x=radius*cp;r.observer.y=radius*sp;
    } else {r.observer.x=r.local.x;r.observer.y=r.local.y;}
  } else r.observer=r.local;
  double time=q.time_path_cm*(1./q.speed_cm_s);
  if(!q.photon)time/=::sqrt((1.-q.mass_MeV/q.total_energy_MeV)*(1.+q.mass_MeV/q.total_energy_MeV));
  double ratio=1.;
  if(q.geometric_step_cm>1.e-10) {
    double dx=r.observer.x-q.previous_observer.x,dy=r.observer.y-q.previous_observer.y,dz=r.observer.z-q.previous_observer.z;
    ratio=::sqrt(dx*dx+dy*dy+dz*dz)/q.geometric_step_cm;
  }
  r.time_increment_s=time*ratio;r.observer_to_local_distance_ratio=ratio;
  if(!validPoint(r.local)||!validPoint(r.observer)||!validDirection(r.direction)||
     !finite(r.time_increment_s)||r.time_increment_s<0.||!finite(ratio))return r;
  r.status=CurvedStatus::advanced;return r;
}
} // namespace c7_egs4
