#pragma once
#include "Egs4CurvedFrame.hpp"

namespace c7_egs4 {
// Native HOWFAR for the C7 CURVED/UPWARD or flat/UPWARD variants, with one
// downward-shower observation level. Region numbers retain C7 semantics:
// 1 above air, 2..5 air, 6 below air. UPWARDOLD is deliberately not selected.
// The atmospheric thickness at the current height is supplied by the caller.
struct GeometryConfig {
  bool curved{true},flat_output{true};
  double earth_radius_cm{},bound_cm[6]{},observation_height_cm{};
  double minimum_horizontal_cm{},maximum_horizontal_cm{},horizontal_slope_cm_per_g_cm2{};
};
struct GeometryInput {
  Point local;
  Direction direction;
  int region{2};
  double wa{1.},thickness_g_cm2{},requested_cm{},nearest_cm{};
};
struct GeometryResult {
  bool valid{},energy_discard{};
  int region{},next_observation{1},discard{};
  double step_cm{},nearest_cm{};
};
KOKKOS_INLINE_FUNCTION GeometryResult howFar(GeometryConfig c,GeometryInput q) {
  GeometryResult r;r.region=q.region;r.step_cm=q.requested_cm;r.nearest_cm=q.nearest_cm;
  if(!validPoint(q.local)||!validDirection(q.direction)||q.region<1||q.region>6||
     !finite(q.requested_cm)||q.requested_cm<0.||!finite(q.nearest_cm)||!finite(q.wa)||q.wa<=0.||q.wa>1.||
     !finite(c.earth_radius_cm)||c.earth_radius_cm<=0.||!finite(c.observation_height_cm)||
     !finite(q.thickness_g_cm2)||q.thickness_g_cm2<0.||!finite(c.minimum_horizontal_cm)||
     c.minimum_horizontal_cm<=0.||!finite(c.maximum_horizontal_cm)||c.maximum_horizontal_cm<=0.||
     !finite(c.horizontal_slope_cm_per_g_cm2))return r;
  for(int i=0;i<6;++i)if(!finite(c.bound_cm[i])||(i&&c.bound_cm[i]>=c.bound_cm[i-1]))return r;
  r.valid=true;
  if(q.region==6){r.discard=1;r.energy_discard=true;return r;}
  if(q.region==1) {
    if(q.direction.z>0.){r.step_cm=.0001;r.region=2;}
    else {r.discard=1;r.energy_discard=false;}
    return r;
  }
  int ir=q.region;double z=q.local.z,w=q.direction.z,upper=c.bound_cm[ir-2],lower=c.bound_cm[ir-1];
  double obs=c.observation_height_cm,obsglob=obs,toaxis=0.,aux3=w;
  if(c.curved) {
    double horizontal=larger(q.direction.x*q.direction.x+q.direction.y*q.direction.y,.001);
    double aux=larger(c.minimum_horizontal_cm,c.horizontal_slope_cm_per_g_cm2*q.thickness_g_cm2+c.maximum_horizontal_cm);
    r.step_cm=smaller(r.step_cm,aux/::sqrt(horizontal));
    double xnew=q.local.x+q.direction.x*r.step_cm,ynew=q.local.y+q.direction.y*r.step_cm;
    if(((::fabs(xnew)<::fabs(q.local.x)||xnew*q.local.x<=0.)&&
        (::fabs(ynew)<::fabs(q.local.y)||ynew*q.local.y<=0.))||
       xnew*xnew+ynew*ynew<q.local.x*q.local.x+q.local.y*q.local.y)toaxis=1.;
    else toaxis=-1.;
    if(c.flat_output)obsglob=(c.earth_radius_cm+obs)/q.wa-c.earth_radius_cm;
    aux3=w*q.wa;
    if(::fabs(w)<1.)aux3-=toaxis*::sqrt((1.-w)*(1.+w)*(1.-q.wa)*(1.+q.wa));
  }
  bool down=c.curved?((c.flat_output&&aux3>0.&&ir==5)||w>0.):w>.0003;
  if(down) {
    double distance;bool reaches=true;
    if(c.curved) {
      double radius=c.earth_radius_cm-z,target=larger(obs,lower-.1)+c.earth_radius_cm;
      double impact=::fabs(w)<1.?radius*::sqrt((1.-w)*(1.+w)):0.;
      double sphere;
      if(target>=impact&&w>0.)sphere=radius*w-::sqrt((target+impact)*(target-impact));
      else {sphere=radius*::fabs(w);reaches=false;}
      if(c.flat_output&&aux3>0.) {
        distance=(radius*q.wa-c.earth_radius_cm-obs)/aux3;
        if(distance<=sphere)reaches=true;else distance=sphere;
      } else distance=sphere;
    } else distance=(-z-larger(lower,obs))/w;
    if(distance>r.step_cm) {
      r.nearest_cm=smaller(z+upper,-z-larger(lower,obsglob));
      if(c.curved)r.nearest_cm=smaller(r.nearest_cm,(reaches?distance:r.step_cm)*::fabs(w));
    } else {
      r.step_cm=larger(distance,.0001);
      if(reaches) {
        if(larger(obsglob,lower)>obsglob) {
          r.region=ir+1;
          if(r.region>=6){r.discard=-1;r.energy_discard=true;}
        } else {r.next_observation=2;r.discard=-1;r.energy_discard=true;}
      }
    }
  } else if(c.curved||w<-.003) {
    double distance;
    if(c.curved) {
      if(::fabs(w)<=.003) {
        if(upper>-z)distance=::sqrt((2.*c.earth_radius_cm+upper+.1-z)*(upper+.1+z));
        else {
          // Preserve the original recovery from a stale layer index.
          while(c.bound_cm[r.region-2]<=-z) {
            --r.region;if(r.region<=1){r.discard=1;r.energy_discard=false;return r;}
          }
          double corrected=c.bound_cm[r.region-2];
          distance=::sqrt((2.*c.earth_radius_cm+corrected+.1-z)*(corrected+.1+z));
        }
      } else distance=::fabs((-(upper+.1)-z)/w);
    } else distance=(-upper-z)/w;
    if(distance>r.step_cm)r.nearest_cm=smaller(z+upper,-z-larger(lower,obsglob));
    else {
      r.step_cm=larger(distance,.0001);r.region=ir-1;
      if(r.region<=1){r.discard=-2;r.energy_discard=false;}
    }
  }
  // Flat UPWARD near-horizontal paths retain the requested step and DNEAR.
  if(!finite(r.step_cm)||r.step_cm<0.||!finite(r.nearest_cm))r.valid=false;
  return r;
}
} // namespace c7_egs4
