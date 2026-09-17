#pragma once
#include <corsika/accelerator/em/common/UniformMagneticField.hpp>
#include <corsika/geometry/terrain/TerrainMagneticAccuracy.hpp>

namespace corsika::interfaces::detail {
// Independent interface policy: avoid B^2-(u.B)^2 cancellation for directions
// parallel to B. Preserve the scalar 1 eV/c and 1e9 m linear-track decisions.
KOKKOS_INLINE_FUNCTION gpu::em::MagneticStepLimitResult maximumInterfaceMagneticStep(
    gpu::em::EmParticleState const& particle,double mass,double charge,double const field[3],
    double requestedAngle=terrain::MaximumQuadraticMagneticDeflection) {
  namespace em=gpu::em;
  namespace m=em::magnetic_detail;
  if(!m::validParticle(particle,mass,charge)||!m::validField(field)||
     !m::finite(requestedAngle)||!(requestedAngle>0.))return {};
  double field2=m::dot(field,field);
  if(charge==0.||field2==0.)return {em::MagneticStepStatus::Linear,0,m::infinity(),m::infinity()};
  double transverse[3];m::cross(particle.direction,field,transverse);
  double momentum=m::squareRoot((particle.energy_GeV-mass)*(particle.energy_GeV+mass));
  double perpendicular=momentum*m::squareRoot(m::dot(transverse,transverse)/field2);
  if(perpendicular<1.e-9)return {em::MagneticStepStatus::Linear,0,m::infinity(),m::infinity()};
  double radius=perpendicular/(em::GeVPerCToTeslaMeter*m::absolute(charge)*m::squareRoot(field2));
  if(!m::finite(radius)||radius>1.e9)return {em::MagneticStepStatus::Linear,0,m::infinity(),radius};
  double angle=requestedAngle<terrain::MaximumQuadraticMagneticDeflection?requestedAngle:terrain::MaximumQuadraticMagneticDeflection;
  double length=2*m::cosine(angle)*m::sine(angle)*radius;
  if(!m::finite(length)||!(length>0.))return {em::MagneticStepStatus::NonFiniteResult,0,0.,radius};
  return {em::MagneticStepStatus::Success,0,length,radius};
}
}
