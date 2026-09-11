/* Host material construction independent of shape, site and adjoining medium.
 * A new composition must also have appropriate CORSIKA medium properties; a
 * density or refractive-index change is not a chemical-composition change. */
#pragma once
#include <corsika/media/HomogeneousMedium.hpp>
#include <corsika/media/MediumPropertyModel.hpp>
#include <corsika/media/UniformRefractiveIndex.hpp>
#include <corsika/media/UniformMagneticField.hpp>
#include <corsika/framework/core/ParticleProperties.hpp>
#include <array>
#include <cmath>
#include <memory>
#include <map>
#include <set>
#include <stdexcept>

namespace corsika::interfaces {
template<class Environment> std::set<Code> collectNuclearTargets(Environment const& env) {
  std::set<Code> targets;
  env.getUniverse()->walk([&](auto const& node) {
    if(node.hasModelProperties()) {
      auto const& nuclei=node.getModelProperties().getNuclearComposition().getComponents();
      targets.insert(nuclei.begin(),nuclei.end());
    }
  });
  if(targets.empty())throw std::invalid_argument("no physical material targets in environment");
  return targets;
}
// Stock PROPOSAL wrappers key calculators by nuclear composition, not region.
// Prevent e.g. water and ice with different Sternheimer data silently sharing
// the last inserted calculator. Distinct per-material calculator owners may
// instead be supplied directly to prepareEmMaterial by another application.
template<class Environment> void validateCalculatorMaterialKeys(Environment const& env) {
  std::map<std::size_t,Medium> seen;
  env.getUniverse()->walk([&](auto const& node) {
    if(!node.hasModelProperties())return;
    auto const& p=node.getModelProperties();auto hash=p.getNuclearComposition().getHash();
    auto [it,inserted]=seen.emplace(hash,p.getMedium());
    if(!inserted&&it->second!=p.getMedium())
      throw std::invalid_argument("same nuclear composition with different medium properties requires independent PROPOSAL calculator owners");
  });
}
struct HomogeneousMaterial {
  Medium medium;
  NuclearComposition composition;
  double density_g_cm3, refractive_index;
  std::array<double,3> magnetic_field_T{};

  template<class Interface>
  auto makeModel(CoordinateSystemPtr const& cs) const {
    using namespace units::si;
    if(!std::isfinite(density_g_cm3)||density_g_cm3<=0.||
       !std::isfinite(refractive_index)||refractive_index<1.)
      throw std::invalid_argument("invalid homogeneous interface material");
    for(double b:magnetic_field_T)if(!std::isfinite(b))
      throw std::invalid_argument("nonfinite interface magnetic field");
    using Model=UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<HomogeneousMedium<Interface>>>>;
    return std::make_unique<Model>(refractive_index,medium,
        MagneticFieldVector(cs,magnetic_field_T[0]*1_T,magnetic_field_T[1]*1_T,magnetic_field_T[2]*1_T),
        density_g_cm3*1_g/(1_cm*1_cm*1_cm),composition);
  }
};
} // namespace corsika::interfaces
