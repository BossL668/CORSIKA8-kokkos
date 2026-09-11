/* Mountain numerical/RNG adapter for the unmodified TAUOLA library.
 * TAUOLA 1.1.8 boostAlongZ reconstructs m from E^2-p^2, which vanishes at
 * extreme gamma. Generate at a well-conditioned momentum and boost the
 * daughters with CORSIKA's known-mass COMBoost instead. No decay is retried.
 * Stable-particle masses are aligned for the duration of this call only;
 * the stock air-shower module and the library defaults are not modified.
 */
#pragma once
#include <corsika/modules/TAUOLA.hpp>
#include <corsika/modules/neutrino/TauolaMassConvention.hpp>
#include <corsika/framework/utility/COMBoost.hpp>
#include <corsika/framework/core/EnergyMomentumOperations.hpp>
#include <corsika/framework/random/RNGManager.hpp>
#include <random>

namespace corsika::neutrino {

class ConditionedTauolaDecay {
 public:
  explicit ConditionedTauolaDecay(tauola::Helicity h) : native_(h) {
    if (!randomInitialized_) {
      // TAUOLA has two distinct RNGs. The C++ acceptance generator is a
      // callback; the Fortran decay kernel is RANMAR. Seed RANMAR ONCE from
      // its own registered CORSIKA stream, not once per particle/CC vertex.
      auto& rng=RNGManager<>::getInstance().getRandomStream("tauola");
      fortranSeed_=std::uniform_int_distribution<int>(0,900000000)(rng);
      int zero=0;
      Tauolapp::rmarin_(&fortranSeed_,&zero,&zero);
      Tauolapp::Tauola::setRandomGenerator(&drawUniform);
      randomInitialized_=true;
    }
  }
  template<class Particle> TimeType getLifetime(Particle const& p) {
    return native_.getLifetime(p);
  }
  template<class View> void doDecay(View& view) {
    // Align before generating the decay, not by rescaling its daughters.
    // In particular the legacy AMNUTA=0.010 GeV must not be converted to a
    // massless CORSIKA daughter in the reference frame and then highly boosted.
    // The guard restores all changed common-block fields, also on exceptions.
    ScopedTauolaMassConvention masses;
    auto p=view.getProjectile();
    if (p.getEnergy() < 1000_GeV) {
      native_.doDecay(view);
    } else {
      ConditionedView<View> conditioned(view);
      native_.doDecay(conditioned);
    }
  }
  static int fortranSeed() noexcept { return fortranSeed_; }
 private:
  static double drawUniform() {
    return std::uniform_real_distribution<double>(0.,1.)(
        RNGManager<>::getInstance().getRandomStream("tauola"));
  }

  template<class View> struct ConditionedView {
    explicit ConditionedView(View& v)
        : destination(v), pid(v.getProjectile().getPID()),
          mass(tauola::TauolaInterfaceParticle::findMassTauola(pid)),
          labBoost(v.getProjectile().getMomentum(),get_mass(pid)),
          referenceMomentum(labBoost.getRotatedCS(),{0_GeV,0_GeV,mass}),
          referenceBoost(referenceMomentum,mass) {}

    struct Projectile {
      ConditionedView* owner;
      Code getPID() const { return owner->pid; }
      MomentumVector getMomentum() const { return owner->referenceMomentum; }
      template<class Tuple> void addSecondary(Tuple const& values) {
        auto const& [pid,ekin,direction]=values;
        auto const m=get_mass(pid);
        MomentumVector momentum=direction*sqrt(ekin*(ekin+2.*m));
        auto rest=owner->referenceBoost.toCoM(FourMomentum{ekin+m,momentum});
        auto lab=owner->labBoost.fromCoM(rest);
        auto p=lab.getSpaceLikeComponents();
        auto norm=p.getNorm();
        if (!(norm>0_GeV) || !std::isfinite(norm/1_GeV))
          throw std::runtime_error("nonfinite/zero TAUOLA daughter after conditioned boost");
        owner->destination.addSecondary(std::make_tuple(pid,calculate_kinetic_energy(norm,m),p.normalized()));
      }
    };
    Projectile getProjectile() { return {this}; }
    View& destination;
    Code pid;
    HEPMassType mass;
    COMBoost labBoost;
    MomentumVector referenceMomentum;
    COMBoost referenceBoost;
  };
  tauola::Decay native_;
  inline static bool randomInitialized_{false};
  inline static int fortranSeed_{0};
};
} // namespace corsika::neutrino
