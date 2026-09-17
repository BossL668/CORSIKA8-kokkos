/* Material-aware heavy charged-particle collision loss. Adapted from the
 * CORSIKA BetheBlochPDG process (BSD-3-Clause), with the fixed nitrogen record
 * and its empirical air radiative term removed for embedded materials. */
#pragma once
#include <corsika/modules/transport/MaterialModel.hpp>
#include <corsika/modules/energy_loss/BetheBlochPDG.hpp>

namespace corsika::interfaces {
inline double densityEffect(MediumData const& d,double betaGamma) {
  double x=std::log10(betaGamma);
  if(x>=d.x1_) return 2.*std::log(10.)*x-d.Cbar_;
  if(x>=d.x0_) return 2.*std::log(10.)*x-d.Cbar_+d.aa_*std::pow(d.x1_-x,d.sk_);
  return d.dlt0_*std::pow(10.,2.*(x-d.x0_));
}
// Positive stopping power in MeV cm2/g. Leading Bethe stopping with Sternheimer
// density effect; no claim of low-energy shell/Barkas/effective-ion-charge physics.
inline double hadronStoppingPower(MediumData const& d,double mass_MeV,
                                 double totalEnergy_MeV,int charge) {
  if(charge==0) return 0.;
  if(!(mass_MeV>0. && totalEnergy_MeV>mass_MeV))
    throw std::invalid_argument("Bethe stopping requires a moving massive charged particle");
  double gamma=totalEnergy_MeV/mass_MeV, bg2=(gamma-1.)*(gamma+1.);
  double beta2=bg2/(gamma*gamma), me=Electron::mass/1_MeV;
  double ratio=me/mass_MeV;
  double tmax=2.*me*bg2/(1.+2.*gamma*ratio+ratio*ratio);
  double I=d.Ieff_*1.e-6;
  double bracket=.5*std::log(2.*me*bg2*tmax/(I*I))-beta2-.5*densityEffect(d,std::sqrt(bg2));
  double loss=.307075*double(charge)*charge*d.Z_over_A_/beta2*bracket;
  if(!std::isfinite(loss) || loss<=0.)
    throw std::runtime_error("material Bethe approximation outside validity; raise the hadron cut or provide a low-energy loss model");
  return loss;
}

template<class Output=WriterOff>
class MaterialHadronLoss : public ContinuousProcess<MaterialHadronLoss<Output>>, public Output {
  BetheBlochPDG<Output> air_;
  template<class Particle> static bool usesCard(Particle const& p) {
    return dynamic_cast<MaterialTransportProvider const*>(&p.getNode()->getModelProperties());
  }
public:
  template<class... Args> explicit MaterialHadronLoss(Args&&... args)
      : Output(args...),air_(args...) {}
  template<class Particle> static auto stopping(Particle const& p) {
    return hadronStoppingPower(transportData(p.getNode()->getModelProperties()),
        p.getMass()/1_MeV,p.getEnergy()/1_MeV,p.getChargeNumber())*1_MeV*square(1_cm)/1_g;
  }
  template<class Particle> ProcessReturn doContinuous(Step<Particle>& step,bool limit) {
    auto const& p=step.getParticlePre();
    if(!usesCard(p)) return air_.doContinuous(step,limit);
    if(p.getChargeNumber()==0) return ProcessReturn::Ok;
    auto grammage=p.getNode()->getModelProperties().getIntegratedGrammage(step.getStraightTrack());
    auto loss=std::min(step.getEkinPre(),stopping(p)*grammage);
    step.add_dEkin(-loss);
    Output::write(step.getPositionPre(),step.getPositionPost(),p.getPID(),p.getWeight()*loss);
    return ProcessReturn::Ok;
  }
  template<class Particle,class Track> LengthType getMaxStepLength(Particle const& p,Track const& track) const {
    if(!usesCard(p)) return air_.getMaxStepLength(p,track);
    if(p.getChargeNumber()==0) return std::numeric_limits<double>::infinity()*1_m;
    auto energy=p.getKineticEnergy();
    auto target=std::max(.9*energy,get_kinetic_energy_propagation_threshold(p.getPID())*.99999);
    auto grammage=std::max(energy-target,HEPEnergyType::zero())/stopping(p);
    return p.getNode()->getModelProperties().getArclengthFromGrammage(track,grammage);
  }
  YAML::Node getConfig() const override {
    YAML::Node n; n["type"]="MaterialHadronLoss";
    n["embedded"]="leading Bethe + material Sternheimer; no empirical air radiative term";
    n["outside"]="original BetheBlochPDG"; return n;
  }
};
} // namespace corsika::interfaces
