// PSR-only oracle: actual scalar PROPOSAL versus portable LPM snapshots,
// using the same resolved material construction as the mountain application.
#include <corsika/modules/transport/MaterialProposalEnvironment.hpp>
#include <corsika/modules/terrain/TerrainAtmosphere.hpp>
#include <corsika/modules/PROPOSAL.hpp>
#include <corsika/accelerator/em/common/tables/ProposalNativeAux.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace corsika;
using namespace corsika::units::si;
struct NoHadronic {
  template<class... Args> void doInteraction(Args&&...) {throw std::runtime_error("not a shower");}
  bool isValid(Code,Code,HEPEnergyType) const {return true;}
};
int main(int argc,char** argv) {
  try {
    if(argc!=4) throw std::runtime_error("usage: MaterialOracle PRESET AUX_CACHE CSV");
    logging::set_level(logging::level::warn);
    RNGManager<>::getInstance().registerRandomStream("proposal");
    for(auto p:proposal::tracked) set_energy_production_threshold(p,100_MeV);
    auto m=interfaces::materialPreset(argv[1]); auto expected=m.transportProperties();
    terrain::Environment env; auto cs=env.getCoordinateSystem();
    auto world=env.createNode<Sphere>(Point(cs,0_m,0_m,0_m),1000_m);
    world->setModelProperties(m.makeModel<terrain::Interface>(cs)); env.getUniverse()->addChild(std::move(world));
    interfaces::MaterialProposalEnvironment view(env); NoHadronic had;
    proposal::Interaction interaction(view,had,had,80_GeV);
    auto views=interaction.nativeCalculatorViews();
    auto aux=gpu::em::tables::loadOrCreateProposalNativeAux(views,argv[2]);
    std::ofstream csv(argv[3]); if(!csv) throw std::runtime_error("cannot create oracle CSV");
    csv<<"material,process,Z,A,energy_MeV,fraction,density_scale,native,portable,relative_error\n"<<std::setprecision(17);
    double maximum=0.; std::size_t count=0;
    for(auto const& v:views) {
      if(v.projectile!=Code::Photon && v.projectile!=Code::Electron) continue;
      auto const& medium=*v.medium;
      if(medium.GetI()!=expected.Ieff_ || medium.GetMassDensity()!=expected.corrected_density_ ||
         medium.GetNumComponents()!=int(m.nuclei.size()) ||
         std::abs(medium.GetZA()-m.ZOverA())>1.e-14)
        throw std::runtime_error("scalar calculator did not receive selected material");
      for(auto const& component:medium.GetComponents())
        for(double energy:{1.e3,1.e6,1.e9,1.e11}) for(double f:{.01,.1,.5,.9,.99}) for(double scale:{.5,1.,2.}) {
          double native=0.,portable=0.; bool ok=false;
          if(v.projectile==Code::Photon) {
            if(!v.photon_pair_lpm) throw std::runtime_error("missing pair LPM");
            native=v.photon_pair_lpm->suppression_factor(energy,f,component,scale);
            auto result=gpu::em::photonPairLpmSuppressionFactor(aux.photon_pair_lpm,component.GetHash(),energy,f,scale*medium.GetMassDensity());
            portable=result.survival_probability; ok=result.status==gpu::em::PhotonPairLpmStatus::Success;
          } else {
            if(!v.brems_lpm) throw std::runtime_error("missing brems LPM");
            native=v.brems_lpm->suppression_factor(energy,f,component,scale);
            auto result=gpu::em::bremsLpmSuppressionFactor(aux.brems_lpm,component.GetHash(),energy,f,scale*medium.GetMassDensity());
            portable=result.survival_probability; ok=result.status==gpu::em::BremsLpmStatus::Success;
          }
          double error=std::abs(native-portable)/std::max(1.e-30,std::abs(native));
          if(!ok || !std::isfinite(error) || error>2.e-10) throw std::runtime_error("material LPM oracle mismatch");
          maximum=std::max(maximum,error); ++count;
          csv<<m.id<<','<<(v.projectile==Code::Photon?"pair":"brems")<<','<<component.GetNucCharge()<<','<<component.GetAtomicNum()
             <<','<<energy<<','<<f<<','<<scale<<','<<native<<','<<portable<<','<<error<<'\n';
        }
    }
    if(!count || !csv) throw std::runtime_error("empty/incomplete oracle output");
    std::cout<<m.id<<" points="<<count<<" maximum_relative_error="<<std::setprecision(17)<<maximum<<'\n';
    return 0;
  } catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 1;}
}
