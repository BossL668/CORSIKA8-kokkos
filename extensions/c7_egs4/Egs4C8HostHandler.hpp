#pragma once
#include "Egs4C8Session.hpp"
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/accelerator/em/common/RouterParticleConversion.hpp>
#include <corsika/framework/geometry/FourVector.hpp>
#include <corsika/framework/process/ProcessReturn.hpp>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace c7_egs4::application {
struct HostStatistics {
  std::uint64_t transported_hadrons{},transported_muons{},selected_photonuclear{};
  std::uint64_t generated_secondaries{},retained_secondaries{},arrival_cuts{};
};

// The supplied photon model is C8's HadronicPhotonModel, NOT its PROPOSAL
// interaction selector. EGS4 has already selected the vertex and air nucleus.
// C8 may select a nucleon inside that nucleus for its LE model, exactly as in
// its existing implementation. No EM rate, free path or branch is resampled.
// Arrival is an explicit application hook: bool(particle, request). It handles
// host transport cuts/output for already-generated native daughters; returning
// false asks this handler to erase the particle. It must not erase it itself.
// C7 polarization scalars are available in request but are not C8 global spin.
template<class Stack,class Environment,class PhotonModel,class Secondaries,class Arrival>
class HostHandler {
  Environment const& environment_;
  ::corsika::CoordinateSystemPtr coordinates_;
  PhotonModel& photons_;
  Secondaries& secondaries_;
  Arrival& arrival_;
  HostStatistics statistics_{};
  std::unordered_set<std::uint64_t> consumed_;
  bool failed_{};
public:
  HostHandler(Environment const& environment,::corsika::CoordinateSystemPtr coordinates,
      PhotonModel& photons,Secondaries& secondaries,Arrival& arrival)
      :environment_(environment),coordinates_(std::move(coordinates)),photons_(photons),
       secondaries_(secondaries),arrival_(arrival) {
    if(!coordinates_)throw std::invalid_argument("Missing EGS4 host coordinate system");
  }
  bool canHandle(HostRequest const& r)const {
    using namespace ::corsika;
    auto const& p=r.particle;
    if(failed_||!p.history_id||consumed_.count(p.history_id)||!std::isfinite(p.energy_GeV)||
        !std::isfinite(p.weight)||p.weight<=0.||!std::isfinite(p.time_s))return false;
    double n=0.;for(int i=0;i<3;++i) {
      if(!std::isfinite(p.position_m[i])||!std::isfinite(p.direction[i]))return false;
      n+=p.direction[i]*p.direction[i];
    }
    if(std::abs(n-1.)>1.e-8)return false;
    try {
      auto code=convert_from_PDG(static_cast<PDGCode>(p.pid));
      if(p.energy_GeV<get_mass(code)/1_GeV)return false;
      if(r.kind==HostRequestKind::muon_transport)return code==Code::MuMinus||code==Code::MuPlus;
      if(r.kind==HostRequestKind::hadron_transport)return is_hadron(code);
      if(r.kind==HostRequestKind::photonuclear_many_hadrons) {
        // Current native SDPM selector supports the actual AIR-NTP nuclei only.
        int z=r.target_pdg==1000070140?7:r.target_pdg==1000080160?8:r.target_pdg==1000180400?18:0;
        return code==Code::Photon&&r.photonuclear_branch==4&&z!=0&&r.target_atomic_number==z;
      }
    }catch(std::exception const&){return false;}
    // Unresolved PIGEN/RHO/RESDEC is not silently substituted with a C8 decay.
    return false;
  }
  std::size_t handle(Stack& stack,HostRequest const& r) {
    using namespace ::corsika;
    if(!canHandle(r))throw std::invalid_argument("Unsupported or duplicate EGS4 host request");
    // A generator/output exception can leave a partial stack; prohibit replay
    // instead of generating the same selected vertex twice.
    try {
      auto p=gpu::em::router_detail::importParticle(stack,r.particle,coordinates_);
      auto node=environment_.getUniverse()->getContainingNode(p.getPosition());
      if(!node||!node->hasModelProperties()) {
        p.erase();throw std::runtime_error("EGS4 host vertex is outside the C8 medium");
      }
      p.setNode(node);
      std::size_t returned=0;
      if(r.kind==HostRequestKind::photonuclear_many_hadrons) {
        typename Stack::stack_view_type view(p);
        auto target=convert_from_PDG(static_cast<PDGCode>(r.target_pdg));
        auto E=r.particle.energy_GeV*1_GeV;
        FourMomentum photon(E,DirectionVector{coordinates_,
          {r.particle.direction[0],r.particle.direction[1],r.particle.direction[2]}}*E);
        auto status=photons_.doHadronicPhotonInteraction(view,coordinates_,photon,target);
        if(status!=ProcessReturn::Ok&&status!=ProcessReturn::Interacted)
          throw std::runtime_error("C8 selected photonuclear final state did not complete");
        statistics_.generated_secondaries+=view.getSize();
        secondaries_.doSecondaries(view);
        returned=view.getSize();statistics_.retained_secondaries+=returned;
        p.erase();++statistics_.selected_photonuclear;
      } else {
        bool keep=arrival_(p,r);
        if(keep) {
          returned=1;
          if(r.kind==HostRequestKind::muon_transport)++statistics_.transported_muons;
          else ++statistics_.transported_hadrons;
        } else {p.erase();++statistics_.arrival_cuts;}
      }
      consumed_.insert(r.particle.history_id);
      return returned;
    }catch(...) {failed_=true;throw;}
  }
  HostStatistics const& statistics()const{return statistics_;}
};
} // namespace c7_egs4::application
