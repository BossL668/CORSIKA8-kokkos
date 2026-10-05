#pragma once
#include <corsika/detail/modules/egs4/Egs4Competition.hpp>
#include <corsika/detail/modules/egs4/Egs4Radiative.hpp>
#include <corsika/detail/modules/egs4/Egs4Scattering.hpp>
#include <corsika/detail/modules/egs4/Egs4Termination.hpp>

namespace c7_egs4 {
// Local-physics interface for the B integration. It is not a complete C8
// resident-wavefront backend: no geometry, stack, callbacks or IO here.
struct ModelView {
  TableView tables;
  RadiativeTable radiative;
  ChannelThresholds thresholds;
};
inline ModelView modelView(Tables const& t,ChannelThresholds thresholds) {
  if(!validThresholds(thresholds)||t.medium.ae<=thresholds.mass_MeV)
    throw std::runtime_error("Invalid EGS4 model thresholds");
  return {hostView(t),radiativeTable(t),thresholds};
}
template<class Exec>class DeviceModel {
  DeviceTables<Exec> tables_;
  RadiativeTable radiative_;
  ChannelThresholds thresholds_;
public:
  DeviceModel(Tables const& t,ChannelThresholds thresholds):
    tables_(t),radiative_(radiativeTable(t)),thresholds_(thresholds) {
    if(!validThresholds(thresholds)||t.medium.ae<=thresholds.mass_MeV)
      throw std::runtime_error("Invalid device EGS4 model thresholds");
  }
  ModelView view()const{return {tables_.view(),radiative_,thresholds_};}
};
struct VertexInput {
  int pdg{};
  double energy_MeV{},density_g_cm3{};
  Direction direction;
};
KOKKOS_INLINE_FUNCTION bool validVertex(ModelView model,VertexInput q) {
  return validThresholds(model.thresholds)&&finite(q.energy_MeV)&&q.energy_MeV>0.&&
    finite(q.density_g_cm3)&&q.density_g_cm3>0.&&validDirection(q.direction)&&
    (q.pdg==22||q.pdg==11||q.pdg==-11)&&(q.pdg==22||q.energy_MeV>model.thresholds.mass_MeV);
}
enum class Disposition { error, replace_parent, keep_parent, needs_host_channel };
struct VertexOutcome {
  Disposition disposition{Disposition::error};
  Channel channel{Channel::invalid};
  LocalOutcome local;
};

// Rare non-EM channels are returned explicitly and unchanged to the caller.
// This function does not call any C8/PROPOSAL or Fortran fallback.
template<class Random>
KOKKOS_INLINE_FUNCTION VertexOutcome sampleAtVertex(
    ModelView model,Channel channel,VertexInput q,Random& rng) {
  VertexOutcome r;r.channel=channel;
  double m=model.thresholds.mass_MeV;
  if(!validVertex(model,q))return r;
  switch(channel) {
    case Channel::bremsstrahlung:
      if(q.pdg==22)return r;
      r.local.secondaries=sampleBremsstrahlung(model.radiative,
        {q.energy_MeV,m,q.density_g_cm3,q.direction,q.pdg},rng);break;
    case Channel::pair_production:
      if(q.pdg!=22)return r;
      r.local.secondaries=samplePairProduction(model.radiative,
        {q.energy_MeV,m,q.density_g_cm3,q.direction,q.pdg},rng);break;
    case Channel::compton:
      if(q.pdg!=22)return r;
      r.local.secondaries=sampleCompton({Collision::compton,q.energy_MeV,m,model.tables.medium.ae-m,q.direction},rng);break;
    case Channel::moller:
      if(q.pdg!=11)return r;
      r.local.secondaries=sampleIonization<false>({Collision::moller,q.energy_MeV,m,model.tables.medium.ae-m,q.direction},rng);break;
    case Channel::bhabha:
      if(q.pdg!=-11)return r;
      r.local.secondaries=sampleIonization<true>({Collision::bhabha,q.energy_MeV,m,model.tables.medium.ae-m,q.direction},rng);break;
    case Channel::annihilation:
      if(q.pdg!=-11)return r;
      r.local.secondaries=annihilateInFlight(q.energy_MeV,m,q.direction,rng);break;
    case Channel::photoelectric:
      if(q.pdg!=22)return r;
      r.local=photoelectric(q.energy_MeV,m,model.tables.medium.binding_energy_MeV,q.direction);break;
    case Channel::electronuclear:
      if(q.pdg==22||q.energy_MeV<model.thresholds.electronuclear_MeV)return r;
      r.disposition=Disposition::needs_host_channel;
      r.local.secondaries.status=InteractionStatus::host_required;return r;
    case Channel::photonuclear:
      if(q.pdg!=22||q.energy_MeV<=model.thresholds.photonuclear_MeV)return r;
      r.disposition=Disposition::needs_host_channel;
      r.local.secondaries.status=InteractionStatus::host_required;return r;
    case Channel::muon_pair:
      if(q.pdg!=22||q.energy_MeV<=model.thresholds.muon_pair_MeV)return r;
      r.disposition=Disposition::needs_host_channel;
      r.local.secondaries.status=InteractionStatus::host_required;return r;
    case Channel::continue_transport:
      r.local.secondaries.status=InteractionStatus::suppressed;break;
    default:return r;
  }
  if(r.local.secondaries.status==InteractionStatus::success)r.disposition=Disposition::replace_parent;
  else if(r.local.secondaries.status==InteractionStatus::suppressed)r.disposition=Disposition::keep_parent;
  return r;
}

// Call only AFTER reaching an accepted collision. In particular the
// ELECTR fictitious-sigma rejection must precede this branch draw.
template<class Random>
KOKKOS_INLINE_FUNCTION VertexOutcome selectAndSample(
    ModelView model,VertexInput q,Random& rng) {
  Channel channel=Channel::invalid;double u;
  if(!validVertex(model,q))return {};
  if(q.pdg==22) {
    auto coefficients=photonQuery(model.tables,q.energy_MeV);
    if(coefficients.status!=Status::success)return {};
    if(!uniform(rng,u)){VertexOutcome r;r.local.secondaries.status=InteractionStatus::random_failure;return r;}
    channel=photonChannel(coefficients,q.energy_MeV,model.thresholds,u);
  } else if(q.pdg==11||q.pdg==-11) {
    int charge=q.pdg==11?-1:1;
    auto coefficients=electronQuery(model.tables,q.energy_MeV,model.thresholds.mass_MeV,charge);
    if(coefficients.status!=Status::success)return {};
    if(!uniform(rng,u)){VertexOutcome r;r.local.secondaries.status=InteractionStatus::random_failure;return r;}
    channel=electronChannel(coefficients,charge,q.energy_MeV,model.tables.medium.ae,model.thresholds,u);
  }
  return sampleAtVertex(model,channel,q,rng);
}
} // namespace c7_egs4
