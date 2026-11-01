/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <corsika/detail/kokkos/egs4/Egs4Competition.inl>
#include <corsika/detail/kokkos/egs4/Egs4Radiative.inl>
#include <corsika/detail/kokkos/egs4/Egs4Scattering.inl>
#include <corsika/detail/kokkos/egs4/Egs4Termination.inl>

namespace corsika::kokkos::egs4 {
  // Local-physics interface for the EGS4 integration. It is not a complete C8
  // resident-wavefront backend: no geometry, stack, callbacks or IO here.
  struct ModelView {
    TableView tables_;
    RadiativeTable radiative_;
    ChannelThresholds thresholds_;
  };
  inline ModelView model_view(Tables const& t, ChannelThresholds thresholds) {
    if (!valid_thresholds(thresholds) || t.medium_.ae_ <= thresholds.massMeV_)
      throw std::runtime_error("Invalid EGS4 model thresholds");
    return {host_view(t), radiative_table(t), thresholds};
  }
  template <class TExec>
  class DeviceModel {
    DeviceTables<TExec> tables_;
    RadiativeTable radiative_;
    ChannelThresholds thresholds_;

  public:
    DeviceModel(Tables const& t, ChannelThresholds thresholds)
        : tables_(t)
        , radiative_(radiative_table(t))
        , thresholds_(thresholds) {
      if (!valid_thresholds(thresholds) || t.medium_.ae_ <= thresholds.massMeV_)
        throw std::runtime_error("Invalid device EGS4 model thresholds");
    }
    ModelView view() const { return {tables_.view(), radiative_, thresholds_}; }
  };
  struct VertexInput {
    int pdg_{};
    double energyMeV_{}, densityGCm3_{};
    Direction direction_;
  };
  KOKKOS_INLINE_FUNCTION bool valid_vertex(ModelView model, VertexInput q) {
    return valid_thresholds(model.thresholds_) && finite(q.energyMeV_) &&
           q.energyMeV_ > 0. && finite(q.densityGCm3_) && q.densityGCm3_ > 0. &&
           valid_direction(q.direction_) &&
           (q.pdg_ == 22 || q.pdg_ == 11 || q.pdg_ == -11) &&
           (q.pdg_ == 22 || q.energyMeV_ > model.thresholds_.massMeV_);
  }
  enum class Disposition { error, replace_parent, keep_parent, needs_host_channel };
  struct VertexOutcome {
    Disposition disposition_{Disposition::error};
    Channel channel_{Channel::invalid};
    LocalOutcome local_;
  };

  // Rare non-EM channels are returned explicitly and unchanged to the caller.
  // This function does not call any C8/PROPOSAL or Fortran fallback.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION VertexOutcome sample_at_vertex(ModelView model, Channel channel,
                                                        VertexInput q, TRandom& rng) {
    VertexOutcome r;
    r.channel_ = channel;
    double m = model.thresholds_.massMeV_;
    if (!valid_vertex(model, q)) return r;
    switch (channel) {
      case Channel::bremsstrahlung:
        if (q.pdg_ == 22) return r;
        r.local_.secondaries_ = sample_bremsstrahlung(
            model.radiative_, {q.energyMeV_, m, q.densityGCm3_, q.direction_, q.pdg_},
            rng);
        break;
      case Channel::pair_production:
        if (q.pdg_ != 22) return r;
        r.local_.secondaries_ = sample_pair_production(
            model.radiative_, {q.energyMeV_, m, q.densityGCm3_, q.direction_, q.pdg_},
            rng);
        break;
      case Channel::compton:
        if (q.pdg_ != 22) return r;
        r.local_.secondaries_ =
            sample_compton({Collision::compton, q.energyMeV_, m,
                            model.tables_.medium_.ae_ - m, q.direction_},
                           rng);
        break;
      case Channel::moller:
        if (q.pdg_ != 11) return r;
        r.local_.secondaries_ =
            sample_ionization<false>({Collision::moller, q.energyMeV_, m,
                                      model.tables_.medium_.ae_ - m, q.direction_},
                                     rng);
        break;
      case Channel::bhabha:
        if (q.pdg_ != -11) return r;
        r.local_.secondaries_ =
            sample_ionization<true>({Collision::bhabha, q.energyMeV_, m,
                                     model.tables_.medium_.ae_ - m, q.direction_},
                                    rng);
        break;
      case Channel::annihilation:
        if (q.pdg_ != -11) return r;
        r.local_.secondaries_ = annihilate_in_flight(q.energyMeV_, m, q.direction_, rng);
        break;
      case Channel::photoelectric:
        if (q.pdg_ != 22) return r;
        r.local_ = photoelectric(q.energyMeV_, m, model.tables_.medium_.bindingEnergyMeV_,
                                 q.direction_);
        break;
      case Channel::electronuclear:
        if (q.pdg_ == 22 || q.energyMeV_ < model.thresholds_.electronuclearMeV_) return r;
        r.disposition_ = Disposition::needs_host_channel;
        r.local_.secondaries_.status_ = InteractionStatus::host_required;
        return r;
      case Channel::photonuclear:
        if (q.pdg_ != 22 || q.energyMeV_ <= model.thresholds_.photonuclearMeV_) return r;
        r.disposition_ = Disposition::needs_host_channel;
        r.local_.secondaries_.status_ = InteractionStatus::host_required;
        return r;
      case Channel::muon_pair:
        if (q.pdg_ != 22 || q.energyMeV_ <= model.thresholds_.muonPairMeV_) return r;
        r.disposition_ = Disposition::needs_host_channel;
        r.local_.secondaries_.status_ = InteractionStatus::host_required;
        return r;
      case Channel::continue_transport:
        r.local_.secondaries_.status_ = InteractionStatus::suppressed;
        break;
      default:
        return r;
    }
    if (r.local_.secondaries_.status_ == InteractionStatus::success)
      r.disposition_ = Disposition::replace_parent;
    else if (r.local_.secondaries_.status_ == InteractionStatus::suppressed)
      r.disposition_ = Disposition::keep_parent;
    return r;
  }

  // Call only AFTER reaching an accepted collision. In particular the
  // ELECTR fictitious-sigma rejection must precede this branch draw.
  template <class TRandom>
  KOKKOS_INLINE_FUNCTION VertexOutcome select_and_sample(ModelView model, VertexInput q,
                                                         TRandom& rng) {
    Channel channel = Channel::invalid;
    double u;
    if (!valid_vertex(model, q)) return {};
    if (q.pdg_ == 22) {
      auto coefficients = photon_query(model.tables_, q.energyMeV_);
      if (coefficients.status_ != Status::success) return {};
      if (!uniform(rng, u)) {
        VertexOutcome r;
        r.local_.secondaries_.status_ = InteractionStatus::random_failure;
        return r;
      }
      channel = photon_channel(coefficients, q.energyMeV_, model.thresholds_, u);
    } else if (q.pdg_ == 11 || q.pdg_ == -11) {
      int charge = q.pdg_ == 11 ? -1 : 1;
      auto coefficients =
          electron_query(model.tables_, q.energyMeV_, model.thresholds_.massMeV_, charge);
      if (coefficients.status_ != Status::success) return {};
      if (!uniform(rng, u)) {
        VertexOutcome r;
        r.local_.secondaries_.status_ = InteractionStatus::random_failure;
        return r;
      }
      channel = electron_channel(coefficients, charge, q.energyMeV_,
                                 model.tables_.medium_.ae_, model.thresholds_, u);
    }
    return sample_at_vertex(model, channel, q, rng);
  }
} // namespace corsika::kokkos::egs4
