/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once
#include "ContinuousAdapters.hpp"
#include "ContinuousDataset.hpp"
#include "../tables/ProposalNativeQueryTestData.hpp"
#include "../../environment/ModelGrammageReference.hpp"
#include "../../environment/UniformMagneticFieldReference.hpp"
#include <corsika/kokkos/environment/ModelGrammage.hpp>
#include <corsika/kokkos/environment/UniformMagneticField.hpp>
#include <array>
#include <cmath>
#include <limits>

namespace continuous_test {
  CORSIKA_HOST_DEVICE inline reference::tables::NativePhysicsView old_view(
      native_table_reference::ProposalNativeDeviceView const& native,
      Query const& query) {
    reference::tables::NativePhysicsView result;
    result.proposal_native = native;
    result.em_transport_cut_MeV = query.emCutMeV_;
    result.muon_transport_cut_MeV = query.muonCutMeV_;
    return result;
  }
  inline current::InteractionRecord interaction(std::int32_t pid, double energyMeV) {
    current::InteractionRecord result;
    auto& particle = result.particle_;
    particle.pid_ = pid;
    particle.mediumId_ = 19;
    particle.environmentNode_ = 7;
    particle.energyGeV_ = energyMeV / 1000.;
    particle.direction_[0] = .6;
    particle.direction_[2] = .8;
    particle.positionM_[0] = 120.;
    particle.positionM_[1] = -30.;
    particle.positionM_[2] = 1500.;
    particle.timeS_ = 3.e-6;
    particle.weight_ = 2.5;
    particle.historyId_ = 0xfedcba9876543210ULL;
    particle.parentHistoryId_ = 0x123456789abcdef0ULL;
    particle.stepId_ = 0x8000000000000021ULL;
    particle.generation_ = 13;
    particle.reserved_ = 17;
    result.inputIndex_ = 0xf123456789abcdefULL;
    result.processId_ = 1000000003;
    result.componentHash_ = 0xba9876543210fedcULL;
    result.interactionVertexReached_ = 1;
    result.status_ = current::InteractionStatus::RequiresReselection;
    result.energyFraction_ = .234;
    result.processUniform_ = .321;
    result.lossQuantile_ = .456;
    result.distanceUniform_ = .654;
    result.proposalSelectionUniform_ = .678;
    result.processRandomProcessId_ = 333;
    result.proposalSelectionRandomProcessId_ = 444;
    result.processDrawId_ = 0x9000000000000321ULL;
    result.lossDrawId_ = 0x9000000000000456ULL;
    result.proposalSelectionDrawId_ = 0x9000000000000678ULL;
    return result;
  }

  CORSIKA_HOST_DEVICE inline current::EnvironmentModel chain_model() {
    current::EnvironmentModel model;
    model.densityKind_ = current::EnvironmentModel::Density::FlatExponential;
    model.referenceDensityGPerCm3_ = .001225;
    model.referenceHeightM_ = 1500.;
    model.scaleLengthM_ = -8000.;
    model.inverseScaleLengthPerM_ = -1. / 8000.;
    model.axis_[2] = 1.;
    return model;
  }

  CORSIKA_HOST_DEVICE inline Outcome evaluate(
      current::ProposalNativeDeviceView const& native, Query const& query) {
    current::ContinuousPhysicsView const view(native, query.emCutMeV_, query.muonCutMeV_);
    auto const pid = query.interaction_.particle_.pid_;
    auto const energy = query.interaction_.particle_.energyGeV_ * 1000.;
    Outcome result;
    current::TableQueryResult value;
    switch (query.operation_) {
      case Operation::Dedx:
        value = current::query_continuous_dedx(view, pid, energy);
        break;
      case Operation::Mass:
        value = current::query_continuous_mass(view, pid);
        break;
      case Operation::TransportMass:
        value = current::query_continuous_transport_mass(view, pid);
        break;
      case Operation::Cut:
        value = current::query_continuous_transport_cut(view, pid);
        break;
      case Operation::Minimum:
        value = current::query_continuous_minimum_energy(view, pid);
        break;
      case Operation::Range:
        value = current::query_continuous_range(view, pid, energy);
        break;
      case Operation::Energy:
        value = current::query_continuous_energy(view, pid, query.argument_);
        break;
      case Operation::AfterLoss:
        value = current::query_energy_after_continuous_loss(view, pid, energy,
                                                            query.argument_);
        break;
      default: {
        auto const preparation =
            current::prepare_lepton_continuous_step(view, query.interaction_);
        append(result, preparation);
        if (query.operation_ == Operation::Prepare || preparation.fallbackFlag_ ||
            preparation.particleCut_)
          return result;
        double grammage = query.argument_;
        if (query.operation_ == Operation::Chain) {
          auto const advanced = current::advance_uniform_magnetic_field(
              query.interaction_.particle_, preparation.massMeV_ / 1000.,
              pid > 0 ? -1. : 1., query.fieldT_, query.argument_);
          append(result, advanced.status_);
          append(result, advanced.reserved_);
          append(result, advanced.particle_);
          append(result, advanced.bendParameter_);
          append(result, advanced.chordLengthM_);
          double direction[3];
          for (int axis = 0; axis < 3; ++axis)
            direction[axis] = (advanced.particle_.positionM_[axis] -
                               query.interaction_.particle_.positionM_[axis]) /
                              advanced.chordLengthM_;
          grammage =
              current::grammage(chain_model(), query.interaction_.particle_.positionM_,
                                direction, advanced.chordLengthM_);
          append(result, grammage);
        }
        append(result, current::evaluate_lepton_continuous_loss(
                           view, query.interaction_, preparation, grammage,
                           query.continuousLimitWon_ != 0));
        return result;
      }
    }
    append(result, value);
    return result;
  }

  CORSIKA_HOST_DEVICE inline Outcome expected(
      native_table_reference::ProposalNativeDeviceView const& native,
      Query const& query) {
    auto const view = old_view(native, query);
    auto const record = convert(query.interaction_);
    auto const pid = record.particle.pid;
    auto const energy = record.particle.energy_GeV * 1000.;
    Outcome result;
    reference::tables::TableQueryResult value;
    using namespace reference::tables;
    switch (query.operation_) {
      case Operation::Dedx:
        value = queryContinuousDedx(view, pid, energy);
        break;
      case Operation::Mass:
        value = queryContinuousMass(view, pid);
        break;
      case Operation::TransportMass:
        value = queryContinuousTransportMass(view, pid);
        break;
      case Operation::Cut:
        value = queryContinuousTransportCut(view, pid);
        break;
      case Operation::Minimum:
        value = queryContinuousMinimumEnergy(view, pid);
        break;
      case Operation::Range:
        value = queryContinuousRange(view, pid, energy);
        break;
      case Operation::Energy:
        value = queryContinuousEnergy(view, pid, query.argument_);
        break;
      case Operation::AfterLoss:
        value = queryEnergyAfterContinuousLoss(view, pid, energy, query.argument_);
        break;
      default: {
        auto const preparation =
            reference::step::prepareLeptonContinuousStep(view, record);
        append(result, preparation);
        if (query.operation_ == Operation::Prepare || preparation.fallback_flag ||
            preparation.particle_cut)
          return result;
        double grammage = query.argument_;
        if (query.operation_ == Operation::Chain) {
          auto const advanced = uniform_magnetic_reference::advanceUniformMagneticField(
              query.interaction_.particle_, preparation.mass_MeV / 1000.,
              pid > 0 ? -1. : 1., query.fieldT_, query.argument_);
          append(result, advanced.status);
          append(result, advanced.reserved);
          append(result, advanced.particle);
          append(result, advanced.bend_parameter);
          append(result, advanced.chord_length_m);
          double direction[3];
          for (int axis = 0; axis < 3; ++axis)
            direction[axis] = (advanced.particle.positionM_[axis] -
                               query.interaction_.particle_.positionM_[axis]) /
                              advanced.chord_length_m;
          grammage = model_grammage_reference::grammage(
              chain_model(), query.interaction_.particle_.positionM_, direction,
              advanced.chord_length_m);
          append(result, grammage);
        }
        append(result,
               reference::step::evaluateLeptonContinuousLoss(
                   view, record, preparation, grammage, query.continuousLimitWon_ != 0));
        return result;
      }
    }
    append(result, value);
    return result;
  }

  template <typename TFloat, typename TState>
  void compare(Outcome const& actual, Outcome const& reference, TFloat floating,
               TState state) {
    state(actual.valueCount_, reference.valueCount_);
    state(actual.stateCount_, reference.stateCount_);
    for (std::uint32_t index = 0; index < actual.valueCount_; ++index)
      floating(actual.values_[index], reference.values_[index]);
    for (std::uint32_t index = 0; index < actual.stateCount_; ++index)
      state(actual.states_[index], reference.states_[index]);
  }

  inline void add_edge_queries(Dataset& dataset) {
    for (auto const& column : dataset.table_.utilityColumns_) {
      auto const mass = 1000. * current::transport_mass_gev(column.pdgId_);
      for (double energy :
           {mass, mass + .4, std::nextafter(mass + .4, 0.),
            std::nextafter(mass + .4, INFINITY), column.lowerEnergyLimitMeV_,
            std::nextafter(column.lowerEnergyLimitMeV_, 0.), column.spline_.axis_.high_,
            std::nextafter(column.spline_.axis_.high_, INFINITY), -1.,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity()}) {
        for (Operation operation : {Operation::Dedx, Operation::Range, Operation::Prepare,
                                    Operation::AfterLoss}) {
          Query query;
          query.operation_ = operation;
          query.interaction_ = interaction(column.pdgId_, energy);
          query.argument_ = .01;
          dataset.queries_.push_back(query);
        }
      }
      for (double cut : {0., -1., std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN(), 1.e12}) {
        Query query;
        query.operation_ = Operation::Prepare;
        query.interaction_ = interaction(column.pdgId_, 1.e3);
        query.emCutMeV_ = cut;
        query.muonCutMeV_ = cut;
        dataset.queries_.push_back(query);
      }
      for (bool native : {false, true}) {
        Query query;
        query.operation_ = Operation::Loss;
        query.interaction_ = interaction(column.pdgId_, 1.e3);
        query.interaction_.proposalSelectionRandomProcessId_ = native ? 444 : 0;
        query.argument_ = -.1;
        dataset.queries_.push_back(query);
      }
    }
    for (auto operation : {Operation::Mass, Operation::TransportMass, Operation::Minimum,
                           Operation::Prepare}) {
      Query query;
      query.operation_ = operation;
      query.interaction_ = interaction(2212, 1.e3);
      dataset.queries_.push_back(query);
    }
  }
} // namespace continuous_test
