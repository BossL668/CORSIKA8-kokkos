/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace corsika::transport {

  inline void TransportIdentityData::clear() {
    identities_.clear();
    next_history_id_ = 1;
  }

  inline unsigned int TransportIdentityData::getSize() const {
    return identities_.size();
  }

  inline unsigned int TransportIdentityData::getCapacity() const {
    return identities_.capacity();
  }

  inline void TransportIdentityData::copy(int const i1, int const i2) {
    identities_.at(i2) = identities_.at(i1);
  }

  inline void TransportIdentityData::swap(int const i1, int const i2) {
    std::swap(identities_.at(i1), identities_.at(i2));
  }

  inline void TransportIdentityData::incrementSize() {
    identities_.emplace_back();
  }

  inline void TransportIdentityData::decrementSize() {
    if (!identities_.empty()) { identities_.pop_back(); }
  }

  inline HistoryId TransportIdentityData::allocateHistoryId() {
    return reserveHistoryIds(1);
  }

  inline HistoryId TransportIdentityData::reserveHistoryIds(
      std::uint64_t const count) {
    if (count == 0) {
      throw std::invalid_argument(
          "Cannot reserve an empty transport history range");
    }
    if (next_history_id_ == NoParentHistoryId ||
        count - 1 >
            std::numeric_limits<HistoryId>::max() -
                next_history_id_) {
      throw std::overflow_error(
          "Transport history ID space exhausted");
    }
    auto const first = next_history_id_;
    auto const last = first + count - 1;
    next_history_id_ =
        last == std::numeric_limits<HistoryId>::max()
            ? NoParentHistoryId
            : last + 1;
    return first;
  }

  inline void TransportIdentityData::assignPrimary(int const index) {
    identities_.at(index) =
        TransportIdentity{allocateHistoryId(), NoParentHistoryId, 0, 0};
  }

  inline void TransportIdentityData::assignSecondary(
      int const index, TransportIdentity const& parent) {
    if (parent.history_id == NoParentHistoryId) {
      throw std::logic_error("Cannot create a secondary from an unassigned history");
    }
    if (parent.generation == std::numeric_limits<Generation>::max()) {
      throw std::overflow_error("Transport generation space exhausted");
    }
    identities_.at(index) = TransportIdentity{
        allocateHistoryId(), parent.history_id,
        static_cast<Generation>(parent.generation + 1), 0};
  }

  inline void TransportIdentityData::importIdentity(
      int const index, TransportIdentity const& identity) {
    if (identity.history_id == NoParentHistoryId) {
      throw std::invalid_argument("Cannot import an unassigned transport history");
    }
    if ((identity.generation == 0) !=
        (identity.parent_history_id == NoParentHistoryId)) {
      throw std::invalid_argument(
          "Imported transport history has inconsistent parent and generation");
    }

    identities_.at(index) = identity;
    if (next_history_id_ != NoParentHistoryId &&
        identity.history_id >= next_history_id_) {
      next_history_id_ =
          identity.history_id == std::numeric_limits<HistoryId>::max()
              ? NoParentHistoryId
              : identity.history_id + 1;
    }
  }

  inline TransportIdentity const& TransportIdentityData::getTransportIdentity(
      int const index) const {
    return identities_.at(index);
  }

  inline void TransportIdentityData::setStepId(int const index, StepId const step_id) {
    identities_.at(index).step_id = step_id;
  }

  template <typename TParentStack>
  inline void TransportIdentityDataInterface<TParentStack>::setParticleData() {
    getStackData().assignPrimary(getIndex());
  }

  template <typename TParentStack>
  inline void TransportIdentityDataInterface<TParentStack>::setParticleData(
      TransportIdentityDataInterface const& parent) {
    getStackData().assignSecondary(getIndex(), parent.getTransportIdentity());
  }

  template <typename TParentStack>
  inline TransportIdentity
  TransportIdentityDataInterface<TParentStack>::getTransportIdentity() const {
    return getStackData().getTransportIdentity(getIndex());
  }

  template <typename TParentStack>
  inline void TransportIdentityDataInterface<TParentStack>::setTransportIdentity(
      TransportIdentity const& identity) {
    getStackData().importIdentity(getIndex(), identity);
  }

  template <typename TParentStack>
  inline HistoryId TransportIdentityDataInterface<TParentStack>::getHistoryId() const {
    return getTransportIdentity().history_id;
  }

  template <typename TParentStack>
  inline HistoryId
  TransportIdentityDataInterface<TParentStack>::getParentHistoryId() const {
    return getTransportIdentity().parent_history_id;
  }

  template <typename TParentStack>
  inline Generation TransportIdentityDataInterface<TParentStack>::getGeneration() const {
    return getTransportIdentity().generation;
  }

  template <typename TParentStack>
  inline StepId TransportIdentityDataInterface<TParentStack>::getStepId() const {
    return getTransportIdentity().step_id;
  }

  template <typename TParentStack>
  inline StepId TransportIdentityDataInterface<TParentStack>::beginTransportStep() {
    StepId const current_step = getStepId();
    if (current_step == std::numeric_limits<StepId>::max()) {
      throw std::overflow_error("Transport step ID space exhausted");
    }
    getStackData().setStepId(getIndex(), current_step + 1);
    return current_step;
  }

  template <typename TParentStack>
  inline std::string TransportIdentityDataInterface<TParentStack>::asString() const {
    auto const identity = getTransportIdentity();
    return fmt::format("history={}, parent={}, generation={}, step={}",
                       identity.history_id, identity.parent_history_id,
                       identity.generation, identity.step_id);
  }

} // namespace corsika::transport
