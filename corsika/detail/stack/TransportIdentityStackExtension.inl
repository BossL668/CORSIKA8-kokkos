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
    nextHistoryId_ = 1;
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

  inline void TransportIdentityData::incrementSize() { identities_.emplace_back(); }

  inline void TransportIdentityData::decrementSize() {
    if (!identities_.empty()) { identities_.pop_back(); }
  }

  inline HistoryId TransportIdentityData::allocateHistoryId() {
    return reserveHistoryIds(1);
  }

  inline HistoryId TransportIdentityData::reserveHistoryIds(std::uint64_t const count) {
    if (count == 0) {
      throw std::invalid_argument("Cannot reserve an empty transport history range");
    }
    if (nextHistoryId_ == NoParentHistoryId ||
        count - 1 > std::numeric_limits<HistoryId>::max() - nextHistoryId_) {
      throw std::overflow_error("Transport history ID space exhausted");
    }
    auto const first = nextHistoryId_;
    auto const last = first + count - 1;
    nextHistoryId_ =
        last == std::numeric_limits<HistoryId>::max() ? NoParentHistoryId : last + 1;
    return first;
  }

  inline void TransportIdentityData::assignPrimary(int const index) {
    identities_.at(index) =
        TransportIdentity{allocateHistoryId(), NoParentHistoryId, 0, 0};
  }

  inline void TransportIdentityData::assignSecondary(int const index,
                                                     TransportIdentity const& parent) {
    if (parent.historyId_ == NoParentHistoryId) {
      throw std::logic_error("Cannot create a secondary from an unassigned history");
    }
    if (parent.generation_ == std::numeric_limits<Generation>::max()) {
      throw std::overflow_error("Transport generation space exhausted");
    }
    identities_.at(index) =
        TransportIdentity{allocateHistoryId(), parent.historyId_,
                          static_cast<Generation>(parent.generation_ + 1), 0};
  }

  inline void TransportIdentityData::importIdentity(int const index,
                                                    TransportIdentity const& identity) {
    if (identity.historyId_ == NoParentHistoryId) {
      throw std::invalid_argument("Cannot import an unassigned transport history");
    }
    if ((identity.generation_ == 0) != (identity.parentHistoryId_ == NoParentHistoryId)) {
      throw std::invalid_argument(
          "Imported transport history has inconsistent parent and generation");
    }

    identities_.at(index) = identity;
    if (nextHistoryId_ != NoParentHistoryId && identity.historyId_ >= nextHistoryId_) {
      nextHistoryId_ = identity.historyId_ == std::numeric_limits<HistoryId>::max()
                           ? NoParentHistoryId
                           : identity.historyId_ + 1;
    }
  }

  inline TransportIdentity const& TransportIdentityData::getTransportIdentity(
      int const index) const {
    return identities_.at(index);
  }

  inline void TransportIdentityData::setStepId(int const index, StepId const stepId) {
    identities_.at(index).stepId_ = stepId;
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
    return getTransportIdentity().historyId_;
  }

  template <typename TParentStack>
  inline HistoryId TransportIdentityDataInterface<TParentStack>::getParentHistoryId()
      const {
    return getTransportIdentity().parentHistoryId_;
  }

  template <typename TParentStack>
  inline Generation TransportIdentityDataInterface<TParentStack>::getGeneration() const {
    return getTransportIdentity().generation_;
  }

  template <typename TParentStack>
  inline StepId TransportIdentityDataInterface<TParentStack>::getStepId() const {
    return getTransportIdentity().stepId_;
  }

  template <typename TParentStack>
  inline StepId TransportIdentityDataInterface<TParentStack>::beginTransportStep() {
    StepId const currentStep = getStepId();
    if (currentStep == std::numeric_limits<StepId>::max()) {
      throw std::overflow_error("Transport step ID space exhausted");
    }
    getStackData().setStepId(getIndex(), currentStep + 1);
    return currentStep;
  }

  template <typename TParentStack>
  inline std::string TransportIdentityDataInterface<TParentStack>::asString() const {
    auto const identity = getTransportIdentity();
    return fmt::format("history={}, parent={}, generation={}, step={}",
                       identity.historyId_, identity.parentHistoryId_,
                       identity.generation_, identity.stepId_);
  }

} // namespace corsika::transport
