/*
 * (c) Copyright 2022 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#pragma once

#include <corsika/modules/radio/detectors/ObserverCollection.hpp>

namespace corsika {

  template <typename TObserverImpl>
  inline void ObserverCollection<TObserverImpl>::addObserver(
      TObserverImpl const& observer) {
    observers_.push_back(observer);
  }

  template <typename TObserverImpl>
  inline TObserverImpl& ObserverCollection<TObserverImpl>::at(std::size_t const i) {
    return observers_.at(i);
  }

  template <typename TObserverImpl>
  inline TObserverImpl const& ObserverCollection<TObserverImpl>::at(
      std::size_t const i) const {
    return observers_.at(i);
  }

  template <typename TObserverImpl>
  inline int ObserverCollection<TObserverImpl>::size() const {
    return observers_.size();
  }

  template <typename TObserverImpl>
  inline std::vector<TObserverImpl>& ObserverCollection<TObserverImpl>::getObservers() {
    return observers_;
  }

  template <typename TObserverImpl>
  inline std::vector<TObserverImpl> const&
  ObserverCollection<TObserverImpl>::getObservers() const {
    return observers_;
  }

  template <typename TObserverImpl>
  inline void ObserverCollection<TObserverImpl>::reset() {
    std::for_each(observers_.begin(), observers_.end(),
                  std::mem_fn(&TObserverImpl::reset));
  }

} // namespace corsika
