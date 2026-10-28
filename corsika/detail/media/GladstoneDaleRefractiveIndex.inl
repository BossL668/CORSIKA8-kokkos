/*
 * (c) Copyright 2023 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/IRefractiveIndexModel.hpp>

namespace corsika {

  template <typename T>
  template <typename... Args>
  inline GladstoneDaleRefractiveIndex<T>::GladstoneDaleRefractiveIndex(
      double const referenceRefractiveIndex, Point const point, Args&&... args)
      : T(std::forward<Args>(args)...)
      , referenceRefractivity_(referenceRefractiveIndex - 1)
      , referenceInvDensity_(1 / this->getMassDensity(point)) {}

  template <typename T>
  inline double GladstoneDaleRefractiveIndex<T>::getRefractiveIndex(
      Point const& point) const {
    return referenceRefractivity_ * (this->getMassDensity(point) * referenceInvDensity_) +
           1.;
  }

  template <typename T>
  inline RefractivityModelParameters GladstoneDaleRefractiveIndex<T>::exportRefractivity()
      const {
    return {RefractivityModelParameters::Kind::GladstoneDale, referenceRefractivity_,
            referenceInvDensity_};
  }

} // namespace corsika
