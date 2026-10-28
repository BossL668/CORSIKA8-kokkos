/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/IRefractiveIndexModel.hpp>

namespace corsika {

  template <typename T>
  template <typename... Args>
  inline UniformRefractiveIndex<T>::UniformRefractiveIndex(double const n, Args&&... args)
      : T(std::forward<Args>(args)...)
      , n_(n) {}

  template <typename T>
  inline double UniformRefractiveIndex<T>::getRefractiveIndex(Point const&) const {
    return n_;
  }

  template <typename T>
  inline void UniformRefractiveIndex<T>::setRefractiveIndex(double const n) {
    n_ = n;
  }

  template <typename T>
  inline RefractivityModelParameters UniformRefractiveIndex<T>::exportRefractivity()
      const {
    return {RefractivityModelParameters::Kind::Uniform, n_,
            InverseMassDensityType::zero()};
  }

} // namespace corsika
