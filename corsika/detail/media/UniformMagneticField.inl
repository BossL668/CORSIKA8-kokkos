/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika {

  template <typename T>
  inline MagneticFieldVector UniformMagneticField<T>::exportUniformField() const {
    return B_;
  }

} // namespace corsika
