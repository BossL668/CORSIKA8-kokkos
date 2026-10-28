/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/IMagneticFieldModel.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/media/IUniformMagneticFieldExport.hpp>

namespace corsika {

  /**
   * A uniform (constant) magnetic field.
   *
   * This class returns the same magnetic field vector
   * for all evaluated locations.
   */
  template <typename T>
  class UniformMagneticField : public T, public IUniformMagneticFieldExport {

  public:
    /**
     * Construct a UniformMagneticField.
     *
     * This is initialized with a fixed magnetic field
     * and returns this magnetic field at all locations.
     *
     * @param field   The fixed magnetic field to return.
     */
    template <typename... Args>
    UniformMagneticField(MagneticFieldVector const& field, Args&&... args)
        : T(std::forward<Args>(args)...)
        , B_(field) {}

    /**
     * Evaluate the magnetic field at a given location.
     *
     * @param  point    The location to evaluate the field at (not used internally).
     * @returns    The magnetic field vector.
     */
    MagneticFieldVector getMagneticField([
        [maybe_unused]] Point const& point) const final override {
      return B_;
    }

    /**
     * Set the magnetic field returned by this instance.
     *
     * @param  Bfield    The new vaue of the global magnetic field.
     */
    void setMagneticField(MagneticFieldVector const& Bfield) { B_ = Bfield; }

    /// Copy the current field with its original units and coordinate system.
    [[nodiscard]] MagneticFieldVector exportUniformField() const override;

  private:
    MagneticFieldVector B_; ///< The constant magnetic field we use.

  }; // END: class MagneticField

} // namespace corsika

#include <corsika/detail/media/UniformMagneticField.inl>
