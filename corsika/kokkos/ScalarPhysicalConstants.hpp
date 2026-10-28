/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <cstdint>
#include <limits>

namespace corsika::kokkos::scalar_constants {

  // Device-safe mirrors of the existing scalar CORSIKA unit system, not a new
  // CODATA set. Host checks below prevent silent drift. PROPOSAL masses,
  // alpha, RE and LPM parameters retain their own calculator values.
  inline constexpr std::uint32_t Version = 1;
  inline constexpr double SpeedOfLightMPerS = 299792458.;
  inline constexpr double ElementaryChargeC = 1.6021766208e-19;
  inline constexpr double VacuumPermittivityFPerM = 8.8541878128e-12;
  inline constexpr double MagneticRigidityGeVPerTeslaMeter = 0.2997925146834389;

} // namespace corsika::kokkos::scalar_constants

// Keep the host-only units library out of CUDA, HIP and SYCL device compilation.
#if !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(__SYCL_DEVICE_ONLY__)
// PhysicalUnits includes PhysicalConstants before defining HEP conversions;
// entering through PhysicalConstants first is not supported by that unit pair.
#include <corsika/framework/core/PhysicalUnits.hpp>

namespace corsika::kokkos::scalar_constants::checks {

  using namespace corsika::units::si;
  inline constexpr auto MomentumOneGeVSI =
      constants::c * convert_HEP_to_SI<MassType::dimension_type>(1_GeV);
  inline constexpr double ScalarMagneticRigidity =
      constants::e * tesla / MomentumOneGeVSI * meter;
  static_assert(SpeedOfLightMPerS == constants::c / (meter / second));
  static_assert(ElementaryChargeC == constants::e / coulomb);
  static_assert(VacuumPermittivityFPerM == constants::epsilonZero / (farad / meter));
  static_assert(MagneticRigidityGeVPerTeslaMeter / ScalarMagneticRigidity >
                        1. - 8. * std::numeric_limits<double>::epsilon() &&
                    MagneticRigidityGeVPerTeslaMeter / ScalarMagneticRigidity <
                        1. + 8. * std::numeric_limits<double>::epsilon(),
                "update the Kokkos magnetic factor to the scalar unit system");

} // namespace corsika::kokkos::scalar_constants::checks
#endif
