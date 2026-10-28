/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

namespace corsika::kokkos::radio {
  inline ::corsika::kokkos::LeptonTransportRecord make_host_radio_record(
      corsika::radio::RadioTrackSegment const& segment) {
    if (!::corsika::kokkos::is_electron_or_positron_pid(segment.pdg_)) {
      throw std::invalid_argument("only electron/positron tracks enter radio projection");
    }
    ::corsika::kokkos::LeptonTransportRecord record{};
    record.start_.pid_ = record.end_.pid_ = segment.pdg_;
    record.start_.weight_ = record.end_.weight_ = segment.weight_;
    record.start_.timeS_ = segment.startTimeS_;
    record.end_.timeS_ = segment.endTimeS_;
    auto const mass = ::corsika::kokkos::transport_mass_gev(segment.pdg_);
    record.start_.energyGeV_ = segment.startKineticEnergyGeV_ + mass;
    record.end_.energyGeV_ = segment.endKineticEnergyGeV_ + mass;
    for (int axis = 0; axis < 3; ++axis) {
      record.start_.positionM_[axis] = segment.startPositionM_[axis];
      record.end_.positionM_[axis] = segment.endPositionM_[axis];
      record.start_.direction_[axis] = segment.startDirection_[axis];
      record.end_.direction_[axis] = segment.endDirection_[axis];
    }
    return record;
  }
} // namespace corsika::kokkos::radio
