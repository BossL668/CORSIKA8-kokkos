/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */
#include <corsika/modules/fluka/InteractionModel.hpp>

#include <type_traits>

static_assert(std::is_same_v<corsika::fluka::InteractionModel::FinalStateParticle,
                             std::tuple<corsika::Code, corsika::HEPEnergyType,
                                        corsika::DirectionVector>>);
