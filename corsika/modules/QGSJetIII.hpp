/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/qgsjetIII/InteractionModel.hpp>

#include <corsika/framework/process/InteractionProcess.hpp>

/**
 * @file QGSJetIII.hpp
 *
 * Includes all the parts of the QGSJetIII model. Defines the InteractionProcess<TModel>
 * classes needed for the ProcessSequence.
 */

namespace corsika::qgsjetIII {
  /**
   * @brief qgsjetIII::Interaction is the process for ProcessSequence.
   *
   * The qgsjetIII::InteractionModel is wrapped as an InteractionProcess here in order
   * to provide all the functions for ProcessSequence.
   */
  class Interaction : public InteractionModel, public InteractionProcess<Interaction> {};
} // namespace corsika::qgsjetIII
