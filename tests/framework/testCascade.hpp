/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/media/Environment.hpp>
#include <corsika/media/IMediumModel.hpp>
#include <corsika/media/HomogeneousMedium.hpp>

#include <corsika/framework/stack/CombinedStack.hpp>
#include <corsika/framework/stack/SecondaryView.hpp>
#include <corsika/stack/GeometryNodeStackExtension.hpp>
#include <corsika/stack/VectorStack.hpp>

using TestEnvironmentInterface = corsika::HomogeneousMedium<corsika::IMediumModel>;
using TestEnvironmentType = corsika::Environment<TestEnvironmentInterface>;

template <typename T>
using SetupGeometryDataInterface =
    corsika::node::GeometryDataInterface<T, TestEnvironmentType>;

// combine particle data stack with geometry information for tracking
template <typename StackIter>
using StackWithGeometryInterface =
    corsika::CombinedParticleInterface<corsika::VectorStack::pi_type,
                                       SetupGeometryDataInterface, StackIter>;

using TestCascadeStack =
    corsika::CombinedStack<typename corsika::VectorStack::stack_data_type,
                           corsika::node::GeometryData<TestEnvironmentType>,
                           StackWithGeometryInterface>;

/*
  See also Issue 161
*/
#if defined(__clang__)
using TestCascadeStackView =
    corsika::SecondaryView<typename TestCascadeStack::stack_data_type,
                           StackWithGeometryInterface>;
#elif defined(__GNUC__) || defined(__GNUG__)
using TestCascadeStackView = corsika::MakeView<TestCascadeStack>::type;
#endif
