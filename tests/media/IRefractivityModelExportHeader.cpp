/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <corsika/media/IRefractivityModelExport.hpp>

#include <type_traits>

static_assert(std::is_abstract_v<corsika::IRefractivityModelExport>);
static_assert(std::has_virtual_destructor_v<corsika::IRefractivityModelExport>);
