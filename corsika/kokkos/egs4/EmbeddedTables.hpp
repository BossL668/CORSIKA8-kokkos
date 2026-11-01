/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <memory>
namespace corsika::kokkos::egs4::application {
  class Session;
  std::unique_ptr<Session> make_embedded_session();
  char const* embedded_table_sha256();
} // namespace corsika::kokkos::egs4::application
