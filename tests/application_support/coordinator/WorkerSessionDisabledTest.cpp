/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// Compile without Kokkos, JSON or model libraries: native applications share this API.
#include <corsika/kokkos/application/cascade/WorkerSession.hpp>

#ifdef CORSIKA_KOKKOS_COORDINATOR
#error "The disabled worker contract must not enable the coordinator"
#endif

namespace {
  struct Output {
    unsigned closed_ = 0;
    void endOfLibrary() { ++closed_; }
  };
} // namespace

int main() {
  corsika::kokkos::application::WorkerSession session;
  Output output;
  int backend = 0;
  int application = 0;
  long seed = 17;
  session.ready();
  session.ready(backend);
  session.ready(backend, backend, backend);
  if (session.isActive() || !session.next(backend, application, output, seed)) return 1;
  session.close(output);
  return output.closed_ == 1 && seed == 17 ? 0 : 1;
}
