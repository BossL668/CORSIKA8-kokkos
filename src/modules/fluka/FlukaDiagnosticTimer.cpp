/*
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <atomic>
#include <csignal>
#include <unistd.h>

// Linker wrapping keeps FLUKA's original initialization (including its floating
// point and SIGTERM setup). Only the standalone diagnostic alarm is disabled.
// Its timer -> quachk -> fwrite path may re-enter malloc from a signal handler
// while the interrupted thread owns the allocator lock. This affects scalar,
// OpenMP and CUDA applications, not just process-isolated hadronic workers.
extern "C" void __real_fpenab_();

namespace {
  std::atomic<unsigned> guardCalls{0};
}

extern "C" unsigned c8_fluka_diagnostic_timer_guard_calls() {
  return guardCalls.load(std::memory_order_relaxed);
}

extern "C" void __wrap_fpenab_() {
  __real_fpenab_();
  struct sigaction action {};
  action.sa_handler = SIG_IGN;
  ::sigemptyset(&action.sa_mask);
  // Install SIG_IGN before cancelling the timer, also discarding any pending
  // alarm. Do not disable floating-point exceptions or alter other signals.
  if (::sigaction(SIGALRM, &action, nullptr) != 0) {
    constexpr char message[] = "C8 FLUKA: cannot disable unsafe diagnostic SIGALRM\n";
    auto const written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)written;
    ::_exit(125);
  }
  ::alarm(0);
  guardCalls.fetch_add(1, std::memory_order_relaxed);
  constexpr char message[] = "C8_FLUKA_DIAGNOSTIC_TIMER_GUARD v1: SIGALRM disabled\n";
  auto const written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
  (void)written;
}
