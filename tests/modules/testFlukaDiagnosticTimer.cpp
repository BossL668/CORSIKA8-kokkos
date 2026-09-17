// Standalone regression: no FLUKA data, GPU, or CORSIKA environment required.
#include <atomic>
#include <cfenv>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <pthread.h>
#include <thread>
#include <unistd.h>

extern "C" void __wrap_fpenab_();
extern "C" unsigned c8_fluka_diagnostic_timer_guard_calls();
namespace {
  unsigned realCalls = 0;
  volatile sig_atomic_t alarmCalls = 0;
  void alarmHandler(int) { ++alarmCalls; }
  void termHandler(int) {}
}
extern "C" void __real_fpenab_() {
  ++realCalls;
  std::signal(SIGALRM, alarmHandler);
  std::signal(SIGTERM, termHandler);
  ::alarm(60);
  std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_INEXACT);
}
int main() {
  auto check = [](bool ok, char const* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::_Exit(1); }
  };
  for (unsigned n = 1; n <= 3; ++n) {
    __wrap_fpenab_();
    struct sigaction action {};
    check(::sigaction(SIGALRM, nullptr, &action) == 0 && action.sa_handler == SIG_IGN,
          "SIGALRM not ignored");
    check(::alarm(0) == 0, "timer still armed");
    check(::sigaction(SIGTERM, nullptr, &action) == 0 && action.sa_handler == termHandler,
          "original SIGTERM behavior changed");
    check(std::fegetround() == FE_DOWNWARD && std::fetestexcept(FE_INEXACT),
          "original floating point state changed");
    check(realCalls == n && c8_fluka_diagnostic_timer_guard_calls() == n,
          "initializer not called exactly once");
  }
  auto const owner = ::pthread_self();
  std::atomic<bool> done{false};
  std::thread interrupts([&] {
    while (!done.load()) { ::pthread_kill(owner, SIGALRM); ::usleep(100); }
  });
  std::size_t sum = 0;
  for (unsigned n = 0; n < 200000; ++n) {
    auto p = std::make_unique<unsigned[]>(64 + n % 64);
    p[n % 64] = n;
    sum += p[n % 64];
  }
  done.store(true);
  interrupts.join();
  check(alarmCalls == 0 && sum != 0, "alarm handler ran during allocation stress");
  std::puts("PASS: init forwarding, alarm cancellation, FP/SIGTERM preservation, allocation stress");
}
