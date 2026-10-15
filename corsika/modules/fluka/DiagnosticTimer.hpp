/* Distributed under the 3-clause BSD license; see LICENSE. */
#pragma once
#include <csignal>
#include <unistd.h>

namespace corsika::fluka {
  // Run immediately after FLUKA initialization even when the loaded historical
  // libfluka predates the linker-level guard. Keep the original physics library,
  // its initializer, RNG bridge, FP environment and other signal handlers intact.
  inline void disableDiagnosticTimerAfterInitialization() {
    struct sigaction action {};
    action.sa_handler=SIG_IGN;
    ::sigemptyset(&action.sa_mask);
    if(::sigaction(SIGALRM,&action,nullptr)!=0) {
      constexpr char message[]="C8 FLUKA: cannot disable unsafe diagnostic alarm\n";
      (void)::write(STDERR_FILENO,message,sizeof(message)-1);
      ::_exit(125);
    }
    ::alarm(0);
    constexpr char message[]="C8_FLUKA_POST_INIT_TIMER_GUARD v1: SIGALRM disabled\n";
    (void)::write(STDERR_FILENO,message,sizeof(message)-1);
  }
}
