#include <stdio.h>
#include <ctype.h>

extern "C" {
extern struct {
  int mjflvr, mnflvr, mrflvr;
} flkavr_;

extern struct {
  char chflvr;
} flkavc_;
}

int main() {
  printf("FLUKA version: %d.%d.%d", flkavr_.mjflvr, flkavr_.mnflvr, flkavr_.mrflvr);

  if (isalnum(flkavc_.chflvr)) { putchar(flkavc_.chflvr); }

  putchar('\n');

  return 0;
}
