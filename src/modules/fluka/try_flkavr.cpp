/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

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
