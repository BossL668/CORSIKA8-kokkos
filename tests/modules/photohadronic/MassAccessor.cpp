/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <sophia.hpp>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    for (int code = 1; code <= 49; ++code) {
      int negative = -code;
      if (get_sophia_mass(code) != so_mass1_.am[code - 1] ||
          get_sophia_mass(negative) != so_mass1_.am[code - 1] ||
          get_sophia_mass2(code) != so_mass1_.am2[code - 1])
        throw std::runtime_error("Rebuilt SOPHIA AM/AM2 accessor differs from COMMON");
    }
    std::cout << "passed real SOPHIA mass accessor checks=147\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
