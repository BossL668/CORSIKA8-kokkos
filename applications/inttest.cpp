#include <qgsjet-II-public.hpp>

#include <fmt/core.h>
#include <functional>
#include <iostream>
#include <random>

void fill_random(double* x, size_t N) {
  static std::mt19937 rng;
  static std::uniform_real_distribution<double> dist;

  for (size_t i = 0; i < N; ++i) x[i] = dist(rng);
}

int main() {
  std::cout << fmt::format("{}", (void*)QGSJetII04::set_rng_function) << std::endl;

  QGSJetII04::set_rng_function(fill_random);
}
