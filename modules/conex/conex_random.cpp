#include <rng_impl.hpp>

IMPLEMENT_RNG(conex)

extern "C" __attribute__((visibility("hidden"))) void rmmard_(double field[],
                                                              int const* N, int*) {
  for (int i = 0; i < *N; ++i) { field[i] = draw_std_rnd(); }
}

extern "C" __attribute__((visibility("hidden"))) double drangen_() { return draw_std_rnd(); }

// this is needed as linker object, but it is not needed to do anything
extern "C" __attribute__((visibility("hidden"))) void rmmaqd_(int[3], int&, char*, int) {}

// this is needed as linker object, but it is not needed to do anything
extern "C" __attribute__((visibility("hidden"))) void ranfgt_(int&) {}
