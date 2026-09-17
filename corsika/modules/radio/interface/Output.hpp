#pragma once
#include <corsika/modules/radio/interface/Types.hpp>
#include <complex>
namespace corsika::radio::interface {
struct Rendered {
  // (observer * 6 + region * 3 + component, frequency/time).
  std::vector<std::complex<double>> field_spectrum;
  std::vector<double> field;
};
Rendered render(Result const&);
// Writes raw moments, optical configuration, E(t), and complex E(f). No fit,
// amplitude normalization, bandpass, antenna/electronics response or time shift.
void writeResult(Result const&,std::string const& directory);
} // namespace corsika::radio::interface
