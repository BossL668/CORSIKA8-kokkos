#pragma once

#include <array>
#include <iostream>
#include <string_view>

namespace corsika {
/**
 * Small helper class to provide a data-directory name in the format qgsjetII expects.
 */
template <size_t N>
class datadir {
public:
  datadir(std::string_view dir) :
    length_{std::min(N, dir.length())}
   {
    if (dir.length() > N) { // we don't test this limitation: LCOV_EXCL_START
      std::cerr << "will cut datadir \"" << dir
                << "\" to " << N << " characters: " << std::endl;
    } // LCOV_EXCL_STOP
    size_t i = 0;
    for (i = 0; i < length_; ++i) data_[i] = dir[i];
    data_[i + 0] = ' ';
    data_[i + 1] = '\0';
  }
  
  char const* data() const {
      return data_.data();
  }
  
  size_t length() const {return length_;}
  
  private:
  datadir operator=(std::string_view dir);
  datadir operator=(const datadir&);
std::array<char, N+2> data_{};
size_t const length_{};
};
}
