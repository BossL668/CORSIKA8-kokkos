#pragma once
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <fstream>
#include <stdexcept>
#include <string>

namespace corsika::applications::terrain {
// Optional lossless compression of diagnostic text. The formatted CSV bytes,
// physics callbacks, particle states and random streams are unchanged.
class TerrainCsvStream : public boost::iostreams::filtering_ostream {
 public:
  void open(std::string const& name, bool compressed) {
    file_.open(name + (compressed ? ".gz" : ""), std::ios::binary);
    if (!file_) throw std::runtime_error("cannot create terrain CSV: " + name);
    if (compressed) push(boost::iostreams::gzip_compressor(boost::iostreams::gzip_params(1)));
    push(file_);
    exceptions(std::ios::badbit | std::ios::failbit);
  }
  void close() {
    if (empty()) return;
    flush();
    reset(); // Finish and verify the gzip footer before the file is closed.
    file_.close();
    if (file_.fail()) throw std::runtime_error("terrain CSV close failed");
  }
  ~TerrainCsvStream() noexcept {
    try { close(); } catch (...) {} // Explicit endOfLibrary reports write errors.
  }
 private:
  std::ofstream file_;
};
}
