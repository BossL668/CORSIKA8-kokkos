// Isolated YAML ownership diagnostic, not a shower or a leak sanitizer.
// Reproduce GpuEmRunOutput's per-shower report retention, then destroy it.
#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

double rssMiB() {
  std::ifstream statm("/proc/self/statm");
  std::size_t total{}, resident{};
  statm >> total >> resident;
  return resident * static_cast<double>(sysconf(_SC_PAGESIZE)) / 1048576.;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: probe SUMMARY.yaml COUNT OUTPUT.csv\n";
    return 2;
  }
  int count = std::stoi(argv[2]);
  if (count < 2 || count > 512) throw std::runtime_error("count must be 2..512");
  YAML::Node prototype;
  {
    auto library = YAML::LoadFile(argv[1]);
    prototype = YAML::Clone(library["shower_0"]["statistics"]);
  }
  malloc_trim(0);
  std::ofstream output(argv[3]);
  if (!output) throw std::runtime_error("cannot open output");
  output << "phase,shower,rss_mib\n";
  auto checkpoint = [&](char const* phase, int n) {
    output << phase << ',' << n << ',' << rssMiB() << '\n';
    output.flush();
  };
  checkpoint("prototype_only", 0);
  {
    YAML::Node reports;
    for (int i = 0; i < count; ++i) {
      auto shower = reports["shower_" + std::to_string(i)];
      shower["complete"] = true;
      shower["status"] = "complete";
      shower["statistics"] = YAML::Clone(prototype);
      if ((i + 1) % 8 == 0 || i + 1 == count) checkpoint("retained", i + 1);
    }
  }
  checkpoint("reports_destroyed", count);
  // Diagnostic only: distinguish reachable data from glibc free arenas.
  // Do not introduce malloc_trim into the simulation's hot path.
  malloc_trim(0);
  checkpoint("destroyed_and_trimmed", count);
  return 0;
}
