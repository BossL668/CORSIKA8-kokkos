#pragma once

// Host-only command-line selection, before Kokkos/model initialization.
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

namespace corsika::applications::gpu_cli {
struct DeviceSelection {
  std::vector<std::string> ids;
  bool multiple() const { return ids.size() > 1; }
};

inline std::string trimDevice(std::string const& s) {
  auto first = s.find_first_not_of(" \t\r\n");
  return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

inline DeviceSelection parseDevices(int argc, char const* const* argv) {
  DeviceSelection result;
  bool seen = false, legacy = false;
  // Help must not query hardware or start a coordinator.
  for (int i = 1; i < argc && std::string(argv[i]) != "--"; ++i)
    if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") return result;
  auto append = [&](std::string const& value) {
    std::size_t begin = 0;
    do {
      auto end = value.find(',', begin);
      auto id = trimDevice(value.substr(begin, end == std::string::npos ? end : end - begin));
      if (id.empty()) throw std::invalid_argument("--device contains an empty GPU ID");
      bool numeric = id.find_first_not_of("0123456789") == std::string::npos;
      if (numeric) {
        auto n = std::stoull(id);
        if (n > unsigned(std::numeric_limits<int>::max())) throw std::invalid_argument("GPU index is too large");
        id = std::to_string(n);
      } else if (id.rfind("GPU-", 0) != 0 || id.size() <= 4 ||
                 id.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-") != std::string::npos) {
        throw std::invalid_argument("--device expects nonnegative NVIDIA GPU indices or full GPU UUIDs");
      }
      if (std::find(result.ids.begin(), result.ids.end(), id) != result.ids.end())
        throw std::invalid_argument("--device contains a duplicate GPU ID: " + id);
      result.ids.push_back(id);
      if (result.ids.size() > 255) throw std::invalid_argument("--device accepts at most 255 GPUs");
      if (end == std::string::npos) break;
      begin = end + 1;
    } while (true);
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--") break;
    auto eq = arg.find('=');
    auto key = arg.substr(0, eq);
    if (key == "--kokkos-device") legacy = true;
    if (key != "--device" && key != "--devices") continue;
    if (seen) throw std::invalid_argument("Specify --device once; do not combine it with --devices");
    seen = true;
    if (eq != std::string::npos) append(arg.substr(eq + 1));
    while (i + 1 < argc && argv[i + 1][0] != '-') append(argv[++i]);
    if (result.ids.empty()) throw std::invalid_argument("--device requires at least one GPU ID");
  }
  if (seen && legacy) throw std::invalid_argument("Do not combine --device/--devices with --kokkos-device");
  return result;
}

inline std::vector<std::string> resolveGpuIds(std::vector<std::string> const& requested,
                                             std::string const& listing) {
  std::map<std::string, std::string> aliases;
  std::size_t begin = 0;
  while (begin < listing.size()) {
    auto end = listing.find('\n', begin);
    auto line = listing.substr(begin, end == std::string::npos ? end : end - begin);
    auto comma = line.find(',');
    if (comma != std::string::npos) {
      auto index = trimDevice(line.substr(0, comma)), uuid = trimDevice(line.substr(comma + 1));
      if (!index.empty() && index.find_first_not_of("0123456789") == std::string::npos && uuid.rfind("GPU-", 0) == 0)
        aliases[index] = aliases[uuid] = uuid;
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (auto const& id : requested) {
    auto found = aliases.find(id);
    if (found == aliases.end()) throw std::invalid_argument("Unknown NVIDIA GPU: " + id);
    if (!seen.insert(found->second).second) throw std::invalid_argument("GPU IDs refer to the same device");
    result.push_back(found->second);
  }
  return result;
}

inline void selectSingleGpu(DeviceSelection const& selection) {
  if (selection.ids.size() != 1) throw std::invalid_argument("Expected exactly one GPU");
  // Resolve physical IDs exactly as the multi-GPU coordinator does. Use UUIDs
  // to avoid CUDA ordinal/PCI-order ambiguity; do not initialize CUDA here.
  auto closeFile = [](std::FILE* file) { std::fclose(file); };
  std::unique_ptr<std::FILE, decltype(closeFile)> log(std::tmpfile(), closeFile);
  if (!log) throw std::runtime_error("Cannot create GPU query scratch file");
  int fd = fileno(log.get());
  auto child = fork();
  if (child < 0) throw std::runtime_error("Cannot start NVIDIA GPU query");
  if (child == 0) {
    if (dup2(fd, STDOUT_FILENO) < 0 || dup2(fd, STDERR_FILENO) < 0) _exit(126);
    if (fd > STDERR_FILENO) close(fd);
    execlp("nvidia-smi", "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits", nullptr);
    execl("/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits", nullptr);
    _exit(127);
  }
  int status = 0;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  for (;;) {
    auto done = waitpid(child, &status, WNOHANG);
    if (done == child) break;
    if (done < 0 && errno != EINTR) throw std::runtime_error("NVIDIA GPU query wait failed");
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(child, SIGKILL);
      while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
      throw std::runtime_error("NVIDIA GPU query timed out");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("Cannot query NVIDIA GPUs with nvidia-smi");
  std::rewind(log.get());
  std::string listing; char buffer[512];
  while (std::fgets(buffer, sizeof(buffer), log.get())) listing += buffer;
  auto uuid = resolveGpuIds(selection.ids, listing).front();
  if (setenv("CUDA_VISIBLE_DEVICES", uuid.c_str(), 1) != 0)
    throw std::runtime_error("Cannot select NVIDIA GPU");
}
} // namespace corsika::applications::gpu_cli
