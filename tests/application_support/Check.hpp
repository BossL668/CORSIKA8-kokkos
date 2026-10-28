/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline unsigned checks = 0;
inline void check(bool value, char const* expression) {
  ++checks;
  if (!value) throw std::runtime_error(expression);
}
#define CHECK(...) check(bool(__VA_ARGS__), #__VA_ARGS__)

struct TempDirectory {
  std::filesystem::path path;
  TempDirectory() {
    std::string text = "/tmp/c8-host-support-XXXXXX";
    auto* created = mkdtemp(text.data());
    if (!created) throw std::runtime_error("mkdtemp failed");
    path = created;
  }
  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

struct EnvironmentValue {
  std::string name;
  std::string old;
  bool existed;
  EnvironmentValue(std::string key, std::string value)
      : name(std::move(key)) {
    auto* before = std::getenv(name.c_str());
    existed = before != nullptr;
    if (before) old = before;
    if (setenv(name.c_str(), value.c_str(), 1)) throw std::runtime_error("setenv failed");
  }
  ~EnvironmentValue() {
    if (existed)
      setenv(name.c_str(), old.c_str(), 1);
    else
      unsetenv(name.c_str());
  }
};

template <typename F>
std::string outcome(F&& run) {
  try {
    run();
    return "success";
  } catch (std::invalid_argument const& e) {
    return "invalid_argument:" + std::string(e.what());
  } catch (std::out_of_range const& e) {
    return "out_of_range:" + std::string(e.what());
  } catch (std::exception const& e) { return "exception:" + std::string(e.what()); }
}
