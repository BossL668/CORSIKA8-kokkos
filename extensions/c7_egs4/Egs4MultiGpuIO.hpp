#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
namespace c7_egs4::multigpu {
namespace fs=std::filesystem;using Json=nlohmann::json;
void checkInterrupt();
void save(fs::path const&,Json const&);
std::string digest(fs::path const&);
Json partition(fs::path const&,std::vector<fs::path> const&,bool allow_muons=false);
void validateOutputs(fs::path const&);
Json merge(std::vector<fs::path> const&,fs::path const&);
}
