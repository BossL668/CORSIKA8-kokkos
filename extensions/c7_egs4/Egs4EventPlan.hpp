#pragma once
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace c7_egs4::application {
struct PlannedEvent {
  std::uint64_t seed{};
  std::string output,capture,frontier,ready;
};
inline std::vector<PlannedEvent> readEventPlan(std::string const& path,PlannedEvent single) {
  if(path.empty())return {single};
  std::ifstream f(path);nlohmann::json plan;
  if(!f||!(f>>plan)||!plan.is_array()||plan.empty()||plan.size()>10000)
    throw std::invalid_argument("Invalid persistent event plan");
  std::set<std::string> outputs,captures;std::vector<PlannedEvent> result;
  for(auto const& row:plan) {
    PlannedEvent e{row.at("seed").get<std::uint64_t>(),row.at("output").get<std::string>(),
      row.value("capture",std::string{}),row.value("frontier",std::string{}),row.value("ready",std::string{})};
    for(auto const& p:{e.output,e.capture,e.frontier,e.ready})
      if(!p.empty()&&!std::filesystem::path(p).is_absolute())throw std::invalid_argument("Event plan requires absolute paths");
    if(e.output.empty()||!outputs.insert(e.output).second||std::filesystem::exists(e.output))
      throw std::invalid_argument("Event output exists or repeats in plan");
    if(!e.capture.empty()&&(!captures.insert(e.capture).second||std::filesystem::exists(e.capture)))
      throw std::invalid_argument("Event frontier capture exists or repeats");
    if(!result.empty()&&(e.capture.empty()!=result.front().capture.empty()||e.frontier.empty()!=result.front().frontier.empty()))
      throw std::invalid_argument("Persistent process role cannot change between events");
    if(!e.capture.empty()&&!e.frontier.empty())throw std::invalid_argument("Conflicting persistent roles");
    result.push_back(std::move(e));
  }
  return result;
}
inline double awaitEvent(PlannedEvent const& e,double timeout) {
  auto start=std::chrono::steady_clock::now();
  if(!e.ready.empty())while(!std::filesystem::exists(e.ready)) {
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>timeout)
      throw std::runtime_error("Persistent event readiness timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
inline void completeEvent(PlannedEvent const& e,std::size_t index) {
  auto target=std::filesystem::path(e.output)/"EVENT_DONE.json";
  if(std::filesystem::exists(target))throw std::runtime_error("Persistent event already acknowledged");
  auto temporary=target;temporary+=".tmp";
  {std::ofstream f(temporary);f<<nlohmann::json{{"seed",e.seed},{"event_index",index},{"pid",getpid()}};
    f.close();if(!f)throw std::runtime_error("Cannot acknowledge persistent event");}
  std::filesystem::rename(temporary,target);
}
} // namespace c7_egs4::application
