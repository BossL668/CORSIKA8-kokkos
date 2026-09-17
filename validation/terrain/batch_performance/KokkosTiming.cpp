// Optional, bounded OpenMP profiling tool; no physics changes or per-step files.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <string>
namespace {
using Clock=std::chrono::steady_clock;
struct Entry { std::uint64_t calls{}; double seconds{}; };
struct Active { std::string name; Clock::time_point start; };
std::mutex lock;
std::map<std::string,Entry> totals;
std::map<std::uint64_t,Active> active;
std::uint64_t nextId=1;
void begin(char const* kind,char const* name,std::uint64_t* id) {
  std::lock_guard<std::mutex> guard(lock);
  *id=nextId++;active.emplace(*id,Active{std::string(kind)+":"+name,Clock::now()});
}
void end(std::uint64_t id) {
  auto now=Clock::now();std::lock_guard<std::mutex> guard(lock);
  auto i=active.find(id);if(i==active.end())return;
  auto& t=totals[i->second.name];++t.calls;
  t.seconds+=std::chrono::duration<double>(now-i->second.start).count();active.erase(i);
}
}
extern "C" {
void kokkosp_init_library(int,std::uint64_t,std::uint32_t,void*) {}
void kokkosp_begin_parallel_for(char const* n,std::uint32_t,std::uint64_t* id){begin("for",n,id);}
void kokkosp_end_parallel_for(std::uint64_t id){end(id);}
void kokkosp_begin_parallel_scan(char const* n,std::uint32_t,std::uint64_t* id){begin("scan",n,id);}
void kokkosp_end_parallel_scan(std::uint64_t id){end(id);}
void kokkosp_begin_parallel_reduce(char const* n,std::uint32_t,std::uint64_t* id){begin("reduce",n,id);}
void kokkosp_end_parallel_reduce(std::uint64_t id){end(id);}
void kokkosp_finalize_library(){
  auto path=std::getenv("C8_BATCH_PROFILE_FILE");if(!path)return;
  std::ofstream f(path);f<<"kernel,calls,host_call_seconds\n"<<std::setprecision(17);
  for(auto const& item:totals)f<<std::quoted(item.first)<<','<<item.second.calls<<','<<item.second.seconds<<'\n';
}
}
