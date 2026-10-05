// C++ process management; children exec the same air-shower binary.
#include "NativeCoordinator.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;

namespace c8::multigpu {
namespace {
using Clock=std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted=0;
void signalHandler(int signal) { interrupted=signal; }
void require(bool ok,std::string const& message) { if(!ok) throw std::runtime_error(message); }
std::string number(double x) { std::ostringstream out; out<<std::setprecision(17)<<x; return out.str(); }
Json readJson(fs::path const& path) {
  std::ifstream in(path); require(bool(in),"Cannot read "+path.string()); return Json::parse(in);
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
struct Child {
  pid_t pid; fs::path folder; Json record; bool reaped=false; Clock::time_point started;
};
struct Processes {
  std::vector<Child> children; Clock::time_point started=Clock::now();
  ~Processes() {
    // Only our own sessions; keep child PIDs unreaped until signals are sent.
    for(auto const& c:children) if(!c.reaped) kill(-c.pid,SIGTERM);
    auto deadline=Clock::now()+std::chrono::seconds(3);
    for(;;) {
      bool pending=false;
      for(auto& c:children) if(!c.reaped) {
        int status;
        if(waitpid(c.pid,&status,WNOHANG)==c.pid) c.reaped=true;
        else pending=true;
      }
      if(!pending || Clock::now()>=deadline) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for(auto& c:children) if(!c.reaped) {
      kill(-c.pid,SIGKILL); while(waitpid(c.pid,nullptr,0)<0 && errno==EINTR) {}
    }
  }
  std::size_t launch(std::vector<std::string> const& command,fs::path const& folder,
                     std::map<std::string,std::string> const& overrides) {
    checkInterrupt();
    require(fs::create_directory(folder),"Process folder exists: "+folder.string());
    save(folder/"COMMAND.json",command);
    std::map<std::string,std::string> env;
    for(char** p=environ;*p;++p) {
      std::string s(*p); auto eq=s.find('=');
      if(eq!=std::string::npos) env[s.substr(0,eq)]=s.substr(eq+1);
    }
    for(auto const& e:overrides) env[e.first]=e.second;
    std::vector<std::string> storage;
    for(auto const& e:env) storage.push_back(e.first+"="+e.second);
    std::vector<char*> argv,envp;
    for(auto const& a:command) argv.push_back(const_cast<char*>(a.c_str()));
    for(auto& a:storage) envp.push_back(a.data());
    argv.push_back(nullptr); envp.push_back(nullptr);
    int log=open((folder/"run.log").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0644);
    require(log>=0,"Cannot create process log");
    auto parent=getpid(); auto before=Clock::now(); pid_t pid=fork();
    if(pid==0) {
      // No allocations, shell, Python, or CUDA operations between fork and exec.
      if(setsid()<0 || chdir(folder.c_str())<0 || dup2(log,1)<0 || dup2(log,2)<0) _exit(126);
      close(log); int input=open("/dev/null",O_RDONLY);
      if(input<0 || dup2(input,0)<0) _exit(126);
      if(input!=0) close(input);
      prctl(PR_SET_PDEATHSIG,SIGTERM);
      if(getppid()!=parent) _exit(126);
      execve(command.front().c_str(),argv.data(),envp.data()); _exit(127);
    }
    close(log); require(pid>0,"fork failed");
    children.push_back({pid,folder,{{"name",folder.filename().string()},{"pid",pid},
      {"start_s",std::chrono::duration<double>(before-started).count()}},false,before});
    save(folder/"OWNED_PROCESS.json",{{"pid",pid}});
    return children.size()-1;
  }
  void wait(std::vector<std::size_t> pending,double timeout) {
    while(!pending.empty()) {
      checkInterrupt();
      for(auto it=pending.begin();it!=pending.end();) {
        auto& c=children[*it]; int status=0; auto rc=waitpid(c.pid,&status,WNOHANG);
        if(rc<0 && errno==EINTR) continue;
        require(rc>=0,"waitpid failed");
        if(rc==c.pid) {
          c.reaped=true;
          int code=WIFEXITED(status)?WEXITSTATUS(status):-WTERMSIG(status);
          c.record["returncode"]=code; c.record["end_s"]=seconds(started); c.record["wall_s"]=seconds(c.started);
          save(c.folder/"PROCESS_EXIT.json",c.record);
          require(code==0,"Process failed: "+c.folder.string()+" (exit "+std::to_string(code)+")");
          it=pending.erase(it);
        } else {
          require(seconds(c.started)<=timeout,"Process deadline exceeded: "+c.folder.string()); ++it;
        }
      }
      if(!pending.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
  }
  Json records() const {
    Json out=Json::array(); for(auto const& c:children) out.push_back(c.record); return out;
  }
};
std::string trim(std::string s) {
  auto first=s.find_first_not_of(" \t\r\n");
  return first==std::string::npos?"":s.substr(first,s.find_last_not_of(" \t\r\n")-first+1);
}
fs::path executableOnPath(std::string const& name) {
  std::istringstream path(std::getenv("PATH")?std::getenv("PATH"):"");
  std::string folder;
  while(std::getline(path,folder,':')) {
    auto p=fs::absolute((folder.empty()?fs::path("."):fs::path(folder))/name);
    if(access(p.c_str(),X_OK)==0) return p;
  }
  if(name=="nvidia-smi" && access("/usr/lib/wsl/lib/nvidia-smi",X_OK)==0) return "/usr/lib/wsl/lib/nvidia-smi";
  throw std::runtime_error("Executable not found: "+name);
}
std::vector<std::string> resolveDevices(std::vector<std::string> const& requested,fs::path const& output) {
  Processes p;
  auto i=p.launch({executableOnPath("nvidia-smi").string(),"--query-gpu=index,uuid","--format=csv,noheader,nounits"},
                  output/"device_query",{});
  p.wait({i},15);
  std::ifstream in(output/"device_query/run.log"); std::string line;
  std::map<std::string,std::string> aliases;
  while(std::getline(in,line)) {
    auto comma=line.find(','); if(comma==std::string::npos) continue;
    auto index=trim(line.substr(0,comma)),uuid=trim(line.substr(comma+1));
    if(uuid.rfind("GPU-",0)!=0) continue;
    aliases[index]=aliases[uuid]=uuid;
  }
  std::vector<std::string> result; std::set<std::string> seen;
  for(auto const& id:requested) {
    auto it=aliases.find(id); require(it!=aliases.end(),"Unknown GPU: "+id);
    require(seen.insert(it->second).second,"GPU IDs refer to the same device"); result.push_back(it->second);
  }
  return result;
}
void feedCheck(fs::path const& folder,std::size_t expected) {
  auto feed=readJson(folder/"FRONTIER_FEED.json");
  auto maximum=feed.at("maximum_imported_batch").get<std::size_t>();
  require(feed.at("mode")=="bounded-high-energy-first" && feed.at("roots_consumed").get<std::size_t>()==expected &&
    maximum>0 && maximum<=65536 && feed.at("batches").get<std::size_t>()>0,"Incomplete frontier consumption: "+folder.string());
}
void event(Options const& o,fs::path const& root,std::uint64_t seed,std::vector<std::string> const& devices) {
  Processes p; auto physics=o.physics;
  physics.insert(physics.end(),{"-E",number(o.energy)});
  auto common=physics;
  common.insert(common.end(),{"-N","1","--hadronic-workers","1","--hadronic-backend","scalar"});
  if(o.emBackend=="egs4")common.insert(common.end(),{"--egs4-stepfc",number(o.egs4Stepfc)});
  std::map<std::string,std::string> env{{"OMP_NUM_THREADS","1"},{"OMP_THREAD_LIMIT","1"},
    {"OPENBLAS_NUM_THREADS","1"},{"MKL_NUM_THREADS","1"},{"CUDA_VISIBLE_DEVICES",devices.front()}};
  auto launch=[&](std::string const& name,std::vector<std::string> const& options,
                  std::map<std::string,std::string> const& environment) {
    std::vector<std::string> args{o.worker.string()};
    args.insert(args.end(),common.begin(),common.end());
    args.insert(args.end(),{"-f",(root/name/"shower").string()});
    args.insert(args.end(),options.begin(),options.end());
    return p.launch(args,root/name,environment);
  };
  try {
    save(root/"CONFIG.json",{{"command",{o.worker.string()}},{"physics_args",physics},{"seed",seed},
      {"devices",devices},{"gpu_memory_fraction",o.memoryFraction},
      {"em_backend",o.emBackend},{"egs4_stepfc",o.egs4Stepfc},
      {"coordinator","C++"},{"output",root.string()}});
    auto frontier=root/"frontier.txt";
    auto cap=o.frontierEnergy>0?o.frontierEnergy:o.energy/(8*devices.size());
    std::cout<<"Seed "<<seed<<": computing shower prefix"<<std::endl;
    auto prefix=launch("prefix",{"-s",std::to_string(seed),"--em-backend",o.emBackend=="egs4"?"egs4":"proposal","--radio-backend","cpu",
      "--frontier-out",frontier.string(),"--frontier-max-energy",number(cap)},env);
    p.wait({prefix},o.timeout); validateOutputs(root/"prefix/shower");
    std::vector<fs::path> shards;
    for(std::size_t i=0;i<devices.size();++i) shards.push_back(root/("roots_"+std::to_string(i)+".txt"));
    auto audit=partition(frontier,shards); save(root/"PARTITION.json",audit);
    std::vector<std::size_t> workers; std::vector<fs::path> parts{root/"prefix/shower"};
    std::cout<<"Seed "<<seed<<": "<<audit["counts"]<<" roots on "<<devices.size()<<" GPUs"<<std::endl;
    for(unsigned i=0;i<devices.size();++i) {
      if(audit["counts"][i]==0) continue;
      auto workerEnv=env; workerEnv["CUDA_VISIBLE_DEVICES"]=devices[i];
      std::vector<std::string> workerOptions{
        "-s",std::to_string(workerSeed(seed,i)),"--em-backend",o.emBackend,"--radio-backend","kokkos",
        "--kokkos-execution","cuda","--kokkos-num-threads","1","--kokkos-device","0",
        "--gpu-memory-fraction",number(o.memoryFraction),"--gpu-resident-batch-limit","0",
        "--frontier-in",shards[i].string(),"--frontier-worker-id",std::to_string(i+1)};
      workers.push_back(launch("worker_"+std::to_string(i),workerOptions,workerEnv));
      parts.push_back(root/("worker_"+std::to_string(i))/"shower");
    }
    p.wait(workers,o.timeout);
    // Per-process times are recorded at exit, before serial output validation.
    for(unsigned i=0;i<devices.size();++i) if(audit["counts"][i]!=0) {
      auto folder=root/("worker_"+std::to_string(i));
      validateOutputs(folder/"shower"); feedCheck(folder,audit["counts"][i].get<std::size_t>());
    }
    require(digest(frontier)==audit["frontier_sha256"],"Frontier changed while running");
    for(unsigned i=0;i<shards.size();++i) require(digest(shards[i])==audit["shard_sha256"][i],"Shard changed while running");
    auto mergeStart=Clock::now(); auto products=merge(parts,root/"merged");
    fs::create_directory(root/"merged/primary");
    for(auto name:{"config.yaml","summary.yaml"}) fs::copy_file(parts.front()/"primary"/name,root/"merged/primary"/name);
    save(root/"merged/config.yaml",{{"creator","CORSIKA8 native multi-GPU"},{"physics_args",physics},
      {"coordinator_config",(root/"CONFIG.json").string()},{"merge",(root/"merged/MERGE.json").string()}});
    save(root/"merged/summary.yaml",{{"showers",1},{"seed",seed},{"runtime_seconds",seconds(p.started)},
      {"experimental",true},{"statistical_acceptance",false},
      {"output_dirs",{"profile","production_profile","energyloss","particles","interactions","primary","CoREAS","ZHS"}}});
    checkInterrupt(); auto wall=seconds(p.started);
    save(root/"TIMING.json",{{"wall_s",wall},{"merge_s",seconds(mergeStart)},{"parts",p.records()}});
    save(root/"COMPLETE.json",{{"products",products},{"wall_s",wall},{"partition_sha256",digest(root/"PARTITION.json")},
      {"process_complete",true},{"physics_validated_against_beta2",false},{"experimental",true},{"coordinator","C++"}});
    std::cout<<"Seed "<<seed<<": complete in "<<wall<<" s; "<<root/"merged"<<std::endl;
  } catch(std::exception const& e) {
    save(root/"INCOMPLETE.json",{{"error",e.what()},{"parts",p.records()}}); throw;
  }
}
} // namespace
void checkInterrupt() {
  if(interrupted) throw std::runtime_error("Interrupted by signal "+std::to_string(interrupted));
}
void installSignalHandlers() {
  interrupted=0; struct sigaction action{}; action.sa_handler=signalHandler;
  sigemptyset(&action.sa_mask); sigaction(SIGINT,&action,nullptr); sigaction(SIGTERM,&action,nullptr);
}
void run(Options const& options) {
  auto o=options; o.output=fs::absolute(o.output).lexically_normal(); o.worker=fs::absolute(o.worker).lexically_normal();
  require(!o.devices.empty() && o.devices.size()<=255,"Select 1..255 distinct GPUs");
  require(std::isfinite(o.energy) && o.energy>0,"Positive fixed primary energy required");
  require(o.seed>0 && o.events>0 && o.seed<=std::uint64_t(std::numeric_limits<std::int64_t>::max())-o.events,"Invalid seed/event count");
  require(std::isfinite(o.memoryFraction) && o.memoryFraction>0 && o.memoryFraction<=.9,"Memory fraction must be in (0,0.9]");
  require(std::isfinite(o.frontierEnergy) && o.frontierEnergy>=0 && std::isfinite(o.timeout) && o.timeout>0,"Invalid scheduling cap/timeout");
  require(access(o.worker.c_str(),X_OK)==0,"Worker executable unavailable: "+o.worker.string());
  require(!fs::exists(o.output),"Output directory exists: "+o.output.string());
  if(o.output.has_parent_path()) fs::create_directories(o.output.parent_path());
  require(fs::create_directory(o.output),"Output directory exists");
  try {
    auto devices=resolveDevices(o.devices,o.output); save(o.output/"DEVICE_UUIDS.json",devices);
    auto start=Clock::now();
    if(o.events==1) event(o,o.output,o.seed,devices);
    else {
      Json events=Json::array();
      save(o.output/"CAMPAIGN.json",{{"events",o.events},{"first_seed",o.seed},{"seed_rule","seed + event_index"},
        {"devices",devices},{"coordinator","C++"}});
      for(unsigned i=0;i<o.events;++i) {
        auto seed=o.seed+i; auto folder=o.output/("seed_"+std::to_string(seed));
        require(fs::create_directory(folder),"Event directory exists"); event(o,folder,seed,devices);
        events.push_back({{"seed",seed},{"wall_s",readJson(folder/"COMPLETE.json")["wall_s"]},{"output",folder.string()}});
      }
      checkInterrupt();
      save(o.output/"TIMING.json",{{"wall_s",seconds(start)},{"events",events}});
      save(o.output/"COMPLETE.json",{{"process_complete",true},{"events",o.events},{"wall_s",seconds(start)},
        {"coordinator","C++"},{"physics_validated_against_beta2",false}});
    }
  } catch(std::exception const& e) {
    if(!fs::exists(o.output/"INCOMPLETE.json")) save(o.output/"INCOMPLETE.json",{{"error",e.what()}});
    throw;
  }
}
bool requested(int argc,char** argv) {
  for(int i=1;i<argc;++i) {
    std::string s(argv[i]); if(s=="--devices" || s.rfind("--devices=",0)==0) return true;
  }
  return false;
}
int mainEntry(int argc,char** argv) {
  Options o; std::string output,worker,devices,emBackend,radioBackend,execution;
  CLI::App app{"CORSIKA 8: each shower on independent CUDA GPUs; native C++ coordination."};
  app.allow_extras();
  app.add_option("--devices",devices,"Comma-separated physical GPU indices or UUIDs")->required();
  app.add_option("-f,--filename",output,"New output directory")->required();
  app.add_option("-E,--energy",o.energy,"Fixed primary energy [GeV]")->required();
  app.add_option("-s,--seed",o.seed,"First event seed; later seeds increment by one")->default_val(1);
  app.add_option("-N,--nevent",o.events,"Sequential showers, each using all selected GPUs")->default_val(1);
  app.add_option("--gpu-memory-fraction",o.memoryFraction,"Memory budget per GPU (0,0.9]")->default_val(.5);
  app.add_option("--multigpu-timeout",o.timeout,"Timeout seconds per prefix/worker")->default_val(7200);
  app.add_option("--multigpu-frontier-energy",o.frontierEnergy,"Scheduling cap [GeV]; 0=automatic")->default_val(0);
  app.add_option("--multigpu-worker",worker,"Optional compatible worker binary; default: this binary");
  app.add_option("--em-backend",emBackend,"Multi-GPU transport backend: kokkos or egs4");
  app.add_option("--egs4-stepfc",o.egs4Stepfc,"C7-EGS4 step factor (default: 1)")->check(CLI::PositiveNumber);
  app.add_option("--radio-backend",radioBackend,"Multi-GPU radio backend: kokkos");
  app.add_option("--kokkos-execution",execution,"Multi-GPU execution space: cuda");
  try {
    app.parse(argc,argv); o.output=output;
    require((emBackend.empty() || emBackend=="kokkos" || emBackend=="egs4") &&
            (radioBackend.empty() || radioBackend=="kokkos") &&
            (execution.empty() || execution=="cuda"), "Multi-GPU requires kokkos/egs4 transport, kokkos radio and cuda execution");
    o.emBackend=emBackend.empty()?"kokkos":emBackend;
#ifndef CORSIKA8_WITH_NATIVE_EGS4
    require(o.emBackend!="egs4","EGS4 is not compiled; build with CORSIKA_ENABLE_EGS4=ON");
#endif
    require(!app.count("--egs4-stepfc") || o.emBackend=="egs4","--egs4-stepfc requires --em-backend egs4");
    require(o.emBackend!="egs4" || !app.count("--multigpu-frontier-energy"),
      "EGS4 exports unstarted roots; --multigpu-frontier-energy applies only to PROPOSAL");
    o.worker=worker.empty()?fs::canonical("/proc/self/exe"):fs::absolute(worker);
    std::istringstream ids(devices); std::string id;
    while(std::getline(ids,id,',')) { id=trim(id); require(!id.empty(),"Empty GPU ID"); o.devices.push_back(id); }
    require(!devices.empty() && devices.back()!=',',"Empty GPU ID");
    o.physics=app.remaining();
    std::set<std::string> reserved{"--energy_range","--force-interaction","--force-decay","--em-backend",
      "--radio-backend","--kokkos-execution","--kokkos-device","--kokkos-num-threads","--hadronic-workers",
      "--hadronic-backend","--gpu-resident-batch-limit","--compress"};
    for(std::size_t i=0;i<o.physics.size();++i) {
      auto key=o.physics[i].substr(0,o.physics[i].find('='));
      require(!reserved.count(key) && key.rfind("--frontier-",0)!=0 && key!="--",
        "Multi-GPU coordinator owns or does not support option: "+key);
      if(key=="--antenna-file" || key=="--gpu-aux-cache-dir" || key=="--kokkos-tuning-cache") {
        auto eq=o.physics[i].find('=');
        if(eq==std::string::npos) {
          require(i+1<o.physics.size(),key+" needs a path"); ++i; o.physics[i]=fs::absolute(o.physics[i]).string();
        } else o.physics[i]=key+"="+fs::absolute(o.physics[i].substr(eq+1)).string();
      }
    }
    if(auto data=std::getenv("CORSIKA_DATA")) setenv("CORSIKA_DATA",fs::absolute(data).c_str(),1);
    installSignalHandlers(); run(o); return 0;
  } catch(CLI::ParseError const& e) { return app.exit(e); }
    catch(std::exception const& e) { std::cerr<<"Multi-GPU: "<<e.what()<<'\n'; return interrupted?128+interrupted:1; }
}
} // namespace c8::multigpu
