// Experimental single-shower static frontier coordinator; no Python runtime.
// Does not initialize Kokkos/CUDA: device visibility is fixed before exec.
#include "Egs4MultiGpuIO.hpp"
#include "Egs4FeReference.hpp"
#include <CLI/CLI.hpp>
#include <csignal>
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <thread>
#include <sys/wait.h>
#include <unistd.h>
namespace c7_egs4::multigpu {
volatile std::sig_atomic_t interrupted=0;
void checkInterrupt(){if(interrupted)throw std::runtime_error("Native EGS4 coordinator interrupted");}
}
namespace {
using namespace c7_egs4::multigpu;
void require(bool value,std::string const& message){if(!value)throw std::runtime_error(message);}
struct Children {
  std::set<pid_t> live;
  pid_t start(std::vector<std::string> const& args,fs::path const& log,fs::path const& cwd,std::string const& device) {
    checkInterrupt();int fd=open(log.c_str(),O_WRONLY|O_CREAT|O_EXCL,0644);
    require(fd>=0,"Cannot create subprocess log");
    auto pid=fork();if(pid<0){close(fd);throw std::runtime_error("fork failed");}
    if(pid==0) {
      setpgid(0,0);
      if(chdir(cwd.c_str())||dup2(fd,STDOUT_FILENO)<0||dup2(fd,STDERR_FILENO)<0)_exit(125);
      close(fd);setenv("OMP_NUM_THREADS","1",1);setenv("OMP_PROC_BIND","false",1);
      if(!device.empty())setenv("CUDA_VISIBLE_DEVICES",device.c_str(),1);
      std::vector<char*> argv;for(auto const& a:args)argv.push_back(const_cast<char*>(a.c_str()));argv.push_back(nullptr);
      execvp(argv[0],argv.data());_exit(127);
    }
    close(fd);setpgid(pid,pid);live.insert(pid);return pid;
  }
  void poll() {
      checkInterrupt();
      for(auto it=live.begin();it!=live.end();) {
        int status=0;pid_t done=waitpid(*it,&status,WNOHANG);
        if(done<0&&errno==EINTR){++it;continue;}
        require(done>=0,"Lost owned worker");
        if(done){it=live.erase(it);require(WIFEXITED(status)&&WEXITSTATUS(status)==0,
          "Worker pid="+std::to_string(done)+(WIFSIGNALED(status)?" signal="+std::to_string(WTERMSIG(status)):
          " exit="+std::to_string(WIFEXITED(status)?WEXITSTATUS(status):-1))+"; outputs retained, no completed merge");}
        else ++it;
      }
  }
  void wait(double seconds) {
    auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
    while(!live.empty()) {
      require(std::chrono::steady_clock::now()<deadline,"Native EGS4 worker timeout");poll();
      if(!live.empty())std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  void event(std::vector<fs::path> const& outputs,double seconds,std::uint64_t seed,std::size_t index) {
    auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
    while(true) {
      poll();bool ready=true;
      for(auto const& out:outputs)if(!fs::exists(out/"EVENT_DONE.json"))ready=false;
      if(ready)break;
      require(!live.empty(),"All workers exited without event completion");
      require(std::chrono::steady_clock::now()<deadline,"Persistent event timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    for(auto const& out:outputs) {
      std::ifstream f(out/"EVENT_DONE.json");Json j;f>>j;
      require(j.at("seed")==seed&&j.at("event_index")==index,"Wrong persistent event acknowledgment");
    }
  }
  ~Children(){
    for(auto pid:live)kill(-pid,SIGTERM);
    if(!live.empty())std::this_thread::sleep_for(std::chrono::milliseconds(200));
    for(auto pid:live){kill(-pid,SIGKILL);int status;while(waitpid(pid,&status,0)<0&&errno==EINTR){}}
  }
};
Json metadata(fs::path const& path){std::ifstream f(path);require(bool(f),"Missing completion metadata");Json j;f>>j;return j;}
std::string number(double value){std::ostringstream s;s<<std::setprecision(17)<<value;return s.str();}
void verifyPhysics(Json const& prefix,Json const& worker) {
  require(worker.at("status")=="completed"&&!worker.at("prefix_only").get<bool>()&&worker.at("scalar_em_steps")==0,"Invalid worker completion");
  for(auto const* key:{"primary_pdg","primary_energy_GeV","seed","height_m","stepfc","source_count","mass_convention",
      "atmosphere","observation_height_m","sea_level_refractive_index","magnetic_field_T","native_electron_kinetic_cut_MeV",
      "native_photon_cut_MeV","hadron_kinetic_cut_GeV","muon_kinetic_cut_GeV","hadronic_models","muon_transport",
      "thin_threshold_GeV","thin_max_weight","fe_reference","zenith_deg","azimuth_deg","antenna_count",
      "profile_bin_g_cm2","profile_crossings","refractivity_normalization","gpu_memory_fraction","radio_backend_requested","muon_backend_requested"})require(prefix.at(key)==worker.at(key),std::string("Worker setting mismatch: ")+key);
}
}
int main(int argc,char** argv){try {
  using namespace c7_egs4::multigpu;
  CLI::App cli{"Experimental native EGS4 static single-shower schedule (no production main integration)"};
  std::string worker,table,destination,devices;unsigned cpu_workers=0;int pdg=2212,source_count=1;
  double energy=10.,height=5000.,stepfc=1.,thin=0.,max_weight=1.,timeout=3600.;std::uint64_t seed=731;
  c7_egs4::application::FeReference reference;reference.options(cli);
  bool force_interaction=false,force_decay=false;
  std::string radio_backend="kokkos";
  std::string muon_backend="cpu";
  std::string profile_backend="cpu",queue_workspace="fresh";
  bool split_rare_kernels=false;
  bool persistent=false;unsigned events=1;std::string event_seeds;
  cli.add_flag("--persistent",persistent,"Keep prefix and device workers/models/queues alive across showers");
  cli.add_option("--events",events)->check(CLI::Range(1,10000));
  cli.add_option("--event-seeds",event_seeds,"Ordered campaign GPU seeds; C8 host streams continue across events");
  cli.add_flag("--split-rare-kernels",split_rare_kernels);
  cli.add_option("--profile-backend",profile_backend)->check(CLI::IsMember({"cpu","kokkos"}));
  cli.add_option("--queue-workspace",queue_workspace)->check(CLI::IsMember({"fresh","reuse"}));
  double muon_cut=-1.;
  cli.add_option("--muon-backend",muon_backend)->check(CLI::IsMember({"cpu","kokkos"}));
  cli.add_option("--muon-cut-GeV",muon_cut)->check(CLI::PositiveNumber);
  cli.add_option("--radio-backend",radio_backend)->check(CLI::IsMember({"kokkos","cpu"}));
  double gpu_memory_fraction=0.;int frontier_batch=0,queue_capacity=0;
  cli.add_option("--gpu-memory-fraction",gpu_memory_fraction,"Per-device total-memory budget, bounded by available memory")
    ->check(CLI::Range(.01,1.));
  cli.add_option("--frontier-batch",frontier_batch)->check(CLI::Range(1,std::numeric_limits<int>::max()/10));
  cli.add_option("--queue-capacity",queue_capacity)->check(CLI::Range(8192,std::numeric_limits<int>::max()/5));
  cli.add_option("--worker",worker)->required();cli.add_option("--table",table)->required();
  cli.add_option("-f,--output",destination)->required();cli.add_option("--devices",devices);
  cli.add_option("--cpu-workers",cpu_workers,"Test topology only; not a multi-GPU result")->check(CLI::Range(1,4));
  cli.add_option("--pdg",pdg);cli.add_option("--energy-GeV",energy)->check(CLI::PositiveNumber);
  cli.add_option("--height-m",height)->check(CLI::Range(1100.01,112750.));cli.add_option("--stepfc",stepfc)->check(CLI::PositiveNumber);
  cli.add_option("--source-count",source_count)->check(CLI::Range(1,1024));cli.add_option("--seed",seed);
  cli.add_option("--thin-threshold-GeV",thin)->check(CLI::NonNegativeNumber);cli.add_option("--thin-max-weight",max_weight)->check(CLI::PositiveNumber);
  cli.add_option("--timeout",timeout)->check(CLI::PositiveNumber);cli.add_flag("--force-interaction",force_interaction);cli.add_flag("--force-decay",force_decay);
  CLI11_PARSE(cli,argc,argv);
  reference.defaults(cli,pdg,energy,height,thin,max_weight);
  if(reference.enabled)reference.antennas=fs::canonical(reference.antennas).string();
  auto start=std::chrono::steady_clock::now();
  require((cpu_workers>0)!=(!devices.empty()),"Choose either explicit devices or CPU test workers");
  require(!cpu_workers||gpu_memory_fraction==0.,"GPU memory fraction is not applicable to CPU EM workers");
  require(!(force_interaction&&force_decay),"Conflicting forced primary actions");
  worker=fs::canonical(worker).string();table=fs::canonical(table).string();
  fs::path root=fs::absolute(destination);require(fs::create_directory(root),"Destination exists");
  std::signal(SIGINT,[](int s){interrupted=s;});std::signal(SIGTERM,[](int s){interrupted=s;});
  Children jobs;std::vector<std::string> gpu;
  if(!devices.empty()) {
    std::stringstream fields(devices);std::string token;std::set<std::string> seen;
    while(std::getline(fields,token,',')){require(!token.empty(),"Empty GPU selector");gpu.push_back(token);}
    require(!gpu.empty()&&gpu.size()<=4,"Use one to four explicit GPUs");
    // Resolve every index/UUID with the driver BEFORE spawning workers, so
    // selectors 0 and GPU-... cannot accidentally lease the same GPU twice.
    for(std::size_t i=0;i<gpu.size();++i) {
      auto log=root/("device_"+std::to_string(i)+".log");
      jobs.start({"nvidia-smi","-i",gpu[i],"--query-gpu=uuid","--format=csv,noheader"},log,root,"");jobs.wait(10.);
      std::ifstream in(log);std::string uuid,extra;std::getline(in,uuid);
      require(uuid.rfind("GPU-",0)==0&&!std::getline(in,extra),"Expected exactly one physical GPU UUID");
      require(seen.insert(uuid).second,"Duplicate physical GPU");gpu[i]=uuid;
    }
  }else gpu.resize(cpu_workers);
  std::vector<std::string> common{worker,"--table",table,"--pdg",std::to_string(pdg),"--energy-GeV",number(energy),
    "--height-m",number(height),"--stepfc",number(stepfc),"--seed",std::to_string(seed),
    "--source-count",std::to_string(source_count),"--thin-threshold-GeV",number(thin),"--thin-max-weight",number(max_weight)};
  common.insert(common.end(),{"--radio-backend",radio_backend});
  if(cpu_workers)common.push_back("--host-reference");
  common.insert(common.end(),{"--muon-backend",muon_backend});
  common.insert(common.end(),{"--profile-backend",profile_backend,"--queue-workspace",queue_workspace});
  if(split_rare_kernels)common.push_back("--split-rare-kernels");
  if(cli.count("--muon-cut-GeV"))common.insert(common.end(),{"--muon-cut-GeV",number(muon_cut)});
  if(gpu_memory_fraction>0.)common.insert(common.end(),{"--gpu-memory-fraction",number(gpu_memory_fraction)});
  if(frontier_batch)common.insert(common.end(),{"--frontier-batch",std::to_string(frontier_batch)});
  if(queue_capacity)common.insert(common.end(),{"--queue-capacity",std::to_string(queue_capacity)});
  if(reference.enabled)common.insert(common.end(),{"--fe-reference","--antenna-file",reference.antennas});
  std::vector<std::uint64_t> seeds;
  if(event_seeds.empty()) {
    require(seed<=std::numeric_limits<std::uint64_t>::max()-(events-1),"Event seed overflow");
    for(unsigned i=0;i<events;++i)seeds.push_back(seed+i);
  }else {
    std::stringstream in(event_seeds);std::string item;
    while(std::getline(in,item,',')) {
      require(!item.empty()&&item.find_first_not_of("0123456789")==std::string::npos,"Invalid event seed");
      seeds.push_back(std::stoull(item));
    }
    require(!seeds.empty()&&seeds.size()<=10000,"Invalid event count");
    if(cli.count("--events"))require(seeds.size()==events,"Event seed/count mismatch");
  }
  require(persistent||seeds.size()==1,"Multiple events require --persistent");
  require(persistent||event_seeds.empty(),"Explicit event seed list requires --persistent");
  std::vector<fs::path> event_roots;
  if(persistent) {
    Json prefix_plan=Json::array();std::vector<Json> worker_plans(gpu.size(),Json::array());
    for(std::size_t n=0;n<seeds.size();++n) {
      auto e=root/("event_"+std::to_string(n)+"_seed_"+std::to_string(seeds[n]));
      require(fs::create_directory(e),"Event destination exists");event_roots.push_back(e);
      prefix_plan.push_back({{"seed",seeds[n]},{"output",(e/"prefix").string()},
        {"capture",(e/"frontier.txt").string()},{"ready",(e/"prefix.READY").string()}});
      for(std::size_t i=0;i<gpu.size();++i)worker_plans[i].push_back({{"seed",seeds[n]},
        {"output",(e/("worker_"+std::to_string(i+1))).string()},
        {"frontier",(e/("shard_"+std::to_string(i+1)+".txt")).string()},
        {"ready",(e/"workers.READY").string()}});
    }
    save(root/"prefix_plan.json",prefix_plan);
    auto args=common;args.insert(args.end(),{"-f",prefix_plan[0]["output"].get<std::string>(),
      "--capture-frontier",prefix_plan[0]["capture"].get<std::string>(),
      "--event-plan",(root/"prefix_plan.json").string(),"--event-timeout",number(timeout)});
    if(!cpu_workers)args.push_back("--host-reference");
    if(force_interaction)args.push_back("--force-interaction");if(force_decay)args.push_back("--force-decay");
    fs::create_directory(root/"prefix_work");jobs.start(args,root/"prefix.log",root/"prefix_work",gpu.front());
    for(std::size_t i=0;i<gpu.size();++i) {
      auto plan=root/("worker_plan_"+std::to_string(i+1)+".json");save(plan,worker_plans[i]);
      args=common;args.insert(args.end(),{"-f",worker_plans[i][0]["output"].get<std::string>(),
        "--frontier",worker_plans[i][0]["frontier"].get<std::string>(),"--worker-id",std::to_string(i+1),
        "--event-plan",plan.string(),"--event-timeout",number(timeout)});
      auto cwd=root/("worker_work_"+std::to_string(i+1));fs::create_directory(cwd);
      jobs.start(args,root/("worker_"+std::to_string(i+1)+".log"),cwd,gpu[i]);
    }
  }else event_roots.push_back(root);
  auto campaign_root=root;auto campaign_start=start;Json event_summaries=Json::array();
  std::vector<int> persistent_pids;
  for(std::size_t event_index=0;event_index<event_roots.size();++event_index) {
  root=event_roots[event_index];start=std::chrono::steady_clock::now();
  auto frontier=root/"frontier.txt";auto prefix=root/"prefix";
  auto command=common;command.insert(command.end(),{"-f",prefix.string(),"--capture-frontier",frontier.string(),"--host-reference"});
  // Avoid passing the same flag twice in the CPU test path.
  if(cpu_workers)command.pop_back();
  if(force_interaction)command.push_back("--force-interaction");if(force_decay)command.push_back("--force-decay");
  if(persistent) {
    save(root/"prefix.READY",{{"seed",seeds[event_index]},{"event_index",event_index}});
    jobs.event({prefix},timeout,seeds[event_index],event_index);
  }else {
    fs::create_directory(root/"prefix_work");jobs.start(command,root/"prefix.log",root/"prefix_work",gpu.front());jobs.wait(timeout);
  }
  auto prefix_finished=std::chrono::steady_clock::now();
  auto initial=metadata(prefix/"native_egs4_run.json");require(initial.at("prefix_only").get<bool>()&&initial.at("scalar_em_steps")==0,"Invalid prefix");
  validateOutputs(prefix);
  std::vector<fs::path> shards,parts{prefix};for(std::size_t i=0;i<gpu.size();++i)shards.push_back(root/("shard_"+std::to_string(i+1)+".txt"));
  auto split=partition(frontier,shards,muon_backend=="kokkos");save(root/"PARTITION.json",split);
  for(std::size_t i=0;i<gpu.size();++i) {
    auto output=root/("worker_"+std::to_string(i+1));parts.push_back(output);
    if(persistent)continue;
    auto cwd=root/("worker_work_"+std::to_string(i+1));fs::create_directory(cwd);
    auto args=common;args.insert(args.end(),{"-f",output.string(),"--frontier",shards[i].string(),"--worker-id",std::to_string(i+1)});
    jobs.start(args,root/("worker_"+std::to_string(i+1)+".log"),cwd,gpu[i]);
  }
  if(persistent) {
    save(root/"workers.READY",{{"seed",seeds[event_index]},{"event_index",event_index}});
    jobs.event(std::vector<fs::path>(parts.begin()+1,parts.end()),timeout,seeds[event_index],event_index);
  }else jobs.wait(timeout);
  auto workers_finished=std::chrono::steady_clock::now();std::uint64_t count=0;Json stats=Json::array();
  for(std::size_t i=1;i<parts.size();++i) {
    auto m=metadata(parts[i]/"native_egs4_run.json");verifyPhysics(initial,m);
    require(m.at("profile_backend_requested")==profile_backend&&m.at("queue_workspace")==queue_workspace,
      "Worker profile/queue mode mismatch");
    if(profile_backend=="kokkos") {
      require(m.at("profile_accumulated_steps")==m.at("native_steps")&&m.at("profile_fixed_point_overflows")==0&&
        m.at("profile_invalid_records")==0,"Device profile failed its output ledger");
      if(!cpu_workers&&m.at("native_steps").get<std::uint64_t>())
        require(m.at("profile_execution_space")=="Cuda","Requested profile did not use CUDA");
    }
    require(m.at("worker_id")==i,"Worker identity mismatch");
    require(m.at("frontier_roots_consumed")==split.at("counts").at(i-1),"Worker did not consume its complete shard");
    if(!cpu_workers)require(m.at("execution_space")=="Cuda","Requested GPU worker did not use CUDA");
    if(radio_backend=="kokkos") {
      require(m.at("radio_projected_tracks")==m.at("native_radio_tracks")&&m.at("radio_fixed_point_overflows")==0,
        "Radio did not process every native segment exactly once");
      // Empty shards legitimately never create a native EM/radio session.
      // Require measured CUDA work only when there are segments to project.
      if(!cpu_workers&&m.at("native_radio_tracks").get<std::uint64_t>()) {
        require(m.at("radio_execution_space")=="Cuda","Requested GPU radio did not use CUDA");
        require(m.at("radio_cuda_kernel_ms").get<double>()>0.&&
          m.at("radio_projection_batches").get<std::uint64_t>()>0,"Missing measured CUDA radio execution");
      }
    }
    if(gpu_memory_fraction>0.)require(m.at("gpu_memory_plan_active").get<bool>()&&
      m.at("queue_capacity").get<std::size_t>()<=m.at("gpu_working_budget_bytes").get<std::size_t>()/
        m.at("gpu_bytes_per_history_bound").get<std::size_t>(),"Worker did not enforce GPU memory budget");
    count+=split.at("counts").at(i-1).get<std::uint64_t>();validateOutputs(parts[i]);stats.push_back(m);
  }
  require(count==initial.at("native_injections").get<std::uint64_t>(),"Lost/duplicated captured roots");
  auto products=merge(parts,root/"merged");validateOutputs(root/"merged");
  auto finish=std::chrono::steady_clock::now();
  double slowest_shower=0.,slowest_transport=0.;std::vector<int> pids;
  if(persistent)pids.push_back(initial.at("process_pid").get<int>());
  for(auto const& m:stats) {
    slowest_shower=std::max(slowest_shower,m.at("shower_wall_s").get<double>());
    slowest_transport=std::max(slowest_transport,m.value("shower_transport_s",m.at("shower_wall_s").get<double>()));
    if(persistent)pids.push_back(m.at("process_pid").get<int>());
  }
  if(persistent) {
    if(event_index==0)persistent_pids=pids;else require(pids==persistent_pids,"Persistent workers changed identity");
    require(initial.at("event_index")==event_index,"Prefix event index mismatch");
    for(auto const& m:stats)require(m.at("event_index")==event_index,"Worker event index mismatch");
  }
  double shower_advance=initial.at("shower_wall_s").get<double>()+slowest_shower;
  save(root/"COMPLETE.json",{{"status","completed"},{"scope","experimental native EGS4 single-shower static EM frontier"},
    {"topology",cpu_workers?"CPU process test (not multi-GPU)":"one CUDA process per physical GPU"},{"devices",gpu},
    {"worker_sha256",digest(worker)},{"table_sha256",digest(table)},{"partition",split},{"prefix",initial},{"workers",stats},
    {"antenna_file_sha256",reference.enabled?digest(reference.antennas):""},
    {"persistent_processes",persistent},{"event_index",event_index},{"shower_advance_s",shower_advance},
    {"shower_transport_s",initial.value("shower_transport_s",initial.at("shower_wall_s").get<double>())+slowest_transport},
    {"shower_advance_definition","prefix shower_wall_s + maximum worker shower_wall_s; excludes initialization, frontier partition and output merge"},
    {"wall_s",std::chrono::duration<double>(finish-start).count()},
    {"prefix_phase_s",std::chrono::duration<double>(prefix_finished-start).count()},
    {"workers_phase_s",std::chrono::duration<double>(workers_finished-prefix_finished).count()},
    {"verification_merge_s",std::chrono::duration<double>(finish-workers_finished).count()},
    {"products",products},{"radio_merge","sum signed electric-field components on identical time/antenna grids; never sum fluence"},
    {"trajectory_identity_with_unsplit_run",false},{"production_ready",false}});
  event_summaries.push_back({{"output",root.string()},{"seed",seeds[event_index]},
    {"shower_advance_s",shower_advance},{"wall_s",std::chrono::duration<double>(finish-start).count()}});
  std::cout<<"COMPLETE "<<root<<std::endl;
  }
  if(persistent) {
    jobs.wait(timeout);save(campaign_root/"COMPLETE.json",{{"status","completed"},{"persistent_processes",true},
      {"events",event_summaries},{"process_pids",persistent_pids},{"devices",gpu},
      {"wall_s",std::chrono::duration<double>(std::chrono::steady_clock::now()-campaign_start).count()}});
  }
  return 0;
}catch(std::exception const& e){std::cerr<<"Incomplete native EGS4 schedule: "<<e.what()<<'\n';return 1;}}
