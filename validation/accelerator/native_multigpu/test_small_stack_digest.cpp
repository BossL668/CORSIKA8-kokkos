#include "NativeCoordinator.hpp"
#include <pthread.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace c8::multigpu;
struct Query { fs::path file; std::string result, error; };
static void* hashOnSmallStack(void* data) {
  auto& q=*static_cast<Query*>(data);
  try { q.result=digest(q.file); }
  catch(std::exception const& e) { q.error=e.what(); }
  return nullptr;
}
int main() {
  char pattern[]="/tmp/c8-digest-stack-XXXXXX";
  auto directory=mkdtemp(pattern);
  if(!directory) return 2;
  fs::path root(directory);
  // A guarded thread stack deterministically catches a 64 KiB local I/O buffer.
  // SHA256's own algorithm is unchanged; only the file-reading buffer is tested.
  for(auto const& test : std::vector<std::pair<std::size_t,std::string>>{
      {0,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {3,"9834876dcfb05cb167a5c24953eba58c4ac89b1adf57f28f2f9d09af107ee8f0"},
      {65535,"6e1bebca6a8229364a162a72ef064826c4cd7457bf54f190ef782bd9deff3e42"},
      {65536,"bf718b6f653bebc184e1479f1935b8da974d701b893afcf49e701f3e2f9f9c5a"},
      {65537,"008ffc88d3c96a9f307524eb361e47c5222a887fc45fa0c1fb8d429c5c23b430"},
      {1048576,"9bc1b2a288b26af7257a36277ae3816a7d4f16e89c1e7e77d0a5c48bad62b360"}}) {
    Query q{root/std::to_string(test.first),{}, {}};
    { std::ofstream out(q.file,std::ios::binary); out<<std::string(test.first,'a'); }
    pthread_attr_t attr;
    if(pthread_attr_init(&attr) || pthread_attr_setstacksize(&attr,65536) ||
       pthread_attr_setguardsize(&attr,4096)) return 3;
    pthread_t thread;
    if(pthread_create(&thread,&attr,hashOnSmallStack,&q)) return 4;
    pthread_attr_destroy(&attr);
    if(pthread_join(thread,nullptr) || !q.error.empty() || q.result!=test.second) {
      std::cerr<<q.file<<": "<<q.error<<" "<<q.result<<std::endl;return 1;
    }
  }
  std::cout<<"64 KiB guarded-stack SHA256 checks passed: "<<root<<std::endl;
}
