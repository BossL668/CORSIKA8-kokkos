#include <corsika/accelerator/em/detail/IndependentEndpointDriver.hpp>
#include <iostream>
#include <atomic>
#include <chrono>
using corsika::accelerator::em::detail::IndependentEndpointDriver;
void require(bool value) { if(!value) throw std::runtime_error("independent driver test failed"); }
int main() {
  try {
    IndependentEndpointDriver driver;
    auto owner=std::this_thread::get_id();
    std::promise<void> release,entered;
    auto permit=release.get_future().share();
    auto began=entered.get_future();
    auto first=driver.submit([&] {
      entered.set_value(); permit.wait();
      return std::this_thread::get_id();
    });
    began.wait();
    bool rejected=false;
    try { driver.submit([]{}); } catch(std::logic_error const&) {rejected=true;}
    release.set_value();
    auto worker=first.get();driver.waitIdle();
    require(rejected && worker!=owner);
    auto failing=driver.submit([]()->int {throw std::runtime_error("endpoint failure");});
    bool propagated=false;
    try {failing.get();}catch(std::runtime_error const&){propagated=true;}
    driver.waitIdle();require(propagated);
    for(int event=0;event<32;++event) {
      auto f=driver.submit([] {return std::this_thread::get_id();});
      require(f.get()==worker);driver.waitIdle();
    }
    std::atomic<bool> drained{false};
    {
      IndependentEndpointDriver other;
      other.submit([&]{drained=true;});
    }
    require(drained);
    std::cout<<"PASS: bounded queue, distinct stable worker, exceptions, 32 reuse, destructor drain\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
