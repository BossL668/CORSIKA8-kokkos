#pragma once
#include <corsika/detail/modules/egs4/Egs4C8AirOutput.hpp>
#include <corsika/modules/egs4/Session.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileAccumulator.hpp>
#include <corsika/accelerator/em/kokkos/KokkosProfileProjection.hpp>

namespace c7_egs4::application {
// Reuse production geometry, crossing rules, fixed-point ledger and continuous
// / endpoint deposit routines. Do not synthesize PROPOSAL interactions or cuts:
// EGS4 already froze the exact total and endpoint energy in AirOutput.
template<class Outputs>struct AccumulateEgs4Profile {
  Outputs outputs;
  ::corsika::gpu::em::detail::DeviceProfileProjection projection;
  ::corsika::gpu::em::detail::DeviceProfileAccumulator accumulator;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t i)const {
    if(!outputs(i).has_step)return;
    auto const& r=outputs(i).step;
    namespace d=::corsika::accelerator::em::detail;
    using Atomic=::corsika::accelerator::em::kokkos_detail::KokkosProfileAtomicOperations;
    double start=d::projectProfileGrammage(projection,r.start_position_m);
    double end=d::projectProfileGrammage(projection,r.end_position_m);
    if(!d::profileFinite(start)||!d::profileFinite(end)||!d::profileFinite(r.weight)||r.weight<0.||
       !d::profileFinite(r.deposited_energy_GeV)||!d::profileFinite(r.cut_deposited_energy_GeV)||
       r.cut_deposited_energy_GeV<0.||r.deposited_energy_GeV<r.cut_deposited_energy_GeV) {
      Atomic::add(&accumulator.counters->invalid_records,1ULL);return;
    }
    d::accumulateParticleProfile<Atomic>(accumulator,r.pid,start,end,r.weight);
    d::accumulateEnergyProfile<Atomic>(accumulator,accumulator.energy_loss,start,end,
      (r.deposited_energy_GeV-r.cut_deposited_energy_GeV)*r.weight);
    d::accumulatePointEnergyProfile<Atomic>(accumulator,accumulator.energy_loss,end,
      r.cut_deposited_energy_GeV*r.weight);
    Atomic::add(&accumulator.counters->steps,1ULL);
  }
};
template<class Exec>class DeviceProfile {
  ::corsika::accelerator::em::kokkos_detail::KokkosProfileAccumulator<Exec> accumulator_;
  ::corsika::accelerator::em::kokkos_detail::KokkosProfileProjection<Exec> projection_;
  OutputCallbacks const& output_;
  double weight_limit_{},energy_limit_{};
public:
  DeviceProfile(::corsika::gpu::em::GpuEmConfig::ProfileProjection const& config,OutputCallbacks const& output):output_(output) {
    if(config.enabled&&(!config.accumulate_on_device||!output.profile))
      throw std::invalid_argument("Native device profile requires accumulation and a final consumer");
    projection_.initialize(config);accumulator_.initialize(config);
    weight_limit_=config.fixed_point_weight_limit;energy_limit_=config.fixed_point_energy_limit_GeV;
  }
  bool enabled()const{return accumulator_.enabled();}
  void reset(){accumulator_.reset(weight_limit_,energy_limit_);}
  template<class Outputs>void accumulate(Outputs const& records) {
    if(!enabled())return;
    accumulator_.requireOpenForAccumulation();
    Kokkos::parallel_for("egs4_accumulate_resident_profile",Kokkos::RangePolicy<Exec>(0,records.extent(0)),
      AccumulateEgs4Profile<Outputs>{records,projection_.deviceView(),accumulator_.deviceView()});
  }
  void finish(RunStatistics& stats) {
    if(!enabled())return;
    auto result=accumulator_.download();
    stats.profile_execution_space=Exec::name();stats.profile_steps=result.steps;
    stats.profile_fixed_point_overflows=result.fixed_point_overflows;stats.profile_invalid_records=result.invalid_records;
    if(result.steps!=stats.steps||result.fixed_point_overflows||result.invalid_records)
      throw std::runtime_error("Native device profile missing/invalid/overflowing records");
    output_.profile(result);
  }
};
} // namespace c7_egs4::application
