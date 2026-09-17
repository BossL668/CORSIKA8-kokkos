/* Stable mixed-species resident FIFO for material-interface transport.
 * Reuses the air implementation's generic pending queue and exclusive scan,
 * without modifying or depending on the air resident physics drivers. */
#pragma once
#include <Kokkos_Core.hpp>
#include <corsika/modules/transport/InterfaceEmTypes.hpp>
#include <corsika/modules/transport/detail/InterfaceRecordValidation.hpp>
#include <corsika/accelerator/em/kokkos/KokkosWavefrontQueue.hpp>

namespace corsika::interfaces::kokkos {
namespace kd = accelerator::em::kokkos_detail;

KOKKOS_INLINE_FUNCTION std::uint32_t successorCount(EmStep const& r) {
  if (r.outcome == EmOutcome::Continuation) return r.end.weight > 0. ? 1 : 0;
  std::uint32_t count = 0;
  if (r.outcome == EmOutcome::Children)
    for (std::uint32_t i = 0; i < r.child_count; ++i)
      if (r.children[i].weight > 0.) ++count;
  return count;
}

template<class Space> class InterfaceQueue {
 public:
  using Memory = typename Space::memory_space;
  using Particles = Kokkos::View<em::EmParticleState*, Memory>;
  using Records = Kokkos::View<EmStep*, Memory>;
  using Counts = Kokkos::View<std::uint32_t*, Memory>;
  using Offsets = Kokkos::View<std::uint64_t*, Memory>;
  struct FrontControl {
    std::uint64_t successors{};
    std::uint32_t error{}, fallbacks{};
  };
  using Control = Kokkos::View<FrontControl, Memory>;

  InterfaceQueue(std::size_t capacity, std::size_t batch, Space const& execution)
      : maximum_(capacity), batch_(batch), pending_("interface_resident_particles") {
    if (!capacity) return;
    pending_.ensureTailCapacity(capacity, capacity, execution);
    successors_ = Particles("interface_resident_successors", 3 * batch);
    counts_ = Counts("interface_successor_counts", batch);
    offsets_ = Offsets("interface_successor_offsets", batch);
    control_ = Control("interface_front_control");
    host_control_ = Kokkos::create_mirror_view(control_);
  }
  std::size_t size() const { return pending_.size(); }
  std::size_t capacity() const { return maximum_; }
  static std::size_t workspaceBytes(std::size_t capacity, std::size_t batch) {
    if (!capacity) return 0;
    return capacity * sizeof(em::EmParticleState) + 3 * batch * sizeof(em::EmParticleState)
        + batch * (sizeof(std::uint32_t) + sizeof(std::uint64_t)) + sizeof(FrontControl);
  }
  static std::size_t peakWorkspaceBytes(std::size_t capacity, std::size_t batch) {
    // PendingQueue compaction briefly retains old and replacement allocations.
    return workspaceBytes(capacity, batch) + capacity * sizeof(em::EmParticleState);
  }
  void append(Particles const& input, std::size_t count, Space const& execution) {
    if (!maximum_) throw std::logic_error("resident queue is disabled");
    if (!pending_.tryAppend(input, count, maximum_, execution))
      throw std::length_error("interface resident particle capacity exceeded");
  }
  void copyPrefix(Particles const& destination, std::size_t count, Space const& execution) const {
    if (count > size() || count > destination.extent(0))
      throw std::length_error("invalid interface resident prefix");
    Kokkos::deep_copy(execution,
        Kokkos::subview(destination, std::make_pair(std::size_t{0}, count)),
        Kokkos::subview(pending_.view(), std::make_pair(pending_.head(), pending_.head() + count)));
  }
  struct Count {
    Records records; Counts counts;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
      counts(i) = successorCount(records(i));
    }
  };
  struct Scatter {
    Records records; Offsets offsets; Particles successors;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
      auto const& r = records(i);
      auto offset = offsets(i);
      if (r.outcome == EmOutcome::Continuation && r.end.weight > 0.)
        successors(offset) = r.end;
      else if (r.outcome == EmOutcome::Children)
        for (std::uint32_t c = 0; c < r.child_count; ++c)
          if (r.children[c].weight > 0.) successors(offset++) = r.children[c];
    }
  };
  // Call only after validating all records. Overflow is checked BEFORE consume,
  // so a rejected wavefront retains its original input and can be diagnosed.
  std::size_t commit(Records const& records, std::size_t consumed, Space const& execution) {
    if (consumed > size() || consumed > batch_ || consumed > records.extent(0))
      throw std::length_error("invalid interface resident wavefront");
    Kokkos::parallel_for("interface_resident_count", Kokkos::RangePolicy<Space>(execution, 0, consumed),
                        Count{records, counts_});
    std::uint64_t total = 0;
    Kokkos::parallel_scan("interface_resident_scan", Kokkos::RangePolicy<Space>(execution, 0, consumed),
                         kd::CountScanFunctor<Space>{counts_, offsets_}, total);
    if (total > maximum_ - (size() - consumed))
      throw std::length_error("interface resident successors exceed capacity; input retained");
    Kokkos::parallel_for("interface_resident_scatter", Kokkos::RangePolicy<Space>(execution, 0, consumed),
                        Scatter{records, offsets_, successors_});
    pending_.consume(consumed);
    append(successors_, total, execution);
    return total;
  }
  struct ValidateCount {
    Records records; Counts counts; Control control; MaterialInterface interface;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i) const {
      auto const& r=records(i);
      auto error=detail::recordError(r,interface);
      counts(i)=error?0:successorCount(r);
      if(error) Kokkos::atomic_max(&control().error,error);
      if(r.outcome==EmOutcome::Fallback) Kokkos::atomic_fetch_add(&control().fallbacks,std::uint32_t{1});
    }
  };
  struct ControlScan {
    using value_type=std::uint64_t;
    Counts counts; Offsets offsets; Control control; std::size_t count;
    KOKKOS_INLINE_FUNCTION void operator()(std::size_t i, value_type& total, bool final) const {
      if(final) offsets(i)=total;
      total+=counts(i);
      if(final && i+1==count) control().successors=total;
    }
  };
  // One control download/fence. Validation and successor totals stay on device
  // until this checkpoint; no full records or live particles are downloaded.
  FrontControl commitControlled(Records const& records, std::size_t consumed,
                                MaterialInterface interface, Space const& execution) {
    if(!consumed || consumed>size() || consumed>batch_ || consumed>records.extent(0))
      throw std::length_error("invalid controlled interface wavefront");
    Kokkos::deep_copy(execution,control_,FrontControl{});
    Kokkos::parallel_for("interface_resident_validate_count",
        Kokkos::RangePolicy<Space>(execution,0,consumed),
        ValidateCount{records,counts_,control_,interface});
    Kokkos::parallel_scan("interface_resident_control_scan",
        Kokkos::RangePolicy<Space>(execution,0,consumed),
        ControlScan{counts_,offsets_,control_,consumed});
    Kokkos::deep_copy(execution,host_control_,control_);
    execution.fence("interface resident front control");
    auto result=host_control_();
    if(result.error) throw std::runtime_error("interface device record error "+std::to_string(result.error));
    if(result.successors>maximum_-(size()-consumed))
      throw std::length_error("interface resident successors exceed capacity; input retained");
    Kokkos::parallel_for("interface_resident_scatter",Kokkos::RangePolicy<Space>(execution,0,consumed),
        Scatter{records,offsets_,successors_});
    pending_.consume(consumed);
    append(successors_,result.successors,execution);
    return result;
  }
 private:
  std::size_t maximum_, batch_;
  kd::KokkosPendingParticleQueue<Space> pending_;
  Particles successors_;
  Counts counts_;
  Offsets offsets_;
  Control control_;
  typename Control::HostMirror host_control_;
};
} // namespace corsika::interfaces::kokkos
