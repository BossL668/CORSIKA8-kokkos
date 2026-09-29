#include "../../applications/detail/air_shower_kokkos/KokkosAirShowerContext.hpp"
#ifdef NDEBUG
#undef NDEBUG // Assertions are required even in Release test builds.
#endif
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>

using namespace corsika::applications::air_shower;

template <int I> struct Probe {
  int state = I;
  Probe() = default;
  Probe(Probe const&) = delete;
  Probe(Probe&&) = delete;
};

template <typename A, typename B>
void check(A const& actual, B const& expected) {
  static_assert(std::is_same_v<A, B>, "Forwarded type changed");
  if constexpr (std::is_arithmetic_v<A> || std::is_pointer_v<A> ||
                std::is_same_v<A, std::array<double, 3>>)
    assert(actual == expected);
  else
    assert(std::addressof(actual) == std::addressof(expected));
}

template <typename A, typename B, std::size_t... I>
void checkAll(A const& actual, B const& expected, std::index_sequence<I...>) {
  (check(std::get<I>(actual), std::get<I>(expected)), ...);
}

int main() {
  Probe<0> v0;
  Probe<1> const v1;
  Probe<2> const v2;
  Probe<3> const v3;
  Probe<4> const v4;
  Probe<5> const v5;
  Probe<6> const v6;
  Probe<10> v10;
  Probe<11> v11;
  Probe<12> v12;
  Probe<13> v13;
  Probe<14> v14;
  Probe<15> v15;
  Probe<16> v16;
  Probe<17> v17;
  Probe<18> v18;
  Probe<19> v19;
  Probe<20> v20;
  Probe<21> v21;
  Probe<22> v22;
  Probe<23> v23;
  Probe<24> v24;
  Probe<25> v25;
  Probe<26> v26;
  Probe<27> v27;
  Probe<28> v28;
  Probe<29> v29;
  Probe<30> v30;
  Probe<31> v31;
  Probe<32> v32;
  Probe<33> v33;
  Probe<34> pool_object;
  auto* pool = &pool_object;
  Probe<35> v35;
  Probe<36> v36;
  Probe<37> v37;
  Probe<38> const v38;
  Probe<39> const v39;
  Probe<40> v40;
  Probe<41> v41;
  Probe<46> const v46;
  Probe<47> const v47;
  KokkosAirShowerContext const context{
      KokkosAirShowerGeometry{v0, v1, v2, v3, v4, v5, v6, 107, 108, 109, std::array<double, 3>{.1, .2, .3}},
      KokkosAirShowerOutputs{v10, v11, v12, v13, v14, v15, v16},
      KokkosAirShowerProcesses{v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29},
      KokkosAirShowerExecution{v30, v31, v32, v33, pool, v35},
      KokkosAirShowerDiagnostics{v36, v37, v38, v39, v40, v41, 142, 143, 144, 145, v46, v47}};

  auto const& g = context.geometry;
  // Independently specified legacy order: all 49 forwarded arguments are checked.
  auto expected = std::tuple{
      std::ref(v0),
      std::ref(v1),
      std::ref(v2),
      std::ref(v3),
      std::ref(v4),
      std::ref(v15),
      std::ref(v16),
      std::ref(v5),
      std::ref(v6),
      double{107},
      double{108},
      double{109},
      std::array<double, 3>{.1, .2, .3},
      std::ref(v10),
      std::ref(v11),
      std::ref(v12),
      std::ref(v13),
      std::ref(v14),
      std::ref(v17),
      std::ref(v18),
      std::ref(v19),
      std::ref(v20),
      std::ref(v21),
      std::ref(v22),
      std::ref(v23),
      std::ref(v24),
      std::ref(v25),
      std::ref(v26),
      std::ref(v27),
      std::ref(v28),
      std::ref(v29),
      std::ref(v30),
      std::ref(v31),
      std::ref(v32),
      std::ref(v33),
      pool,
      std::ref(v35),
      std::ref(v36),
      std::ref(v37),
      std::ref(v38),
      std::ref(v39),
      std::ref(v40),
      std::ref(v41),
      std::uint64_t{142},
      std::uint64_t{143},
      std::size_t{144},
      std::size_t{145},
      std::ref(v46),
      std::ref(v47)
  };
  int calls = 0;
  auto inspect = [&](auto&&... arguments) {
    static_assert(sizeof...(arguments) == 49);
    auto forwarded = std::forward_as_tuple(arguments...);
    static_assert(std::is_same_v<decltype(std::get<0>(forwarded)), Probe<0>&>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(std::get<1>(forwarded))>>);
    auto unwrapped = std::apply([](auto const&... items) {
      // make_tuple unwraps reference_wrapper; models themselves are never copied.
      return std::make_tuple(items...);
    }, expected);
    checkAll(forwarded, unwrapped, std::make_index_sequence<49>{});
    ++calls;
    return 17;
  };
  assert(detail::withKokkosAirShowerArguments(context, inspect) == 17);
  auto copied = context; // reference groups remain non-owning
  g.environment.state = 91;
  assert(detail::withKokkosAirShowerArguments(copied, inspect) == 17);
  assert(calls == 2 && g.environment.state == 91);
  bool caught = false;
  try {
    detail::withKokkosAirShowerArguments(context, [](auto&&...) {
      throw std::runtime_error("same exception");
    });
  } catch (std::runtime_error const& error) {
    caught = std::string(error.what()) == "same exception";
  }
  assert(caught);
  copied.execution.hadronic_pool = nullptr;
  detail::withKokkosAirShowerArguments(copied, [](auto&&... arguments) {
    auto forwarded = std::forward_as_tuple(arguments...);
    assert(std::get<35>(forwarded) == nullptr);
  });
  std::cout << "PASS: all 49 bindings, constness, noncopyable objects, reuse, "
               "owned field temporary, null pool and exception propagation\n";
}
