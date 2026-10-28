/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/kokkos/utility/StableQuartic.hpp>

#include "PolynomialRootsReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace polynomial_roots_test {

  enum class Operation { Linear, Quadratic, MonicQuadratic, Resolvent, Quartic };
  enum class Branch { None, SmallRootHalley, MonotoneHalley, Cardano, Trigonometric };

  struct Query {
    Operation operation_{};
    double a_{};
    double b_{};
    double c_{};
    double epsilon_{1.e-12};
    Branch branch_{Branch::None};
    int scaleExponent_{};
    int knownCount_{-1};
    double knownRoots_[4]{};
    bool failure_{};
    bool sphere_{};
    bool relativeRoots_{};
  };

  struct Result {
    double roots_[4]{};
    unsigned int count_{};
  };

  template <typename TRoots>
  KOKKOS_INLINE_FUNCTION Result migrated_result(TRoots const& roots) {
    Result result;
    result.count_ = roots.getSize();
    for (unsigned int index = 0; index < roots.getSize(); ++index) {
      result.roots_[index] = roots[index];
    }
    return result;
  }

  // Copy the small result by value. The validation NVCC toolchain loses its
  // array elements through a const-reference adapter for quadratic temporaries.
  // Direct device stores and this value copy agree; the frozen solver is unchanged.
  KOKKOS_INLINE_FUNCTION Result
  reference_result(polynomial_roots_reference::Roots roots) {
    Result result;
    result.count_ = roots.size;
    for (unsigned int index = 0; index < roots.size; ++index) {
      result.roots_[index] = roots.values[index];
    }
    return result;
  }

  KOKKOS_INLINE_FUNCTION Result evaluate(Query const& query) {
    namespace implementation = corsika::kokkos;
    using real_type = implementation::polynomial_detail::real_type;
    switch (query.operation_) {
      case Operation::Linear:
        return migrated_result(implementation::linear_roots(query.a_, query.b_));
      case Operation::Quadratic:
        return migrated_result(
            implementation::quadratic_roots(real_type(query.a_), real_type(query.b_),
                                            real_type(query.c_), query.epsilon_));
      case Operation::MonicQuadratic:
        return migrated_result(implementation::polynomial_detail::stable_monic_quadratic(
            real_type(query.a_), real_type(query.b_)));
      case Operation::Resolvent: {
        Result result;
        result.count_ = 1;
        result.roots_[0] =
            static_cast<double>(implementation::polynomial_detail::stable_resolvent(
                real_type(query.a_), real_type(query.b_), real_type(query.c_)));
        return result;
      }
      case Operation::Quartic:
        return migrated_result(
            implementation::stable_quartic_roots(query.a_, query.b_, query.c_));
    }
    return {};
  }

  KOKKOS_INLINE_FUNCTION Result reference(Query const& query) {
    namespace frozen = polynomial_roots_reference;
    using real_type = frozen::Real;
    switch (query.operation_) {
      case Operation::Linear:
        return reference_result(frozen::linear(query.a_, query.b_));
      case Operation::Quadratic:
        return reference_result(frozen::quadratic(real_type(query.a_),
                                                  real_type(query.b_),
                                                  real_type(query.c_), query.epsilon_));
      case Operation::MonicQuadratic:
        return reference_result(
            frozen::stableMonicQuadratic(real_type(query.a_), real_type(query.b_)));
      case Operation::Resolvent: {
        Result result;
        result.count_ = 1;
        result.roots_[0] = static_cast<double>(frozen::stableResolvent(
            real_type(query.a_), real_type(query.b_), real_type(query.c_)));
        return result;
      }
      case Operation::Quartic:
        return reference_result(frozen::stableQuartic(query.a_, query.b_, query.c_));
    }
    return {};
  }

  KOKKOS_INLINE_FUNCTION int quartic_scale(double p, double q, double r) {
    int ep = -4096, eq = -4096, er = -4096;
    if (p != 0) {
      int exponent;
      ::frexp(p, &exponent);
      ep = exponent >= 0 ? (exponent + 1) / 2 : exponent / 2;
    }
    if (q != 0) {
      int exponent;
      ::frexp(q, &exponent);
      eq = exponent >= 0 ? (exponent + 2) / 3 : exponent / 3;
    }
    if (r != 0) {
      int exponent;
      ::frexp(r, &exponent);
      er = exponent >= 0 ? (exponent + 3) / 4 : exponent / 4;
    }
    int exponent = ep > eq ? ep : eq;
    exponent = exponent > er ? exponent : er;
    return exponent == -4096 ? 0 : exponent;
  }

  // Independent production predicates, evaluated on the scaled coefficients
  // actually entering the quartic resolvent. q==0 bypasses that solver. These
  // labels do not instrument or modify the migrated implementation.
  KOKKOS_INLINE_FUNCTION Branch reference_branch(Query const& query) {
    if (query.operation_ != Operation::Resolvent &&
        query.operation_ != Operation::Quartic) {
      return Branch::None;
    }
    if (query.operation_ == Operation::Quartic && query.b_ == 0.) { return Branch::None; }
    using polynomial_roots_reference::Real;
    int const exponent = query.operation_ == Operation::Quartic
                             ? quartic_scale(query.a_, query.b_, query.c_)
                             : 0;
    Real const p(::ldexp(query.a_, -2 * exponent));
    Real const q(::ldexp(query.b_, -3 * exponent));
    Real const r(::ldexp(query.c_, -4 * exponent));
    Real const B = p * p / 4 - r, D = q * q / 8;
    if (p < 0 && r < 0 && D < (-p) * B / 64 && D < p * r / 8) {
      return Branch::SmallRootHalley;
    }
    if (!(p < 0) && !(B < 0)) { return Branch::MonotoneHalley; }
    Real const Q = (3 * B - p * p) / 9;
    Real const R = (p * (9 * B - 2 * p * p) + 27 * D) / 54;
    auto const discriminant = polynomial_roots_reference::fma(R, R, Q * Q * Q);
    return !(discriminant < 0) ? Branch::Cardano : Branch::Trigonometric;
  }

  // Coefficients only: reproduce the fixed production curved-sphere expressions
  // for a finite, nonparallel charged track. The root test does not instantiate
  // CurvedAccess, classify crossings, select a node or advance a particle.
  inline Query sphere_query(double gap, double directionX, double chargeSign) {
    namespace frozen = polynomial_roots_reference;
    double const radius = 6371000.;
    double const delta[3]{0., 0., radius + gap};
    double const direction[3]{directionX, 0., directionX == 0. ? -1. : -.8};
    double const field[3]{0., 5.e-5, 0.};
    double const charge = chargeSign * 1.602176634e-19;
    double const momentum = 1.602176634e-19;
    double const projected[3]{
        frozen::difference(frozen::product(direction[1], field[2]),
                           frozen::product(direction[2], field[1])),
        frozen::difference(frozen::product(direction[2], field[0]),
                           frozen::product(direction[0], field[2])),
        frozen::difference(frozen::product(direction[0], field[1]),
                           frozen::product(direction[1], field[0]))};
    auto const projected2 = frozen::dot(projected, projected);
    double const deltaNorm = std::sqrt(frozen::dot(delta, delta));
    double const k = charge / momentum, denominator = 4. / (projected2 * k * k);
    double const p = frozen::product(
        frozen::sum(frozen::product(frozen::dot(projected, delta), k), 1), denominator);
    double const q = frozen::dot(direction, delta) * 2 * denominator;
    double const r = (deltaNorm + radius) * (deltaNorm - radius) * denominator;
    Query result{Operation::Quartic, p, q, r};
    result.scaleExponent_ = quartic_scale(p, q, r);
    result.sphere_ = true;
    return result;
  }

  inline std::vector<Query> queries() {
    double const tiny = std::numeric_limits<double>::denorm_min();
    double const nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<Query> result{
        {Operation::Linear, 0., 0.},
        {Operation::Linear, 0., 1.},
        {Operation::Linear, 2., -6.},
        {Operation::Linear, 1., 0.},
        {Operation::Linear, -1., 0.},
        {Operation::Linear, tiny, 1.},
        {Operation::Linear, 1., nan},
        {Operation::Quadratic, 0., 0., 1.},
        {Operation::Quadratic, 0., 2., -6.},
        {Operation::Quadratic, .5e-12, 2., -6.},
        {Operation::Quadratic, 1.e-12, 2., -6.},
        {Operation::Quadratic, 1., 0., 0.},
        {Operation::Quadratic, 1., 2., .5e-12},
        {Operation::Quadratic, 1., 2., 1.e-12},
        {Operation::Quadratic, 1., 2., 1.},
        {Operation::Quadratic, 1., 0., 1.},
        {Operation::Quadratic, 1., 0., -4.},
        {Operation::Quadratic, 1., -3., 2.},
        {Operation::Quadratic, 1., 3., 2.},
        {Operation::Quadratic, 1.e-20, 1., 1., 0.},
        {Operation::MonicQuadratic, 0., 0.},
        {Operation::MonicQuadratic, 2., 0.},
        {Operation::MonicQuadratic, -2., 0.},
        {Operation::MonicQuadratic, 2., 1.},
        {Operation::MonicQuadratic, 0., 1.},
        {Operation::MonicQuadratic, -3., 2.},
        {Operation::Resolvent, -4., .25, -1., 1.e-12, Branch::SmallRootHalley},
        {Operation::Resolvent, -4., 1., -1., 1.e-12, Branch::SmallRootHalley},
        {Operation::Resolvent, 1., 1., -1., 1.e-12, Branch::MonotoneHalley},
        {Operation::Resolvent, 0., 1., -1., 1.e-12, Branch::MonotoneHalley},
        {Operation::Resolvent, -1., 4., -1., 1.e-12, Branch::Cardano},
        {Operation::Resolvent, -1., -4., -1., 1.e-12, Branch::Cardano},
        {Operation::Resolvent, -15., -10., 24., 1.e-12, Branch::Trigonometric},
        {Operation::Resolvent, -15., 10., 24., 1.e-12, Branch::Trigonometric},
        {Operation::Quartic, -4., .25, -1., 1.e-12, Branch::SmallRootHalley},
        {Operation::Quartic, 1., 1., -1., 1.e-12, Branch::MonotoneHalley},
        {Operation::Quartic, -1., 4., -1., 1.e-12, Branch::Cardano},
        {Operation::Quartic, -15., 10., 24., 1.e-12, Branch::Trigonometric}};

    // Both roots of x*x + 1e16*x + 1 are well separated and well conditioned.
    // Their corrections to -1e16 and -1e-16 are below binary64 resolution.
    // Check each root relatively: a coefficient-scale absolute tolerance must
    // not accept the small nonzero root being rounded or changed to zero.
    Query separatedQuadratic{Operation::Quadratic, 1., 1.e16, 1.};
    separatedQuadratic.knownCount_ = 2;
    separatedQuadratic.knownRoots_[0] = -1.e-16;
    separatedQuadratic.knownRoots_[1] = -1.e16;
    separatedQuadratic.relativeRoots_ = true;
    result.push_back(separatedQuadratic);
    Query separatedMonic{Operation::MonicQuadratic, 1.e16, 1.};
    separatedMonic.knownCount_ = 2;
    separatedMonic.knownRoots_[0] = -1.e16;
    separatedMonic.knownRoots_[1] = -1.e-16;
    separatedMonic.relativeRoots_ = true;
    result.push_back(separatedMonic);

    // Factored polynomials give a second oracle independent of the copied code.
    // Repeated roots are deliberately algebraically exact, not rounded products
    // of nearly coincident roots whose multiplicity is undefined after rounding.
    std::array<Query, 9> const factored{
        {{Operation::Quartic, 0., 0., 0., 1.e-12, Branch::None, 0, 1, {0.}},
         {Operation::Quartic, 0., 0., -1., 1.e-12, Branch::None, 0, 2, {-1., 1.}},
         {Operation::Quartic, 0., 0., 1., 1.e-12, Branch::None, 0, 0, {}},
         {Operation::Quartic, -2., 0., 1., 1.e-12, Branch::None, 0, 2, {-1., 1.}},
         {Operation::Quartic, -1., 0., 0., 1.e-12, Branch::None, 0, 3, {-1., 0., 1.}},
         {Operation::Quartic,
          -5.,
          0.,
          4.,
          1.e-12,
          Branch::None,
          0,
          4,
          {-2., -1., 1., 2.}},
         {Operation::Quartic,
          -15.,
          -10.,
          24.,
          1.e-12,
          Branch::Trigonometric,
          0,
          4,
          {-3., -2., 1., 4.}},
         {Operation::Quartic, -3., 2., 0., 1.e-12, Branch::Cardano, 0, 3, {-2., 0., 1.}},
         {Operation::Quartic,
          0.,
          1.,
          0.,
          1.e-12,
          Branch::MonotoneHalley,
          0,
          2,
          {-1., 0.}}}};
    for (auto const& polynomial : factored) {
      for (int exponent : {-240, -120, -20, 0, 20, 120, 240}) {
        Query scaled = polynomial;
        scaled.a_ = std::ldexp(polynomial.a_, 2 * exponent);
        scaled.b_ = std::ldexp(polynomial.b_, 3 * exponent);
        scaled.c_ = std::ldexp(polynomial.c_, 4 * exponent);
        scaled.scaleExponent_ = exponent;
        for (int index = 0; index < scaled.knownCount_; ++index) {
          scaled.knownRoots_[index] = std::ldexp(polynomial.knownRoots_[index], exponent);
        }
        result.push_back(scaled);
      }
    }

    // Both backends lose the nonzero q while normalizing in binary64, producing
    // the production explicit +infinity failure sentinel, not an empty result.
    Query failure{Operation::Quartic, 1., tiny, -1.};
    failure.branch_ = Branch::MonotoneHalley;
    failure.failure_ = true;
    result.push_back(failure);
    for (double gap : {-1000., -100., 100., 1000.}) {
      for (double directionX : {0., .6}) {
        for (double chargeSign : {-1., 1.}) {
          result.push_back(sphere_query(gap, directionX, chargeSign));
        }
      }
    }
    return result;
  }

  struct DistinctQuery {
    double candidates_[6]{};
    unsigned int count_{};
    Result expected_;
  };

  KOKKOS_INLINE_FUNCTION Result evaluate(DistinctQuery const& query) {
    corsika::kokkos::PolynomialRoots roots;
    for (unsigned int index = 0; index < query.count_; ++index) {
      corsika::kokkos::polynomial_detail::append_distinct(roots,
                                                          query.candidates_[index]);
    }
    return migrated_result(roots);
  }

  KOKKOS_INLINE_FUNCTION Result reference(DistinctQuery const& query) {
    polynomial_roots_reference::Roots roots;
    for (unsigned int index = 0; index < query.count_; ++index) {
      polynomial_roots_reference::appendDistinct(roots, query.candidates_[index]);
    }
    return reference_result(roots);
  }

  inline std::vector<DistinctQuery> distinct_queries() {
    std::vector<DistinctQuery> result;
    double const epsilon = std::numeric_limits<double>::epsilon();
    for (int exponent : {-500, 0, 500}) {
      for (double sign : {-1., 1.}) {
        double const first = sign * std::ldexp(1., exponent);
        for (int separation : {7, 8, 9}) {
          double const second = first * (1 + separation * epsilon);
          // The production inequality uses max(abs(first),abs(second)); these
          // exactly representable candidates straddle its inclusive 8-epsilon
          // boundary. No solver residual or root approximation enters this test.
          Result expected;
          expected.roots_[0] = first;
          expected.count_ = separation <= 8 ? 1 : 2;
          if (expected.count_ == 2) { expected.roots_[1] = second; }
          result.push_back({{first, second}, 2, expected});
        }
      }
    }
    double const tiny = std::numeric_limits<double>::denorm_min();
    result.push_back({{0., -0., tiny, -tiny}, 4, {{0., tiny, -tiny}, 3}});
    result.push_back({{-2., -1., 0., 1., 2., -2.}, 6, {{-2., -1., 0., 1.}, 4}});
    return result;
  }

  struct ExtendedQuery {
    double firstHigh_{}, firstLow_{}, secondHigh_{}, secondLow_{};
  };

  struct ExtendedResult {
    double high_[10]{};
    double low_[10]{};
    double rounded_[10]{};
    bool less_{}, greater_{}, equal_{}, finite_{};
  };

  template <typename TExtended>
  KOKKOS_INLINE_FUNCTION void save_migrated(ExtendedResult& result, int index,
                                            TExtended const& value) {
    result.high_[index] = value.getHigh();
    result.low_[index] = value.getLow();
    result.rounded_[index] = static_cast<double>(value);
  }

  KOKKOS_INLINE_FUNCTION void save_reference(ExtendedResult& result, int index,
                                             polynomial_roots_reference::Extended value) {
    result.high_[index] = value.hi;
    result.low_[index] = value.lo;
    result.rounded_[index] = static_cast<double>(value);
  }

  KOKKOS_INLINE_FUNCTION ExtendedResult evaluate(ExtendedQuery const& query) {
    using corsika::kokkos::ExtendedPrecision;
    ExtendedPrecision const first(query.firstHigh_, query.firstLow_);
    ExtendedPrecision const second(query.secondHigh_, query.secondLow_);
    ExtendedResult result;
    save_migrated(result, 0, first + second);
    save_migrated(result, 1, first - second);
    save_migrated(result, 2, first * second);
    save_migrated(result, 3, first / second);
    save_migrated(result, 4, -first);
    save_migrated(result, 5, corsika::kokkos::extended_sqrt(first));
    save_migrated(result, 6, corsika::kokkos::extended_cbrt(first));
    auto value = first;
    value += second;
    save_migrated(result, 7, value);
    value = first;
    value -= second;
    save_migrated(result, 8, value);
    value = first;
    value /= second;
    save_migrated(result, 9, value);
    result.less_ = first < second;
    result.greater_ = first > second;
    result.equal_ = first == second;
    result.finite_ = corsika::kokkos::extended_finite(query.firstHigh_);
    return result;
  }

  KOKKOS_INLINE_FUNCTION ExtendedResult reference(ExtendedQuery const& query) {
    using polynomial_roots_reference::Extended;
    Extended const first(query.firstHigh_, query.firstLow_);
    Extended const second(query.secondHigh_, query.secondLow_);
    ExtendedResult result;
    save_reference(result, 0, first + second);
    save_reference(result, 1, first - second);
    save_reference(result, 2, first * second);
    save_reference(result, 3, first / second);
    save_reference(result, 4, -first);
    save_reference(result, 5, polynomial_roots_reference::extendedSqrt(first));
    save_reference(result, 6, polynomial_roots_reference::extendedCbrt(first));
    auto value = first;
    value += second;
    save_reference(result, 7, value);
    value = first;
    value -= second;
    save_reference(result, 8, value);
    value = first;
    value /= second;
    save_reference(result, 9, value);
    result.less_ = first < second;
    result.greater_ = first > second;
    result.equal_ = first == second;
    result.finite_ = polynomial_roots_reference::extendedFinite(query.firstHigh_);
    return result;
  }

  inline std::vector<ExtendedQuery> extended_queries() {
    double const infinity = std::numeric_limits<double>::infinity();
    double const nan = std::numeric_limits<double>::quiet_NaN();
    return {{0., 0., 1., 0.},
            {-0., 0., 1., 0.},
            {1., 0., 0., 0.},
            {0., 0., 0., 0.},
            {1., std::ldexp(1., -54), 1., -std::ldexp(1., -54)},
            {std::ldexp(1., 500), 0., std::ldexp(1., -500), 0.},
            {std::ldexp(1., -500), 0., std::ldexp(1., 500), 0.},
            {-8., std::ldexp(1., -52), 3., 0.},
            {2., std::ldexp(1., -54), -2., std::ldexp(1., -54)},
            {infinity, 0., 1., 0.},
            {nan, 0., 1., 0.}};
  }

  inline long double scaled_residual(Query const& query, double root) {
    int const exponent = query.scaleExponent_;
    long double const x = std::ldexp(static_cast<long double>(root), -exponent);
    long double const p = std::ldexp(static_cast<long double>(query.a_), -2 * exponent);
    long double const q = std::ldexp(static_cast<long double>(query.b_), -3 * exponent);
    long double const r = std::ldexp(static_cast<long double>(query.c_), -4 * exponent);
    long double const fourth = x * x * x * x;
    long double const second = p * x * x;
    long double const first = q * x;
    long double const numerator = std::abs((fourth + second) + first + r);
    long double const denominator =
        std::abs(fourth) + std::abs(second) + std::abs(first) + std::abs(r);
    return denominator == 0. ? numerator : numerator / denominator;
  }

} // namespace polynomial_roots_test
