/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/Logging.hpp>

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace corsika::transport {

  using HistoryId = std::uint64_t;
  using StepId = std::uint64_t;
  using Generation = std::uint32_t;

  inline constexpr HistoryId NoParentHistoryId = 0;

  /**
   * Scheduler-facing identity of one particle history.
   *
   * A history ID is assigned once when a particle is created and follows the particle
   * when the stack compacts or swaps entries. step_id is the number of transport
   * advances already started for this history.
   */
  struct TransportIdentity {
    HistoryId history_id = NoParentHistoryId;
    HistoryId parent_history_id = NoParentHistoryId;
    Generation generation = 0;
    StepId step_id = 0;
  };

  static_assert(std::is_standard_layout_v<TransportIdentity>);
  static_assert(std::is_trivially_copyable_v<TransportIdentity>);

  /**
   * Stack storage for TransportIdentity records.
   */
  class TransportIdentityData {
  public:
    void clear();
    unsigned int getSize() const;
    unsigned int getCapacity() const;
    void copy(int i1, int i2);
    void swap(int i1, int i2);
    void incrementSize();
    void decrementSize();

    void assignPrimary(int index);
    void assignSecondary(int index, TransportIdentity const& parent);
    HistoryId reserveHistoryIds(std::uint64_t count);
    void importIdentity(int index, TransportIdentity const& identity);
    TransportIdentity const& getTransportIdentity(int index) const;
    void setStepId(int index, StepId step_id);

  private:
    HistoryId allocateHistoryId();

    std::vector<TransportIdentity> identities_;
    HistoryId next_history_id_ = 1;
  };

  /**
   * Particle interface mixed into a CombinedStack.
   */
  template <typename TParentStack>
  class TransportIdentityDataInterface : public TParentStack {
  protected:
    using TParentStack::getIndex;
    using TParentStack::getStackData;

  public:
    void setParticleData();
    void setParticleData(TransportIdentityDataInterface const& parent);

    TransportIdentity getTransportIdentity() const;
    void setTransportIdentity(TransportIdentity const& identity);
    HistoryId getHistoryId() const;
    HistoryId getParentHistoryId() const;
    Generation getGeneration() const;
    StepId getStepId() const;

    /**
     * Mark the beginning of one scalar or device transport advance.
     *
     * @return the step ID assigned to the advance that is about to run.
     */
    StepId beginTransportStep();

    std::string asString() const;
  };

  template <typename TParentStack>
  struct MakeTransportIdentityDataInterface {
    using type = TransportIdentityDataInterface<TParentStack>;
  };

  template <typename TParticle, typename = void>
  struct HasTransportIdentity : std::false_type {};

  template <typename TParticle>
  struct HasTransportIdentity<
      TParticle,
      std::void_t<decltype(std::declval<TParticle const&>().getHistoryId()),
                  decltype(std::declval<TParticle const&>().getGeneration()),
                  decltype(std::declval<TParticle const&>().getStepId()),
                  decltype(std::declval<TParticle&>().beginTransportStep())>>
      : std::true_type {};

  template <typename TParticle>
  inline constexpr bool has_transport_identity_v = HasTransportIdentity<TParticle>::value;

} // namespace corsika::transport

#include <corsika/detail/stack/TransportIdentityStackExtension.inl>
