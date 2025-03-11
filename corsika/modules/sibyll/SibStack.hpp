/*
 * (c) Copyright 2018 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/stack/Stack.hpp>
#include <corsika/modules/sibyll/ParticleConversion.hpp>

#include <sibyll2.3d-public.hpp>

namespace corsika::sibyll {

  typedef corsika::Vector<hepmomentum_d> MomentumVector;

  class SibStackData {
    static auto constexpr invGeV = 1 / 1_GeV;

  public:
    void dump() const {}

    void clear() { ::sibyll23d::s_plist_->np = 0; }
    unsigned int getSize() const { return ::sibyll23d::s_plist_->np; }
    unsigned int getCapacity() const { return 8000; }

    void setId(const unsigned int i, const int v) { ::sibyll23d::s_plist_->llist[i] = v; }
    void setEnergy(const unsigned int i, const HEPEnergyType v) {
      ::sibyll23d::s_plist_->p[3][i] = v * invGeV;
    }
    void setMass(const unsigned int i, const HEPMassType v) {
      ::sibyll23d::s_plist_->p[4][i] = v * invGeV;
    }
    void setMomentum(const unsigned int i, const MomentumVector& v) {
      auto tmp = v.getComponents();
      for (int idx = 0; idx < 3; ++idx)
        ::sibyll23d::s_plist_->p[idx][i] = tmp[idx] * invGeV;
    }

    int getId(const unsigned int i) const { return ::sibyll23d::s_plist_->llist[i]; }
    HEPEnergyType getEnergy(const int i) const {
      return ::sibyll23d::s_plist_->p[3][i] * 1_GeV;
    }
    HEPEnergyType getMass(const unsigned int i) const {
      return ::sibyll23d::s_plist_->p[4][i] * 1_GeV;
    }
    MomentumVector getMomentum(const unsigned int i) const {
      CoordinateSystemPtr const& rootCS = get_root_CoordinateSystem();
      QuantityVector<hepmomentum_d> components = {::sibyll23d::s_plist_->p[0][i] * 1_GeV,
                                                  ::sibyll23d::s_plist_->p[1][i] * 1_GeV,
                                                  ::sibyll23d::s_plist_->p[2][i] * 1_GeV};
      return MomentumVector(rootCS, components);
    }

    void copy(const unsigned int i1, const unsigned int i2) {
      ::sibyll23d::s_plist_->llist[i2] = ::sibyll23d::s_plist_->llist[i1];
      for (unsigned int i = 0; i < 5; ++i)
        ::sibyll23d::s_plist_->p[i][i2] = ::sibyll23d::s_plist_->p[i][i1];
    }

    void swap(const unsigned int i1, const unsigned int i2) {
      std::swap(::sibyll23d::s_plist_->llist[i1], ::sibyll23d::s_plist_->llist[i2]);
      for (unsigned int i = 0; i < 5; ++i)
        std::swap(::sibyll23d::s_plist_->p[i][i1], ::sibyll23d::s_plist_->p[i][i2]);
    }

    void incrementSize() { ::sibyll23d::s_plist_->np++; }
    void decrementSize() {
      if (::sibyll23d::s_plist_->np > 0) { ::sibyll23d::s_plist_->np--; }
    }
  };

  template <typename TStackIterator>
  class ParticleInterface : public corsika::ParticleBase<TStackIterator> {

    using corsika::ParticleBase<TStackIterator>::getStackData;
    using corsika::ParticleBase<TStackIterator>::getIndex;

  public:
    void setParticleData(const int vID, const HEPEnergyType vE, const MomentumVector& vP,
                         const HEPMassType vM) {
      setPID(vID);
      setEnergy(vE);
      setMomentum(vP);
      setMass(vM);
    }

    void setParticleData(ParticleInterface<TStackIterator>& /*parent*/, const int vID,
                         const HEPEnergyType vE, const MomentumVector& vP,
                         const HEPMassType vM) {
      setPID(vID);
      setEnergy(vE);
      setMomentum(vP);
      setMass(vM);
    }

    void setEnergy(const HEPEnergyType v) { getStackData().setEnergy(getIndex(), v); }

    HEPEnergyType getEnergy() const { return getStackData().getEnergy(getIndex()); }

    bool hasDecayed() const { return abs(getStackData().getId(getIndex())) > 100; }

    void setMass(const HEPMassType v) { getStackData().setMass(getIndex(), v); }

    HEPEnergyType getMass() const { return getStackData().getMass(getIndex()); }

    void setPID(const int v) { getStackData().setId(getIndex(), v); }

    corsika::sibyll::SibyllCode getPID() const {
      return static_cast<corsika::sibyll::SibyllCode>(getStackData().getId(getIndex()));
    }

    MomentumVector getMomentum() const { return getStackData().getMomentum(getIndex()); }

    void setMomentum(const MomentumVector& v) {
      getStackData().setMomentum(getIndex(), v);
    }
  };

  typedef corsika::Stack<SibStackData, ParticleInterface> SibStack;

} // end namespace corsika::sibyll
