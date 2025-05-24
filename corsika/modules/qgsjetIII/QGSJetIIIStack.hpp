/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/CoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/geometry/PhysicalGeometry.hpp>
#include <corsika/framework/stack/Stack.hpp>
#include <corsika/modules/qgsjetIII/ParticleConversion.hpp>

namespace corsika::qgsjetIII {

  class QGSJetIIIStackData {

  public:
    void dump() const {}

    void clear();
    unsigned int getSize() const;
    unsigned int getCapacity() const;

    void setId(const unsigned int i, const int v);
    void setEnergy(const unsigned int i, const HEPEnergyType v);

    void setMomentum(const unsigned int i, const MomentumVector& v);

    int getId(const unsigned int i) const;
    HEPEnergyType getEnergy(const int i) const;
    MomentumVector getMomentum(const unsigned int i, const CoordinateSystemPtr& CS) const;

    void copy(const unsigned int i1, const unsigned int i2);
    void swap(const unsigned int i1, const unsigned int i2);

    void incrementSize();
    void decrementSize();
  };

  template <typename TStackIterator>
  class ParticleInterface : public corsika::ParticleBase<TStackIterator> {

    using corsika::ParticleBase<TStackIterator>::getStackData;
    using corsika::ParticleBase<TStackIterator>::getIndex;

  public:
    void setParticleData(const int vID, const HEPEnergyType vE, const MomentumVector& vP,
                         const HEPMassType);
    void setParticleData(ParticleInterface<TStackIterator>& /*parent*/, const int vID,
                         const HEPEnergyType vE, const MomentumVector& vP,
                         const HEPMassType);

    void setEnergy(const HEPEnergyType v);
    HEPEnergyType getEnergy() const;

    void setPID(const int v);
    corsika::qgsjetIII::QgsjetIIICode getPID() const;

    MomentumVector getMomentum(const CoordinateSystemPtr& CS) const;
    void setMomentum(const MomentumVector& v);
  };

  typedef corsika::Stack<QGSJetIIIStackData, ParticleInterface> QGSJetIIIStack;

} // end namespace corsika::qgsjetIII

#include <corsika/detail/modules/qgsjetIII/QGSJetIIIStack.inl>
