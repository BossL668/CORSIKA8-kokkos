/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/stack/Stack.hpp>
#include <corsika/modules/qgsjetIII/ParticleConversion.hpp>

#include <qgsjet-III-04.hpp>

namespace corsika::qgsjetIII {

  class QGSJetIIIFragmentsStackData {

  public:
    void dump() const {}

    void clear() {
      qgarr13_.nsf = 0;
      qgarr55_.nwt = 0;
    }
    unsigned int getSize() const { return qgarr13_.nsf; }
    unsigned int getCapacity() const { return iapmax; }

    static unsigned int getWoundedNucleonsTarget() { return qgarr55_.nwt; }
    static unsigned int getWoundedNucleonsProjectile() { return qgarr55_.nwp; }

    int getFragmentSize(const unsigned int i) const { return qgarr13_.iaf[i]; }
    void setFragmentSize(const unsigned int i, const int v) { qgarr13_.iaf[i] = v; }

    void copy(const unsigned int i1, const unsigned int i2) {
      qgarr13_.iaf[i2] = qgarr13_.iaf[i1];
    }

    void swap(const unsigned int i1, const unsigned int i2) {
      std::swap(qgarr13_.iaf[i1], qgarr13_.iaf[i2]);
    }

    void incrementSize() { qgarr13_.nsf++; }
    void decrementSize() {
      if (qgarr13_.nsf > 0) { qgarr13_.nsf--; }
    }
  };

  template <typename TStackIterator>
  class FragmentsInterface : public corsika::ParticleBase<TStackIterator> {

    using corsika::ParticleBase<TStackIterator>::getStackData;
    using corsika::ParticleBase<TStackIterator>::getIndex;

  public:
    void setParticleData(const int vSize) { setFragmentSize(vSize); }

    void setParticleData(FragmentsInterface<TStackIterator>& /*parent*/,
                         const int vSize) {
      setFragmentSize(vSize);
    }

    void setFragmentSize(const int v) { getStackData().setFragmentSize(getIndex(), v); }

    double getFragmentSize() const { return getStackData().getFragmentSize(getIndex()); }
  };

  typedef corsika::Stack<QGSJetIIIFragmentsStackData, FragmentsInterface>
      QGSJetIIIFragmentsStack;

} // end namespace corsika::qgsjetIII

