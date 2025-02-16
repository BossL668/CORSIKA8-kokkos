/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/core/PhysicalUnits.hpp>
#include <corsika/framework/geometry/RootCoordinateSystem.hpp>
#include <corsika/framework/geometry/Vector.hpp>
#include <corsika/framework/stack/Stack.hpp>
#include <corsika/modules/qgsjetII/ParticleConversion.hpp>

#include <qgsjet-II-public.hpp>

namespace corsika::qgsjetII {

  class QGSJetIIFragmentsStackData {

  public:
    void dump() const {}

    void clear() {
      ::QGSJetII04::qgarr13_->nsf = 0;
      ::QGSJetII04::qgarr55_->nwt = 0;
    }
    unsigned int getSize() const { return ::QGSJetII04::qgarr13_->nsf; }
    unsigned int getCapacity() const { return iapmax; }

    static unsigned int getWoundedNucleonsTarget() { return ::QGSJetII04::qgarr55_->nwt; }
    static unsigned int getWoundedNucleonsProjectile() {
      return ::QGSJetII04::qgarr55_->nwp;
    }

    int getFragmentSize(const unsigned int i) const {
      return ::QGSJetII04::qgarr13_->iaf[i];
    }
    void setFragmentSize(const unsigned int i, const int v) {
      ::QGSJetII04::qgarr13_->iaf[i] = v;
    }

    void copy(const unsigned int i1, const unsigned int i2) {
      ::QGSJetII04::qgarr13_->iaf[i2] = ::QGSJetII04::qgarr13_->iaf[i1];
    }

    void swap(const unsigned int i1, const unsigned int i2) {
      std::swap(::QGSJetII04::qgarr13_->iaf[i1], ::QGSJetII04::qgarr13_->iaf[i2]);
    }

    void incrementSize() { ::QGSJetII04::qgarr13_->nsf++; }
    void decrementSize() {
      if (::QGSJetII04::qgarr13_->nsf > 0) { ::QGSJetII04::qgarr13_->nsf--; }
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

  typedef corsika::Stack<QGSJetIIFragmentsStackData, FragmentsInterface>
      QGSJetIIFragmentsStack;

} // end namespace corsika::qgsjetII

// #include <corsika/detail/modules/qgsjetII/QGSJetIIFragmentsStack.inl>
