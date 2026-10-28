/*
 * (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/framework/geometry/IVolume.hpp>
#include <corsika/media/IEmpty.hpp>
#include <memory>
#include <vector>

namespace corsika {

  template <typename TModelProperties = IEmpty>
  class VolumeTreeNode {

  public:
    using IModelProperties = TModelProperties;
    using VTN_type = VolumeTreeNode<IModelProperties>;
    using VTNUPtr = std::unique_ptr<VTN_type>;
    using IMPSharedPtr = std::shared_ptr<IModelProperties>;
    using VolUPtr = std::unique_ptr<IVolume>;

    VolumeTreeNode(VolUPtr pVolume = nullptr)
        : geoVolume_(std::move(pVolume)) {}

    //! convenience function equivalent to Volume::isInside
    bool contains(Point const& p) const;

    VTN_type const* excludes(Point const& p) const;

    /** returns a pointer to the sub-VolumeTreeNode which is "responsible" for the given
     * \class Point \p p, or nullptr iff \p p is not contained in this volume.
     */
    VolumeTreeNode<IModelProperties> const* getContainingNode(Point const& p) const;
    VolumeTreeNode<IModelProperties>* getContainingNode(Point const& p);

    /**
     * Traverses the VolumeTree pre- or post-order and calls the functor  \p func for each
     * node. \p func takes a reference to VolumeTreeNode as argument. The return value \p
     * func is ignored.
     */
    template <typename TCallable, bool preorder = true>
    void walk(TCallable func) const;

    void addChild(VTNUPtr pChild);

    /**
     * Adds a child to the node containing \p using the same logic as getContainingNode
     */
    void addChildToContainingNode(Point const& p, VTNUPtr pChild);

    void excludeOverlapWith(VTNUPtr const& pNode);

    VTN_type const* getParent() const { return parentNode_; };

    auto const& getChildNodes() const { return childNodes_; }

    auto const& getExcludedNodes() const { return excludedNodes_; }

    auto const& getVolume() const { return *geoVolume_; }

    /** Whether this node owns a volume that can be read with getVolume(). */
    bool hasVolume() const { return geoVolume_ != nullptr; }

    auto const& getModelProperties() const { return *modelProperties_; }

    bool hasModelProperties() const { return modelProperties_.get() != nullptr; }

    template <typename ModelProperties, typename... Args>
    auto setModelProperties(Args&&... args) {
      // static_assert(std::is_base_of_v<IModelProperties, ModelProperties>,
      //            "unusable type provided");
      modelProperties_ = std::make_shared<ModelProperties>(std::forward<Args>(args)...);
      return modelProperties_;
    }

    void setModelProperties(IMPSharedPtr ptr) { modelProperties_ = ptr; }

    // template <class MediumType, typename... Args>
    // static auto createMedium(Args&&... args);

  private:
    std::vector<VTNUPtr> childNodes_;
    std::vector<VTN_type const*> excludedNodes_;
    VTN_type const* parentNode_ = nullptr;
    VolUPtr geoVolume_;
    IMPSharedPtr modelProperties_;
  };

} // namespace corsika

#include <corsika/detail/media/VolumeTreeNode.inl>
