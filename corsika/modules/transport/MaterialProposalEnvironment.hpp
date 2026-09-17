/* Construction-only view: inject resolved material properties into the original
 * PROPOSAL constructors without modifying the air application's modules. */
#pragma once
#include <corsika/modules/transport/MaterialModel.hpp>
#include <iomanip>
#include <sstream>

namespace corsika::interfaces {
struct MaterialMediumReference { MediumData const* data; };
// ADL in ProposalProcessBase's dependent mediumData(prop.getMedium()) call.
inline MediumData const& mediumData(MaterialMediumReference ref) { return *ref.data; }

inline std::string transportKey(MediumData const& d) {
  std::ostringstream s; s<<std::setprecision(17);
  for(double v:{d.Ieff_,d.Cbar_,d.x0_,d.x1_,d.aa_,d.sk_,d.dlt0_,d.corrected_density_}) s<<v<<',';
  return s.str();
}
template<class Environment> class MaterialProposalEnvironment {
  Environment const& env_;
  template<class Properties> struct PropertiesView {
    Properties const& p;
    MaterialMediumReference getMedium() const { return {&transportData(p)}; }
    auto const& getNuclearComposition() const { return p.getNuclearComposition(); }
  };
  template<class Node> struct NodeView {
    Node const& node;
    bool hasModelProperties() const { return node.hasModelProperties(); }
    auto getModelProperties() const {
      using P=std::remove_reference_t<decltype(node.getModelProperties())>;
      return PropertiesView<P>{node.getModelProperties()};
    }
  };
public:
  explicit MaterialProposalEnvironment(Environment const& env):env_(env) {
    // Include composition itself: the legacy composition hash is not collision
    // resistant, and equal hashes must never merge two distinct materials.
    std::map<std::size_t,std::string> keys;
    env_.getUniverse()->walk([&](auto const& node) {
      if(!node.hasModelProperties()) return;
      auto const& p=node.getModelProperties(); auto const& c=p.getNuclearComposition();
      std::vector<std::pair<int,double>> components;
      for(std::size_t i=0;i<c.getSize();++i)
        components.emplace_back(static_cast<int>(c.getComponents()[i]),c.getFractions()[i]);
      std::sort(components.begin(),components.end());
      std::ostringstream key; key<<transportKey(transportData(p))<<std::setprecision(17);
      for(auto x:components) key<<x.first<<':'<<x.second<<',';
      auto [it,inserted]=keys.emplace(c.getHash(),key.str());
      if(!inserted && it->second!=key.str())
        throw std::invalid_argument("different transport materials share a PROPOSAL composition key; independent calculator owners required");
    });
  }
  MaterialProposalEnvironment const* getUniverse() const { return this; }
  template<class Visitor> void walk(Visitor&& visitor) const {
    env_.getUniverse()->walk([&](auto const& node) {
      using N=std::remove_reference_t<decltype(node)>; NodeView<N> view{node}; visitor(view);
    });
  }
};
} // namespace corsika::interfaces
