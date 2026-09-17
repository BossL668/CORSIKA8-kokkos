// YAML admission and reproducible, resolved output for independent material cards.
#pragma once
#include <corsika/modules/transport/MaterialModel.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace corsika::applications::terrain {
inline void materialKeys(YAML::Node const& n, std::set<std::string> const& allowed) {
  if(!n.IsMap()) throw std::invalid_argument("material section must be a mapping");
  for(auto const& p:n) if(!allowed.count(p.first.as<std::string>()))
    throw std::invalid_argument("unknown material field: "+p.first.as<std::string>());
}
inline interfaces::MaterialModel readMaterial(YAML::Node const& input) {
  using namespace interfaces;
  if(!input || input.IsNull()) return materialPreset("SiO2");
  if(input.IsScalar()) return materialPreset(input.as<std::string>());
  materialKeys(input,{"preset","id","description","provenance","transport","radio","magnetic_field_enu_T"});
  auto m=materialPreset(input["preset"].as<std::string>("SiO2"));
  if(input["id"]) m.id=input["id"].as<std::string>();
  if(input["description"]) m.description=input["description"].as<std::string>();
  if(input["provenance"]) m.provenance=input["provenance"].as<std::string>();
  if(auto t=input["transport"]) {
    materialKeys(t,{"density_kg_m3","composition_basis","nuclei","ionisation"});
    if(t["density_kg_m3"]) m.density_kg_m3=t["density_kg_m3"].as<double>();
    if(t["composition_basis"] && t["composition_basis"].as<std::string>()!="nucleus_number_fraction")
      throw std::invalid_argument("material composition_basis must be nucleus_number_fraction");
    if(auto ns=t["nuclei"]) {
      if(!ns.IsSequence()) throw std::invalid_argument("material nuclei must be a sequence");
      m.nuclei.clear();
      for(auto n:ns) {
        materialKeys(n,{"element","Z","A","number_fraction"});
        m.nuclei.push_back({n["Z"].as<int>(),n["A"].as<int>(),n["number_fraction"].as<double>()});
      }
      if(!t["ionisation"])
        throw std::invalid_argument("a changed composition requires an explicit ionisation model");
    }
    if(auto p=t["ionisation"]) {
      materialKeys(p,{"model","reference_medium","I_eV","Cbar","x0","x1","a","m","delta0","reference_density_kg_m3"});
      auto model=p["model"].as<std::string>();
      if(model=="nist_density_scaled") {
        materialKeys(p,{"model","reference_medium"});
        m.ionisation_model=IonisationModel::NistDensityScaled;
        if(p["reference_medium"]) m.reference_medium=materialPreset(p["reference_medium"].as<std::string>()).reference_medium;
      } else if(model=="bragg_sternheimer") {
        materialKeys(p,{"model"}); m.ionisation_model=IonisationModel::BraggSternheimer;
      }
      else if(model=="explicit") {
        materialKeys(p,{"model","I_eV","Cbar","x0","x1","a","m","delta0","reference_density_kg_m3"});
        m.ionisation_model=IonisationModel::Explicit;
        auto& d=m.explicit_transport;
        d.Ieff_=p["I_eV"].as<double>(); d.Cbar_=p["Cbar"].as<double>();
        d.x0_=p["x0"].as<double>(); d.x1_=p["x1"].as<double>();
        d.aa_=p["a"].as<double>(); d.sk_=p["m"].as<double>(); d.dlt0_=p["delta0"].as<double>();
        d.corrected_density_=p["reference_density_kg_m3"].as<double>()*.001;
      } else throw std::invalid_argument("unknown material ionisation model: "+model);
    }
  }
  if(auto r=input["radio"]) {
    materialKeys(r,{"model","refractive_index","field_attenuation_length_m","relative_permeability",
                   "material_dispersion_enabled","frequency_dependent_attenuation_enabled"});
    if(r["model"].as<std::string>("constant_real_index_and_grey_field_attenuation")!="constant_real_index_and_grey_field_attenuation" ||
       r["relative_permeability"].as<double>(1.)!=1. || r["material_dispersion_enabled"].as<bool>(false) ||
       r["frequency_dependent_attenuation_enabled"].as<bool>(false))
      throw std::invalid_argument("material radio currently requires isotropic mu_r=1, constant real n and grey field attenuation");
    if(r["refractive_index"]) m.refractive_index=r["refractive_index"].as<double>();
    if(r["field_attenuation_length_m"]) m.field_attenuation_length_m=r["field_attenuation_length_m"].as<double>();
  }
  if(auto b=input["magnetic_field_enu_T"]) {
    if(!b.IsSequence() || b.size()!=3) throw std::invalid_argument("material magnetic field requires three ENU components");
    for(int i=0;i<3;++i) m.magnetic_field_T[i]=b[i].as<double>();
  }
  m.transportProperties(); return m;
}
inline YAML::Node materialConfig(interfaces::MaterialModel const& m) {
  auto d=m.transportProperties(); YAML::Node n;
  n["id"]=m.id; n["description"]=m.description; n["provenance"]=m.provenance;
  auto t=n["transport"]; t["density_kg_m3"]=m.density_kg_m3;
  t["composition_basis"]="nucleus_number_fraction";
  for(auto c:m.nuclei) {
    YAML::Node row; row["element"]=mediumData(static_cast<Medium>(c.Z-1)).getSymbol();
    row["Z"]=c.Z; row["A"]=c.A; row["number_fraction"]=c.number_fraction;
    t["nuclei"].push_back(row);
  }
  auto p=t["ionisation"]; p["model"]="explicit";
  p["reference_density_kg_m3"]=d.corrected_density_*1000.;
  p["I_eV"]=d.Ieff_; p["Cbar"]=d.Cbar_; p["x0"]=d.x0_; p["x1"]=d.x1_;
  p["a"]=d.aa_; p["m"]=d.sk_; p["delta0"]=d.dlt0_;
  n["radio"]["model"]="constant_real_index_and_grey_field_attenuation";
  n["radio"]["refractive_index"]=m.refractive_index;
  n["radio"]["field_attenuation_length_m"]=m.field_attenuation_length_m;
  n["radio"]["relative_permeability"]=1.; n["magnetic_field_enu_T"]=m.magnetic_field_T;
  return n;
}
} // namespace corsika::applications::terrain
