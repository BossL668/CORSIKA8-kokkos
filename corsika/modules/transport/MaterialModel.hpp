/* Homogeneous, isotropic material cards shared by shower and radio adapters.
 * No YAML, geometry, PROPOSAL or device ownership is required by this interface. */
#pragma once
#include <corsika/modules/transport/HomogeneousMaterial.hpp>
#include <algorithm>
#include <limits>
#include <string>

namespace corsika::interfaces {
struct MaterialTransportProvider {
  virtual MediumData const& materialTransportData() const = 0;
  virtual ~MaterialTransportProvider() = default;
};
template<class Properties>
MediumData const& transportData(Properties const& properties) {
  if(auto p=dynamic_cast<MaterialTransportProvider const*>(&properties))
    return p->materialTransportData();
  return mediumData(properties.getMedium());
}
template<class Base> class MaterialPropertyModel : public Base, public MaterialTransportProvider {
  MediumData data_;
public:
  template<class... Args> MaterialPropertyModel(MediumData data, Args&&... args)
      : Base(std::forward<Args>(args)...), data_(std::move(data)) {}
  MediumData const& materialTransportData() const override { return data_; }
};

struct MaterialNucleus { int Z, A; double number_fraction; };
enum class IonisationModel { NistDensityScaled, BraggSternheimer, Explicit };

struct MaterialModel {
  std::string id, description, provenance;
  std::vector<MaterialNucleus> nuclei;
  double density_kg_m3{}, refractive_index{}, field_attenuation_length_m{};
  std::array<double,3> magnetic_field_T{};
  // This enum identifies the reference record only. transportProperties() is
  // authoritative; the granite/custom mixture never uses standard-rock data.
  Medium reference_medium{Medium::SiliconDioxideFusedQuartz};
  IonisationModel ionisation_model{IonisationModel::NistDensityScaled};
  MediumData explicit_transport{};

  void validate() const {
    if(id.empty() || nuclei.empty() || nuclei.size()>16 ||
       !std::isfinite(density_kg_m3) || density_kg_m3<=0. ||
       !std::isfinite(refractive_index) || refractive_index<1. ||
       !(field_attenuation_length_m>0.))
      throw std::invalid_argument("invalid material identity, density, index or field attenuation length");
    std::set<std::pair<int,int>> seen; double sum=0.;
    for(auto const& n:nuclei) {
      // The supported nuclear/hadronic interface uses A<=238 and Z<=92.
      if(n.Z<1 || n.Z>92 || n.A<n.Z || n.A>238 ||
         !std::isfinite(n.number_fraction) || n.number_fraction<=0. ||
         !seen.emplace(n.Z,n.A).second)
        throw std::invalid_argument("invalid or duplicate material nucleus");
      sum+=n.number_fraction;
    }
    if(std::abs(sum-1.)>1.e-10)
      throw std::invalid_argument("material requires normalized nucleus NUMBER fractions (not mass fractions)");
    for(double b:magnetic_field_T) if(!std::isfinite(b))
      throw std::invalid_argument("nonfinite material magnetic field");
    if(ionisation_model==IonisationModel::NistDensityScaled) {
      // Do not silently reuse a compound's I and density effect for a different
      // chemical recipe. Order-independent comparison, with isotope mass allowed.
      std::map<int,double> actual, expected;
      for(auto n:nuclei) actual[n.Z]+=n.number_fraction;
      if(reference_medium==Medium::SiliconDioxideFusedQuartz) expected={{8,2./3.},{14,1./3.}};
      else if(reference_medium==Medium::CalciumCarbonate) expected={{6,.2},{8,.6},{20,.2}};
      else if(reference_medium==Medium::WaterLiquid || reference_medium==Medium::WaterIce) expected={{1,2./3.},{8,1./3.}};
      else throw std::invalid_argument("unsupported NIST compound; use an explicit or Bragg-Sternheimer transport model");
      if(actual.size()!=expected.size()) throw std::invalid_argument("composition does not match NIST compound");
      for(auto const& p:expected) if(std::abs(actual[p.first]-p.second)>1.e-10)
        throw std::invalid_argument("composition does not match NIST compound");
    }
  }
  NuclearComposition composition() const {
    validate(); std::vector<Code> codes; std::vector<double> fractions;
    for(auto n:nuclei) {
      codes.push_back(get_nucleus_code(n.A,n.Z)); fractions.push_back(n.number_fraction);
    }
    return NuclearComposition(codes,fractions);
  }
  double ZOverA() const {
    validate();
    double z=0.,a=0.;
    for(auto n:nuclei) { z+=n.number_fraction*n.Z; a+=n.number_fraction*n.A; }
    return z/a; // A-number approximation, identical to PROPOSAL components.
  }
  std::vector<double> massFractions() const {
    validate();
    double a=0.; for(auto n:nuclei) a+=n.number_fraction*n.A;
    std::vector<double> w; for(auto n:nuclei) w.push_back(n.number_fraction*n.A/a);
    return w;
  }
  MediumData transportProperties() const {
    validate(); auto data=ionisation_model==IonisationModel::Explicit ? explicit_transport : mediumData(reference_medium);
    double const rho=density_kg_m3*.001;
    if(ionisation_model==IonisationModel::NistDensityScaled) {
      // Tabulated parameters are referenced to corrected_density, exactly as
      // in stock PROPOSAL construction. Shift delta(x) to the selected density.
      double const ratio=rho/data.getCorrectedDensity();
      double const shift=.5*std::log10(ratio);
      data.Cbar_-=std::log(ratio); data.x0_-=shift; data.x1_-=shift;
    } else if(ionisation_model==IonisationModel::BraggSternheimer) {
      // Bragg electron-weighted log(I); NIST elemental records occupy Z-1.
      // Chemical binding is not measured by this mixture approximation.
      double electrons=0.,logI=0.;
      for(auto n:nuclei) {
        double weight=n.number_fraction*n.Z;
        logI+=weight*std::log(mediumData(static_cast<Medium>(n.Z-1)).getIeff());
        electrons+=weight;
      }
      data.Ieff_=std::exp(logI/electrons);
      double const plasma=28.816*std::sqrt(rho*ZOverA()); // eV, rho in g/cm3
      data.Cbar_=1.+2.*std::log(data.Ieff_/plasma);
      bool low=data.Ieff_<100.;
      data.x0_=data.Cbar_<=(low?3.681:5.215) ? .2 : .326*data.Cbar_-(low?1.:1.5);
      data.x1_=low?2.:3.; data.sk_=3.; data.dlt0_=0.;
      data.aa_=(data.Cbar_-2.*std::log(10.)*data.x0_)/std::pow(data.x1_-data.x0_,3.);
    } else if(!std::isfinite(data.getCorrectedDensity()) || data.getCorrectedDensity()<=0. ||
              std::abs(data.getCorrectedDensity()-rho)>1.e-10*std::max(1.,rho)) {
      throw std::invalid_argument("explicit transport coefficients must be supplied at the selected material density");
    }
    data.name_=id; data.pretty_name_=description; data.Z_over_A_=ZOverA();
    data.sternheimer_density_=rho; data.corrected_density_=rho;
    data.state_=ionisation_model==IonisationModel::NistDensityScaled ?
        mediumData(reference_medium).getStateOfMatter() : StateOfMatter::Solid;
    data.type_=MediumType::Mixture;
    for(double v:{data.Ieff_,data.Cbar_,data.x0_,data.x1_,data.aa_,data.sk_,data.dlt0_})
      if(!std::isfinite(v)) throw std::invalid_argument("nonfinite material ionisation parameter");
    if(data.Ieff_<=0. || data.x1_<=data.x0_ || data.aa_<0. || data.sk_<=0. || data.dlt0_<0.)
      throw std::invalid_argument("invalid material ionisation/density-effect model");
    return data;
  }
  template<class Interface> auto makeModel(CoordinateSystemPtr const& cs) const {
    using namespace units::si;
    using Base=UniformRefractiveIndex<MediumPropertyModel<UniformMagneticField<HomogeneousMedium<Interface>>>>;
    return std::make_unique<MaterialPropertyModel<Base>>(transportProperties(),
        refractive_index,reference_medium,
        MagneticFieldVector(cs,magnetic_field_T[0]*1_T,magnetic_field_T[1]*1_T,magnetic_field_T[2]*1_T),
        density_kg_m3*1_kg/(1_m*1_m*1_m),composition());
  }
};

inline MaterialModel materialPreset(std::string const& name) {
  MaterialModel m; m.provenance="documentation/materials.yaml (2026-09-13); synthetic radio constants, not site measurements";
  if(name=="SiO2" || name=="S0_SILICA_TRANSPORT_PROXY") {
    m.id="S0_SILICA_TRANSPORT_PROXY"; m.description="Silica transport proxy (SiO2)";
    m.nuclei={{14,28,1./3.},{8,16,2./3.}};
    m.density_kg_m3=2650.; m.refractive_index=std::sqrt(5.); m.field_attenuation_length_m=100.;
  } else if(name=="Limestone" || name=="Calcite" || name=="L0_LIMESTONE_CALCITE_PROXY") {
    m.id="L0_LIMESTONE_CALCITE_PROXY"; m.description="Dense calcite / limestone proxy (CaCO3)";
    m.nuclei={{6,12,.2},{8,16,.6},{20,40,.2}}; m.reference_medium=Medium::CalciumCarbonate;
    m.density_kg_m3=2710.; m.refractive_index=std::sqrt(6.); m.field_attenuation_length_m=14.476482730108396;
  } else if(name=="Granite" || name=="G0_GRANITE_REFERENCE") {
    m.id="G0_GRANITE_REFERENCE"; m.description="Granite reference mixture (H, C, O, Na, Mg, Al, Si, K, Ca, Fe); PNNL composition, Bragg-Sternheimer ionisation approximation";
    m.nuclei={{1,1,.027122},{6,12,.000502},{8,16,.607735},{11,23,.025866},{12,24,.018081},
              {13,27,.062783},{14,28,.205927},{19,39,.013938},{20,40,.018960},{26,56,.019086}};
    m.density_kg_m3=2729.; m.refractive_index=std::sqrt(5.); m.field_attenuation_length_m=86.85889638065036;
    m.ionisation_model=IonisationModel::BraggSternheimer;
    m.reference_medium=Medium::StandardRock; // compatibility enum only; never its coefficients
  } else if(name=="Water" || name=="Ice") {
    m.id=name; m.description=name+" (H2O) homogeneous compatibility model";
    m.nuclei={{1,1,2./3.},{8,16,1./3.}};
    m.reference_medium=name=="Water"?Medium::WaterLiquid:Medium::WaterIce;
    m.density_kg_m3=name=="Water"?1000.:917.; m.refractive_index=name=="Water"?1.33:1.78;
    m.field_attenuation_length_m=std::numeric_limits<double>::infinity();
    m.provenance="Legacy compatibility defaults; supply a measured radio model for physical studies";
  } else throw std::invalid_argument("unknown material preset: "+name);
  m.transportProperties(); return m;
}
} // namespace corsika::interfaces
