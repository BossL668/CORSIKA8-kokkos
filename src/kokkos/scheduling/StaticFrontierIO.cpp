/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

// Native I/O for single-shower multi-GPU coordination.
#include <corsika/kokkos/scheduling/AirShowerCoordinator.hpp>
#include <corsika/kokkos/radio/RadioTrackSpoolReader.hpp>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <openssl/evp.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unistd.h>

namespace corsika::kokkos::coordinator {
  namespace {
    void require(bool ok, std::string const& message) {
      if (!ok) throw std::runtime_error(message);
    }
    void check(arrow::Status const& s) {
      if (!s.ok()) throw std::runtime_error(s.ToString());
    }
    template <typename T>
    T take(arrow::Result<T> r) {
      check(r.status());
      return std::move(r).ValueOrDie();
    }
    struct Hash {
      std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx_{EVP_MD_CTX_new(),
                                                                   EVP_MD_CTX_free};
      Hash() {
        require(ctx_ && EVP_DigestInit_ex(ctx_.get(), EVP_sha256(), nullptr) == 1,
                "SHA256 init failed");
      }
      void add(void const* p, std::size_t n) {
        require(EVP_DigestUpdate(ctx_.get(), p, n) == 1, "SHA256 update failed");
      }
      std::array<unsigned char, 32> finish() {
        std::array<unsigned char, 32> out{};
        unsigned n = 0;
        require(EVP_DigestFinal_ex(ctx_.get(), out.data(), &n) == 1 && n == 32,
                "SHA256 final failed");
        return out;
      }
    };
    std::unique_ptr<parquet::arrow::FileReader> reader(fs::path const& file) {
      auto input = take(arrow::io::ReadableFile::Open(file.string()));
      std::unique_ptr<parquet::arrow::FileReader> r;
      check(parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &r));
      r->set_use_threads(false);
      r->set_batch_size(65536);
      return r;
    }
    std::shared_ptr<arrow::Table> table(fs::path const& file) {
      auto r = reader(file);
      std::shared_ptr<arrow::Table> t;
      check(r->ReadTable(&t));
      check(t->ValidateFull());
      return t;
    }
    void validate_array(std::shared_ptr<arrow::Array> const& a) {
      require(a->null_count() == 0, "Nulls in output");
      if (a->type_id() == arrow::Type::DOUBLE) {
        for (auto x : *std::static_pointer_cast<arrow::DoubleArray>(a))
          require(std::isfinite(*x), "Non-finite output");
      } else if (a->type_id() == arrow::Type::FLOAT) {
        for (auto x : *std::static_pointer_cast<arrow::FloatArray>(a))
          require(std::isfinite(*x), "Non-finite output");
      }
    }
    template <typename T>
    void append(std::shared_ptr<arrow::Array> const& a, std::vector<double>& out) {
      for (auto x : *std::static_pointer_cast<T>(a))
        out.push_back(static_cast<double>(*x));
    }
    std::vector<double> values(std::shared_ptr<arrow::ChunkedArray> const& col) {
      std::vector<double> out;
      out.reserve(col->length());
      for (auto const& a : col->chunks()) {
        validate_array(a);
        switch (a->type_id()) {
          case arrow::Type::DOUBLE:
            append<arrow::DoubleArray>(a, out);
            break;
          case arrow::Type::FLOAT:
            append<arrow::FloatArray>(a, out);
            break;
          case arrow::Type::INT32:
            append<arrow::Int32Array>(a, out);
            break;
          case arrow::Type::UINT32:
            append<arrow::UInt32Array>(a, out);
            break;
          case arrow::Type::INT64:
            append<arrow::Int64Array>(a, out);
            break;
          case arrow::Type::UINT64:
            append<arrow::UInt64Array>(a, out);
            break;
          default:
            throw std::runtime_error("Non-numeric observable: " + a->type()->ToString());
        }
      }
      return out;
    }
    json_type yaml_value(YAML::Node const& n) {
      if (!n || n.IsNull()) return nullptr;
      if (n.IsScalar()) return n.Scalar();
      if (n.IsSequence()) {
        json_type out = json_type::array();
        for (auto const& x : n) out.push_back(yaml_value(x));
        return out;
      }
      json_type out = json_type::object();
      for (auto const& x : n) out[x.first.as<std::string>()] = yaml_value(x.second);
      return out;
    }
    void configs(std::vector<fs::path> const& parts, fs::path const& rel) {
      auto ref = yaml_value(YAML::LoadFile((parts.front() / rel).string()));
      for (auto const& p : parts)
        require(yaml_value(YAML::LoadFile((p / rel).string())) == ref,
                "Configuration mismatch: " + (p / rel).string());
    }
  } // namespace
  void save(fs::path const& file, json_type const& value) {
    auto tmp = file.string() + ".pending";
    auto data = value.dump(2) + "\n";
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    require(fd >= 0, "Cannot create metadata: " + tmp);
    std::size_t offset = 0;
    while (offset < data.size()) {
      auto n = write(fd, data.data() + offset, data.size() - offset);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        close(fd);
        throw std::runtime_error("Metadata write failed: " + tmp);
      }
      offset += static_cast<std::size_t>(n);
    }
    require(close(fd) == 0, "Metadata close failed");
    // Publish only closed metadata; link refuses to overwrite an existing marker.
    require(link(tmp.c_str(), file.c_str()) == 0,
            "Cannot publish metadata: " + file.string());
    require(unlink(tmp.c_str()) == 0, "Cannot remove metadata staging link");
  }
  std::string digest(fs::path const& file) {
    Hash h;
    std::ifstream in(file, std::ios::binary);
    require(bool(in), "Cannot hash " + file.string());
    // The CUDA-linked application already has a large caller stack frame. Keep
    // bulk file I/O off that stack (also safe for small guarded worker stacks).
    // This changes allocation only, not the bytes hashed or the SHA256 algorithm.
    std::vector<char> buf(65536);
    while (in) {
      in.read(buf.data(), buf.size());
      h.add(buf.data(), in.gcount());
    }
    require(in.eof(), "Hash read failed");
    auto bytes = h.finish();
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto b : bytes) out << std::setw(2) << unsigned(b);
    return out.str();
  }
  std::uint32_t worker_seed(std::uint64_t seed, unsigned index) {
    // Same little-endian SHA256 seed derivation as the Python reference.
    Hash h;
    auto text = std::to_string(seed) + ":" + std::to_string(index);
    h.add(text.data(), text.size());
    auto b = h.finish();
    std::uint32_t v = 0;
    for (unsigned i = 0; i < 4; ++i) v |= std::uint32_t(b[i]) << (8 * i);
    return v ? v : 1;
  }
  json_type partition(fs::path const& source, std::vector<fs::path> const& destinations) {
    require(!destinations.empty(), "No shard destinations");
    std::ifstream in(source);
    std::string line;
    require(bool(std::getline(in, line)), "Missing frontier header");
    auto const header = line;
    bool const cpuRoots = header.rfind("C8_CPU_FRONTIER_V1 ", 0) == 0;
    bool const logicalNodes = cpuRoots ||
                              header.rfind("C8_STATIC_FRONTIER_V2 ", 0) == 0 ||
                              header.rfind("C8_STATIC_FRONTIER_V3 ", 0) == 0;
    require(logicalNodes || header == "C8_STATIC_FRONTIER_V1", "Invalid frontier header");
    struct Root {
      double energy_, weight_;
      std::uint64_t history_;
      std::string line_;
    };
    std::vector<Root> roots;
    std::set<std::uint64_t> seen;
    long double weighted = 0;
    while (std::getline(in, line)) {
      std::istringstream row(line);
      std::vector<std::string> c{std::istream_iterator<std::string>(row), {}};
      require(c.size() == (cpuRoots ? 16u : logicalNodes ? 15u : 14u),
              "Invalid frontier column count");
      std::size_t used = 0;
      auto history = std::stoull(c[1], &used);
      require(used == c[1].size() && c[1].front() != '-' && history > 0,
              "Invalid frontier history");
      auto energy = std::stod(c[5], &used);
      require(used == c[5].size(), "Invalid energy");
      auto weight = std::stod(c[6], &used);
      require(used == c[6].size() && std::isfinite(energy * weight) &&
                  std::isfinite(energy) && std::isfinite(weight) && energy >= 0 &&
                  weight >= 0 && seen.insert(history).second,
              "Duplicate/invalid root");
      weighted += static_cast<long double>(energy) * weight;
      roots.push_back({energy, weight, history, line});
    }
    require(in.eof(), "Frontier read failed");
    std::sort(roots.begin(), roots.end(), [](Root const& a, Root const& b) {
      return std::tie(a.energy_, a.history_) > std::tie(b.energy_, b.history_);
    });
    std::vector<std::ofstream> files;
    for (auto const& p : destinations) {
      require(!fs::exists(p), "Shard exists");
      files.emplace_back(p);
      require(bool(files.back()), "Cannot create shard");
      files.back() << header << '\n';
    }
    std::vector<double> cost(destinations.size(), 0);
    std::vector<std::size_t> counts(destinations.size(), 0);
    for (auto const& r : roots) {
      auto k = std::min_element(cost.begin(), cost.end()) - cost.begin();
      files[k] << r.line_ << '\n';
      cost[k] += r.energy_;
      ++counts[k];
      require(std::isfinite(cost[k]), "Frontier cost overflow");
    }
    std::vector<std::string> hashes;
    for (std::size_t i = 0; i < files.size(); ++i) {
      files[i].close();
      require(bool(files[i]), "Shard write failed");
      hashes.push_back(digest(destinations[i]));
    }
    require(std::isfinite(double(weighted)), "Weighted energy overflow");
    return {{"roots", roots.size()},
            {"counts", counts},
            {"cost_energy_GeV", cost},
            {"weighted_energy_GeV", double(weighted)},
            {"frontier_sha256", digest(source)},
            {"shard_sha256", hashes}};
  }
  json_type join_frontiers(std::vector<fs::path> const& sources,
                           fs::path const& destination) {
    require(!sources.empty() && !fs::exists(destination),
            "Invalid frontier join destination/sources");
    std::ofstream out(destination);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    std::string header, line;
    std::set<std::uint64_t> seen;
    json_type hashes = json_type::array();
    for (auto const& path : sources) {
      std::ifstream in(path);
      std::string current;
      require(bool(std::getline(in, current)), "Missing CPU-produced frontier header");
      if (header.empty()) {
        header = current;
        out << header << '\n';
      }
      require(current == header && header.rfind("C8_STATIC_FRONTIER_V3 ", 0) == 0,
              "Incompatible CPU-produced frontier headers");
      while (std::getline(in, line)) {
        int pid;
        std::uint64_t id;
        std::istringstream row(line);
        require(bool(row >> pid >> id) && seen.insert(id).second,
                "Duplicate joined frontier root");
        out << line << '\n';
      }
      require(in.eof(), "CPU-produced frontier read failed");
      hashes.push_back({{"path", path.string()}, {"sha256", digest(path)}});
    }
    out.close();
    return {{"roots", seen.size()},
            {"sources", hashes},
            {"joined_sha256", digest(destination)}};
  }
  json_type partition_radio_tracks(std::vector<fs::path> const& sources,
                                   std::vector<unsigned> const& workers) {
    // Independent, immutable track files are assigned only once. Particle
    // frontier allocation and random histories are completely unchanged.
    struct Spool {
      fs::path path_;
      std::uint64_t count_;
      std::string hash_;
    };
    std::vector<Spool> spools;
    std::set<fs::path> seen;
    std::uint64_t total = 0;
    for (auto const& path : sources) {
      require(seen.insert(fs::canonical(path)).second, "Duplicate prefix radio spool");
      corsika::kokkos::radio::RadioTrackSpoolReader input(path);
      auto const count = input.count();
      require(count <= std::numeric_limits<std::uint64_t>::max() - total,
              "Prefix radio record count overflow");
      total += count;
      spools.push_back({path, count, digest(path)});
    }
    require(!total || !workers.empty(), "Prefix radio has no receiving GPU worker");
    require(std::set<unsigned>(workers.begin(), workers.end()).size() == workers.size(),
            "Duplicate prefix radio worker");
    std::stable_sort(spools.begin(), spools.end(),
                     [](auto const& a, auto const& b) { return a.count_ > b.count_; });
    std::vector<std::uint64_t> counts(workers.size());
    json_type assignments = json_type::array();
    for (auto const& spool : spools) {
      json_type assignment{{"path", spool.path_.string()},
                           {"records", spool.count_},
                           {"sha256", spool.hash_},
                           {"worker", nullptr}};
      if (spool.count_) {
        auto const index =
            std::min_element(counts.begin(), counts.end()) - counts.begin();
        counts[index] += spool.count_;
        assignment["worker"] = workers[index];
      }
      assignments.push_back(std::move(assignment));
    }
    return {{"mode", "static-whole-spool"},
            {"records", total},
            {"workers", workers},
            {"worker_records", counts},
            {"sources", assignments}};
  }
  void validate_outputs(fs::path const& folder) {
    for (auto rel :
         {"profile/profile.parquet", "CoREAS/observers.parquet", "ZHS/observers.parquet"})
      require(reader(folder / rel)->parquet_reader()->metadata()->num_rows() > 0,
              "Empty required output (select --ring 1 or an antenna file): " +
                  std::string(rel));
    for (auto const& path : fs::recursive_directory_iterator(folder)) {
      if (path.path().extension() != ".parquet") continue;
      auto r = reader(path.path());
      std::shared_ptr<arrow::RecordBatchReader> batches;
      check(r->GetRecordBatchReader(&batches));
      for (;;) {
        check_interrupt();
        auto batch = take(batches->Next());
        if (!batch) break;
        check(batch->ValidateFull());
        for (auto const& c : batch->columns()) validate_array(c);
      }
    }
    auto p = folder / "gpu_em/summary.yaml";
    if (fs::exists(p)) {
      auto stats = YAML::LoadFile(p.string())["shower_0"]["statistics"];
      require(stats["queue_overflows"].as<std::uint64_t>() == 0 &&
                  stats["radio"]["fixed_point_overflows"].as<std::uint64_t>() == 0,
              "Queue/fixed-point overflow");
    }
    auto native = folder / "native_egs4/summary.yaml";
    if (fs::exists(native)) {
      auto s = YAML::LoadFile(native.string())["shower_0"];
      require(s["status"].as<std::string>() == "completed" &&
                  s["scalar_em_steps"].as<std::uint64_t>() == 0 &&
                  s["radio_fixed_point_overflows"].as<std::uint64_t>() == 0 &&
                  s["profile_fixed_point_overflows"].as<std::uint64_t>() == 0 &&
                  s["profile_invalid_records"].as<std::uint64_t>() == 0,
              "Incomplete/invalid native EGS4 output");
    }
  }
  json_type merge(std::vector<fs::path> const& parts, fs::path const& destination) {
    require(!parts.empty(), "Cannot merge no parts");
    require(fs::create_directory(destination), "Merge destination exists");
    json_type products = json_type::object();
    for (auto name : {"profile/profile.parquet", "production_profile/profile.parquet",
                      "energyloss/dEdX.parquet", "CoREAS/observers.parquet",
                      "ZHS/observers.parquet"}) {
      check_interrupt();
      fs::path rel(name), target = destination / rel;
      bool radio = rel.begin()->string() == "CoREAS" || rel.begin()->string() == "ZHS";
      auto grid = radio ? "Time" : "X";
      auto ref = table(parts.front() / rel);
      require(ref->GetColumnByName("shower") && ref->GetColumnByName(grid),
              "Missing grid columns");
      configs(parts, rel.parent_path() / "config.yaml");
      std::vector<std::vector<double>> sum(ref->num_columns()),
          correction(ref->num_columns());
      for (int k = 0; k < ref->num_columns(); ++k) {
        auto key = ref->field(k)->name();
        for (auto const& a : ref->column(k)->chunks()) validate_array(a);
        if (key == "shower" || key == grid) continue;
        sum[k] = values(ref->column(k));
        correction[k].resize(sum[k].size(), 0);
      }
      for (std::size_t p = 1; p < parts.size(); ++p) {
        auto t = table(parts[p] / rel);
        require(t->schema()->Equals(*ref->schema(), false) &&
                    t->num_rows() == ref->num_rows(),
                "Schema/shape mismatch: " + rel.string());
        require(t->GetColumnByName("shower")->Equals(ref->GetColumnByName("shower")) &&
                    t->GetColumnByName(grid)->Equals(ref->GetColumnByName(grid)),
                "Grid mismatch: " + rel.string());
        for (int k = 0; k < ref->num_columns(); ++k) {
          auto key = ref->field(k)->name();
          if (key == "shower" || key == grid) continue;
          auto v = values(t->column(k));
          for (std::size_t j = 0; j < v.size(); ++j) {
            auto y = v[j] - correction[k][j], total = sum[k][j] + y;
            correction[k][j] = (total - sum[k][j]) - y;
            sum[k][j] = total;
          }
        }
      }
      std::vector<std::shared_ptr<arrow::Field>> fields;
      std::vector<std::shared_ptr<arrow::ChunkedArray>> arrays;
      std::size_t nonzero = 0;
      double maximum = 0;
      for (int k = 0; k < ref->num_columns(); ++k) {
        auto key = ref->field(k)->name();
        if (key == "shower" || key == grid) {
          fields.push_back(ref->field(k));
          arrays.push_back(ref->column(k));
        } else {
          for (auto x : sum[k]) {
            require(std::isfinite(x), "Non-finite merged output");
            nonzero += x != 0;
            maximum = std::max(maximum, std::abs(x));
          }
          arrow::DoubleBuilder b;
          check(b.AppendValues(sum[k]));
          arrays.push_back(std::make_shared<arrow::ChunkedArray>(take(b.Finish())));
          fields.push_back(arrow::field(key, arrow::float64()));
        }
      }
      fs::create_directories(target.parent_path());
      auto out = take(arrow::io::FileOutputStream::Open(target.string()));
      auto t = arrow::Table::Make(arrow::schema(fields), arrays);
      check(parquet::arrow::WriteTable(*t, arrow::default_memory_pool(), out, 65536));
      check(out->Close());
      fs::copy_file(parts.front() / rel.parent_path() / "config.yaml",
                    target.parent_path() / "config.yaml");
      products[name] = {{"rows", t->num_rows()}, {"sha256", digest(target)}};
      if (radio) {
        products[name]["nonzero_field_entries"] = nonzero;
        products[name]["maximum_absolute_field"] = maximum;
      }
    }
    for (auto name :
         {"particles/particles.parquet", "interactions/interactions.parquet"}) {
      check_interrupt();
      fs::path rel(name), target = destination / rel;
      configs(parts, rel.parent_path() / "config.yaml");
      fs::create_directories(target.parent_path());
      std::shared_ptr<arrow::Schema> schema;
      check(reader(parts.front() / rel)->GetSchema(&schema));
      auto out = take(arrow::io::FileOutputStream::Open(target.string()));
      auto writer = take(
          parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out));
      std::int64_t rows = 0;
      auto count = rel.begin()->string() == "interactions" ? 1 : parts.size();
      for (std::size_t i = 0; i < count; ++i) {
        auto r = reader(parts[i] / rel);
        std::shared_ptr<arrow::Schema> s;
        check(r->GetSchema(&s));
        require(schema->Equals(*s, false), "Record schema mismatch");
        std::shared_ptr<arrow::RecordBatchReader> batches;
        check(r->GetRecordBatchReader(&batches));
        for (;;) {
          check_interrupt();
          auto batch = take(batches->Next());
          if (!batch) break;
          for (auto const& c : batch->columns()) validate_array(c);
          check(writer->WriteRecordBatch(*batch));
          rows += batch->num_rows();
        }
      }
      check(writer->Close());
      check(out->Close());
      fs::copy_file(parts.front() / rel.parent_path() / "config.yaml",
                    target.parent_path() / "config.yaml");
      auto summary = parts.front() / rel.parent_path() / "summary.yaml";
      if (rel.begin()->string() == "interactions" && fs::exists(summary))
        fs::copy_file(summary, target.parent_path() / "summary.yaml");
      products[name] = {{"rows", rows}, {"sha256", digest(target)}};
    }
    std::vector<std::string> names;
    for (auto const& p : parts) names.push_back(p.string());
    save(destination / "MERGE.json",
         {{"parts", names},
          {"products", products},
          {"arithmetic",
           "compensated float64 sum of closed arrays; no cross-worker fixed-point claim"},
          {"statistical_acceptance", false}});
    return products;
  }
} // namespace corsika::kokkos::coordinator
