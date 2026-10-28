/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

namespace corsika {
  inline ShowerSummary::ShowerSummary(ShowerSummary const& other)
      : current_(YAML::Clone(other.current_))
      , lastOffset_(other.lastOffset_)
      , lastId_(other.lastId_)
      , hasEvent_(other.hasEvent_) {
    if (!other.journal_.empty()) {
      journal_ = other.journal_.parent_path() /
                 boost::filesystem::unique_path(".summary-%%%%-%%%%.yaml");
      boost::filesystem::copy_file(other.journal_, journal_);
    }
  }

  inline ShowerSummary& ShowerSummary::operator=(ShowerSummary const& other) {
    ShowerSummary copy(other);
    swap(copy);
    return *this;
  }

  inline ShowerSummary::~ShowerSummary() {
    if (!journal_.empty()) {
      boost::system::error_code ignored;
      boost::filesystem::remove(journal_, ignored);
    }
  }

  inline void ShowerSummary::open(boost::filesystem::path const& directory) {
    auto path = directory / boost::filesystem::unique_path(".summary-%%%%-%%%%.yaml");
    std::ofstream stream(path.string(), std::ios::binary);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.close();
    ShowerSummary fresh;
    fresh.journal_ = std::move(path);
    swap(fresh);
  }

  inline YAML::Node ShowerSummary::event(unsigned int id) {
    if (hasEvent_ && id < lastId_)
      throw std::logic_error("ShowerSummary cannot rewrite an earlier shower");
    if (!hasEvent_ || id != lastId_) {
      flush();
      lastOffset_ = journal_.empty() ? 0 : boost::filesystem::file_size(journal_);
      lastId_ = id;
      hasEvent_ = true;
      current_.reset(YAML::Node(YAML::NodeType::Map));
    } else if (current_.IsNull()) {
      std::ifstream stream(journal_.string(), std::ios::binary);
      stream.exceptions(std::ios::badbit | std::ios::failbit);
      stream.seekg(static_cast<std::streamoff>(lastOffset_));
      std::string text{std::istreambuf_iterator<char>(stream), {}};
      current_.reset(YAML::Load(text));
    }
    return current_["shower_" + std::to_string(id)];
  }

  inline void ShowerSummary::flush() const {
    if (current_.IsNull()) return;
    if (journal_.empty())
      throw std::logic_error("ShowerSummary requires startOfLibrary before flush");
    YAML::Emitter emitter;
    emitter << current_;
    if (!emitter.good()) throw std::runtime_error(emitter.GetLastError());
    std::string const text = std::string(emitter.c_str()) + "\n";
    std::fstream stream(journal_.string(),
                        std::ios::in | std::ios::out | std::ios::binary);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.seekp(static_cast<std::streamoff>(lastOffset_));
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.flush();
    stream.close();
    boost::filesystem::resize_file(journal_, lastOffset_ + text.size());
    current_.reset();
  }

  inline YAML::Node ShowerSummary::snapshot() const {
    if (!hasEvent_) return YAML::Node();
    // Some diagnostic callers inspect an uninitialized output. No disk
    // operation is necessary until a library has actually been opened.
    if (journal_.empty()) return YAML::Clone(current_);
    flush();
    return YAML::LoadFile(journal_.string());
  }

  inline void ShowerSummary::writeSummary(
      boost::filesystem::path const& destination) const {
    flush();
    if (!hasEvent_) return;
    auto temporary = destination;
    temporary += ".tmp";
    boost::filesystem::copy_file(journal_, temporary,
                                 boost::filesystem::copy_options::overwrite_existing);
    boost::filesystem::rename(temporary, destination);
  }

  inline bool ShowerSummary::hasResidentRecord() const noexcept {
    return !current_.IsNull();
  }

  inline void ShowerSummary::swap(ShowerSummary& other) {
    using std::swap;
    swap(journal_, other.journal_);
    auto current = current_;
    current_.reset(other.current_);
    other.current_.reset(current);
    swap(lastOffset_, other.lastOffset_);
    swap(lastId_, other.lastId_);
    swap(hasEvent_, other.hasEvent_);
  }
} // namespace corsika
