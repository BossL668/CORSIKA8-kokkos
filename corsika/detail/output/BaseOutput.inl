/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

namespace corsika {
  inline void BaseOutput::writeSummary(boost::filesystem::path const& path) const {
    auto const summary = getSummary();
    if (!summary.IsNull()) YAMLStreamer{}.writeYAML(summary, path);
  }
} // namespace corsika
