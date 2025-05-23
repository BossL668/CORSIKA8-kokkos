/*
 * (c) Copyright 2025 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#pragma once

#include <corsika/modules/qgsjetIII/qgsjet-III-04.hpp>

#include <corsika/framework/random/RNGManager.hpp>
#include <corsika/framework/core/Logging.hpp>

#include <iostream>
#include <random>

inline datadir::datadir(const std::string& dir) {
  if (dir.length() > 130) {
    CORSIKA_LOG_ERROR("QGSJetIII error, will cut datadir \"{}\" to 130 characters: ", {});
  }
  int i = 0;
  for (i = 0; i < std::min(130, int(dir.length())); ++i) data[i] = dir[i];
  data[i + 0] = ' ';
  data[i + 1] = '\0';
}

inline void lzmaopenfile_(const char*, int) {}
inline void lzmaclosefile_() {}
inline void lzmafillarray_(const double&, const int&) {}
