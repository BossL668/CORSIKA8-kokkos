/*
 * (c) Copyright 2026 CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */

#include <chrono>
#include <thread>
int main() { std::this_thread::sleep_for(std::chrono::seconds(30)); }
