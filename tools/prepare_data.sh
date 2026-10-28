#!/usr/bin/env bash
# Keep geomagnetic resources in C8's ordinary data directory.
set -euo pipefail
c8_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -f "$c8_source/modules/data/GeoMag/IGRF13.COF" ||
      ! -f "$c8_source/modules/conex/cxroot/CMakeLists.txt" ]]; then
  git -C "$c8_source" submodule update --init --recursive
fi
c8_resource="$c8_source/resources/GeoMag/IGRF14.COF"
c8_target="$c8_source/modules/data/GeoMag/IGRF14.COF"
c8_expected=02fc7f4573ef453158eec55d2e5b7f4a25df98ca2475a20463f40b8f2bbfebcf
printf '%s  %s\n' "$c8_expected" "$c8_resource" | sha256sum --check --status
if [[ -e "$c8_target" ]]; then
  cmp --silent "$c8_resource" "$c8_target" || {
    echo "Existing IGRF14 data differ; preserve and review $c8_target" >&2; exit 1;
  }
else
  cp -- "$c8_resource" "$c8_target"
fi
echo 'C8 data prepared in modules/data, including GeoMag/IGRF14.COF.'
