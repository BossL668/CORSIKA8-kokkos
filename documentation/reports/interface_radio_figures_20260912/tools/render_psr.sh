#!/usr/bin/env bash
set -eo pipefail
c8_ray_root=$(cd "$(dirname "$0")/.." && pwd)
c8_ray_old="$(dirname "$c8_ray_root")/interface-radio-report-20260912"
cd "$c8_ray_root"
for c8_ray_format in html pdf pptx; do
 if [ "$c8_ray_format" = html ]; then
  taskset -c 0-255 "$c8_ray_old/tools/marp" FIGURES_CN.md --html --output FIGURES_CN.html > data/render-html.log 2>&1
 else
  taskset -c 0-255 "$c8_ray_old/tools/marp" FIGURES_CN.md --html --"$c8_ray_format" --allow-local-files --browser-path "$c8_ray_old/tools/chromium_cpu.sh" --browser-timeout 60 --output "FIGURES_CN.$c8_ray_format" > "data/render-$c8_ray_format.log" 2>&1
 fi
done

