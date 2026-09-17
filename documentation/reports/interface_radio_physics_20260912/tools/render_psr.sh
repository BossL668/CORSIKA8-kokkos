#!/usr/bin/env bash
set -eo pipefail
c8_report=$(cd "$(dirname "$0")/.." && pwd)
cd "$c8_report"
chmod +x tools/chromium_cpu.sh
taskset -c 0-255 tools/marp REPORT_SLIDES_CN.md --html --output REPORT_SLIDES_CN.html > data/render-html.log 2>&1
for c8_format in pdf pptx; do
 taskset -c 0-255 tools/marp REPORT_SLIDES_CN.md --html --"$c8_format" --allow-local-files --browser-path "$c8_report/tools/chromium_cpu.sh" --browser-timeout 60 --output "REPORT_SLIDES_CN.$c8_format" > "data/render-$c8_format.log" 2>&1
done
