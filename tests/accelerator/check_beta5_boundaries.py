"""Source-level regression guard for the standalone Kokkos product."""
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
errors = []
for removed in ['src/gpu', 'corsika/gpu', 'gpu_em_tables',
                'applications/gpu_em_tablegen.cpp',
                'applications/gpu_em_table_prepare.cpp',
                'corsika/accelerator/em/NativeCudaBackendAdapter.hpp']:
    if (root / removed).exists(): errors.append(f'obsolete component: {removed}')
for directory in ['corsika/accelerator', 'src/accelerator', 'applications']:
    for path in (root / directory).rglob('*'):
        if path.suffix not in ('.cpp', '.hpp', '.inl', '.cu'): continue
        text = path.read_text()
        if path.suffix == '.cu': errors.append(f'native kernel TU: {path}')
        for token in ['class CudaEmBackend', 'readRateTable(', 'writeRateTable(',
                      'flattenRateTable(', '#include <corsika/gpu/', '__global__']:
            if token in text: errors.append(f'{path}: {token}')
        if '/common/' in str(path):
            if re.search(r'#include\s*[<"](?:cuda|cub/|hip/|sycl/)', text):
                errors.append(f'device runtime in shared physics: {path}')
        for include in re.findall(r'#include\s*[<"](corsika/accelerator/[^>"]+)', text):
            if not (root / include).exists(): errors.append(f'missing {include}')
for flag in ['"--gpu-table-cache"', '"--gpu-table-tolerance"']:
    if flag in (root / 'applications/c8_air_shower.cpp').read_text():
        errors.append(f'removed CLI still accepted: {flag}')
if errors:
    raise SystemExit('\n'.join(errors))
print('PASS: Kokkos-only product, native-only physics queries, isolated runtimes.')
