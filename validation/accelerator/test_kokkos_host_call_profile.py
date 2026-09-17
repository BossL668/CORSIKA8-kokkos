#!/usr/bin/env python3
"""Callback ABI/aggregation tests. Pass a prebuilt diagnostic .so; no GPU use."""
import ctypes as c
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


class Settings(c.Structure):
    _fields_ = [('requires_global_fencing', c.c_bool), ('padding', c.c_bool * 255)]


class Space(c.Structure):
    _fields_ = [('name', c.c_char * 64)]


def child(library, output, case):
    os.environ['C8_KOKKOS_HOST_PROFILE'] = str(output)
    os.environ['C8_KOKKOS_HOST_PROFILE_SAMPLE_LOG2'] = '6' if case == 'sampled' else ('bad' if case == 'sample_bad' else '0')
    lib = c.CDLL(str(library))
    lib.kokkosp_request_tool_settings.argtypes = [c.c_uint32, c.POINTER(Settings)]
    settings = Settings(True)
    lib.kokkosp_request_tool_settings(1, c.byref(settings))
    assert settings.requires_global_fencing is False
    lib.kokkosp_init_library.argtypes = [c.c_int, c.c_uint64, c.c_uint32, c.c_void_p]
    lib.kokkosp_init_library(0, 20240906 if case != 'version' else 0, 0, None)
    for kind in ('parallel_for', 'parallel_reduce', 'parallel_scan', 'fence'):
        getattr(lib, 'kokkosp_begin_' + kind).argtypes = [c.c_char_p, c.c_uint32, c.POINTER(c.c_uint64)]
        getattr(lib, 'kokkosp_end_' + kind).argtypes = [c.c_uint64]
    lib.kokkosp_begin_deep_copy.argtypes = [Space, c.c_char_p, c.c_void_p, Space, c.c_char_p, c.c_void_p, c.c_uint64]
    lib.kokkosp_allocate_data.argtypes = [Space, c.c_char_p, c.c_void_p, c.c_uint64]
    lib.kokkosp_deallocate_data.argtypes = lib.kokkosp_allocate_data.argtypes

    def begin(name=b'kernel', device=1 << 24):
        token = c.c_uint64()
        lib.kokkosp_begin_parallel_for(name, device, c.byref(token))
        return token

    if case in ('normal', 'overwrite'):
        # Independent host callers; each ID may complete out of submission order.
        def worker(device):
            for _ in range(100):
                first = begin(b'kernel "with\\escape\n', device)
                second = begin(b'nested', device)
                lib.kokkosp_end_parallel_for(second)
                lib.kokkosp_end_parallel_for(first)
        threads = [threading.Thread(target=worker, args=(kind << 24,)) for kind in (1, 2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        copy_args = (Space(b'HostSpace'), b'control', None, Space(b'HostSpace'), b'control', None)
        lib.kokkosp_begin_deep_copy(*copy_args, 16)
        lib.kokkosp_begin_deep_copy(*copy_args, 32)
        lib.kokkosp_end_deep_copy()
        lib.kokkosp_end_deep_copy()
        lib.kokkosp_allocate_data(Space(b'HostSpace'), b'workspace', None, 1234)
        lib.kokkosp_deallocate_data(Space(b'HostSpace'), b'workspace', None, 1234)
        token = c.c_uint64()
        lib.kokkosp_begin_fence(b'fixture fence', 1 << 24, c.byref(token))
        time.sleep(.002)
        lib.kokkosp_end_fence(token)
    elif case == 'unmatched':
        token = begin()
        lib.kokkosp_end_parallel_for(token)
        lib.kokkosp_end_parallel_for(token)  # must not double count
    elif case == 'unfinished':
        begin()
    elif case == 'row_limit':
        for i in range(2100):
            lib.kokkosp_end_parallel_for(begin(('kernel-' + str(i)).encode()))
    elif case == 'active_limit':
        tokens = [begin() for _ in range(1050)]
        for token in tokens: lib.kokkosp_end_parallel_for(token)
    elif case == 'copy_limit':
        for _ in range(70):
            lib.kokkosp_begin_deep_copy(Space(b'HostSpace'), b'a', None, Space(b'HostSpace'), b'b', None, 4)
        for _ in range(70): lib.kokkosp_end_deep_copy()
    elif case == 'sampled':
        for _ in range(10000):
            lib.kokkosp_end_parallel_for(begin())
        lib.kokkosp_allocate_data(Space(b'HostSpace'), b'exact allocation', None, 1234)
    lib.kokkosp_finalize_library()


def main():
    library = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        child(library, Path(sys.argv[2]), sys.argv[3])
        return
    with tempfile.TemporaryDirectory(prefix='c8-host-profile-abi-') as directory:
        for case in ('normal', 'unmatched', 'unfinished', 'row_limit', 'active_limit', 'copy_limit', 'version', 'overwrite', 'sampled', 'sample_bad'):
            output = Path(directory) / (case + '.json')
            if case == 'overwrite': output.write_text('KEEP EXISTING\n')
            subprocess.run([sys.executable, __file__, str(library), str(output), case], check=True, timeout=20)
            if case == 'overwrite':
                assert output.read_text() == 'KEEP EXISTING\n'
                continue
            data = json.loads(output.read_text())
            assert data['requires_global_fencing'] is False
            assert data['settings_callback_seen'] is True
            assert data['complete'] == (case in ('normal', 'sampled')), (case, data)
            assert len(data['rows']) <= 2048
            if case == 'normal':
                rows = data['rows']
                assert sum(r['calls'] for r in rows if r['kind'] == 'parallel_for') == 400
                assert {r['execution_type'] for r in rows if r['kind'] == 'parallel_for'} == {'OpenMP', 'Cuda'}
                assert sum(r['bytes'] for r in rows if r['kind'] == 'deep_copy') == 48
                assert next(r for r in rows if r['kind'] == 'fence')['host_inclusive_s'] >= .001
                assert data['active_at_finalize'] == data['errors'] == 0
            elif case == 'sampled':
                assert data['timing_sample_probability'] == 1/64
                selected = sum(r['calls'] for r in data['rows'] if r['kind'] == 'parallel_for')
                assert 100 < selected < 220, selected
                assert next(r for r in data['rows'] if r['kind'] == 'allocate')['bytes'] == 1234
            elif case != 'unfinished':
                assert data['errors'] > 0
            print('PASS', case)


if __name__ == '__main__':
    main()
