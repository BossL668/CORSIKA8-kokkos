"""Host-side batch supervision; never changes transport, seeds or acceptance.

An idle process is only a *shutdown* timeout once every expected shower has
closed its timing callback. That is not proof of complete output: the caller
must still require a normal exit and validate all files before accepting it.
"""
import json
import os
from pathlib import Path
import tempfile


class BatchTimeout(RuntimeError):
    """Operational timeout; distinct from memory, physics and integrity errors."""


def atomic_json(path, data):
    """Concurrent writers must not share the same .tmp filename."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cpu_ticks(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        return None


class ShutdownWatchdog:
    def __init__(self, idle_seconds=120.):
        if idle_seconds <= 0:
            raise ValueError('shutdown idle timeout must be positive')
        self.idle_seconds = idle_seconds
        self.previous = None
        self.last_progress = None

    def stalled(self, now, showers_closed, ticks, output_signature):
        if not showers_closed or ticks is None:
            self.previous = self.last_progress = None
            return False
        current = (ticks, output_signature)
        if self.last_progress is None or current != self.previous:
            self.previous = current
            self.last_progress = now
        return now - self.last_progress >= self.idle_seconds


def proc_diagnostic(pid, output):
    """Best-effort procfs snapshot before killing our own launched process.

Kernel stack/syscall access can be restricted; record that rather than
weakening ptrace policy. These are not substitutes for a C++ GDB backtrace.
"""
    result = {'pid': pid, 'cpu_ticks': cpu_ticks(pid), 'threads': {},
              'root_summary_exists': (Path(output)/'summary.yaml').is_file()}
    for task in (Path('/proc')/str(pid)/'task').glob('*'):
        state = {}
        for name in ('stat', 'wchan', 'syscall', 'stack'):
            try:
                state[name] = (task/name).read_text()
            except OSError as error:
                state[name] = {'unavailable': str(error)}
        result['threads'][task.name] = state
    return result
