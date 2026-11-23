"""Live progress of a long analysis, as a file in its own output directory.

A Sobol run, an emulator training and a profile likelihood each write ``progress.json``
(:data:`PROGRESS_FILE`) into the directory their results go to, so a front end (CUFLynx) or a
person can see how far a run is without reading its log::

    {"schema_version": 1, "kind": "sobol", "state": "running",
     "started": "2026-11-23T02:10:00+00:00", "updated": "2026-11-23T02:10:41+00:00",
     "phases": [{"name": "sampling", "state": "running", "done": 1200, "total": 4608,
                 "failed": 3}]}

``schemas/progress.schema.json`` is its definition. ``kind`` is one of :data:`KINDS`, ``state``
and each phase's ``state`` one of ``running`` / ``done`` / ``failed`` (a phase also starts
``queued``). A phase counts ``done`` of ``total`` items, ``failed`` of them among the done, and
may carry fields of its kind (an emulation's ``training`` phase lists its ``models``).

:class:`ProgressFile` writes it: on rank 0 only, atomically (a reader never sees half a file),
and at most once every ``min_interval`` seconds unless the write is forced, which every change
of state is. Writing never stops a run: an ``OSError`` is warned about once and the run goes on.

:class:`RankReports` brings the other MPI ranks' counts to rank 0. The analyses split their
items over the ranks in blocks and gather once at the end, so rank 0 would otherwise know only
its own block: each rank reports the items it finishes, a worker sends them to rank 0 about
once a second, and rank 0 receives them between its own items.
"""

import datetime
import json
import os
import time
import warnings

import numpy as np

PROGRESS_FILE = 'progress.json'
SCHEMA_VERSION = 1
#: the analyses that write a progress file
KINDS = ('sobol', 'emulation', 'profile_likelihood')
STATES = ('running', 'done', 'failed')
PHASE_STATES = ('queued',) + STATES
#: seconds between two unforced writes, and between a worker's reports to rank 0
MIN_INTERVAL = 1.0
#: MPI tag of a worker's report (RankReports); nothing else in libcuflynx sends point-to-point
#: messages while an analysis' items run
REPORT_TAG = 7301


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def write_atomic(path, write):
    """Write ``path`` through ``write(tmp_path)`` and a rename, so a reader sees the previous
    file or the new one, never a partial one. The temporary file keeps ``path``'s extension
    (numpy appends ``.npy`` to a name without it)."""
    stem, ext = os.path.splitext(path)
    tmp = f'{stem}.tmp{os.getpid()}{ext}'
    write(tmp)
    os.replace(tmp, path)


def write_json_atomic(path, document):
    def write(tmp):
        with open(tmp, 'w') as f:
            json.dump(document, f, indent=1)
    write_atomic(path, write)


def _is_rank0(comm):
    return comm is None or comm.Get_rank() == 0


class ProgressFile:
    """The ``progress.json`` of one run of an analysis of ``kind`` in ``directory``.

    Phases are made with :meth:`phase` and moved with :meth:`update` / :meth:`advance`;
    :meth:`finish` closes the run. Used as a context manager, leaving the block finishes it:
    ``done``, or ``failed`` when an exception leaves it. Every method does nothing on a rank
    other than 0, and when ``directory`` is None."""

    def __init__(self, directory, kind, comm=None, min_interval=MIN_INTERVAL):
        if kind not in KINDS:
            raise ValueError(f'progress kind {kind!r} is not one of {list(KINDS)}')
        self.active = directory is not None and _is_rank0(comm)
        self.path = os.path.join(directory, PROGRESS_FILE) if directory is not None else None
        self.min_interval = min_interval
        self._phases = {}
        self._last_write = None
        self._warned = False
        self.document = {'schema_version': SCHEMA_VERSION, 'kind': kind, 'state': 'running',
                         'started': _now(), 'updated': _now(), 'phases': []}
        if self.active:
            os.makedirs(directory, exist_ok=True)
            self.write(force=True)

    def phase(self, key, total, name=None, state='queued', **fields):
        """Add the phase ``key`` (its ``name`` unless one is given: two phases may share a
        name, as an emulator's sampling stages do) of ``total`` items; returns its dict."""
        entry = {'name': name or key, 'state': state, 'done': 0, 'total': int(total), 'failed': 0,
                 **fields}
        self._phases[key] = entry
        self.document['phases'].append(entry)
        self.write(force=True)
        return entry

    def update(self, key, done=None, failed=None, state=None, force=False, **fields):
        """Set fields of phase ``key``; returns whether the file was written. A phase that
        reaches its total is ``done``, one that has moved is ``running``."""
        entry = self._phases[key]
        if done is not None:
            entry['done'] = int(done)
        if failed is not None:
            entry['failed'] = int(failed)
        entry.update(fields)
        if state is None and entry['state'] in ('queued', 'running'):
            state = 'done' if entry['done'] >= entry['total'] else 'running'
        if state is not None and state != entry['state']:
            entry['state'] = state
            force = True
        return self.write(force=force)

    def advance(self, key, done=1, failed=0, force=False):
        """Count ``done`` more items of phase ``key``, ``failed`` of them failed."""
        entry = self._phases[key]
        return self.update(key, done=entry['done'] + done, failed=entry['failed'] + failed,
                           force=force)

    def finish(self, state='done'):
        """Close the run in ``state``; a phase still running or queued in a failed run is
        failed."""
        if self.document['state'] != 'running':
            return
        self.document['state'] = state
        if state == 'failed':
            for entry in self.document['phases']:
                if entry['state'] == 'running':
                    entry['state'] = 'failed'
        self.write(force=True)

    def due(self):
        """Whether an unforced write would be made now."""
        return (self._last_write is None
                or time.monotonic() - self._last_write >= self.min_interval)

    def write(self, force=False):
        if not self.active or not (force or self.due()):
            return False
        self.document['updated'] = _now()
        try:
            write_json_atomic(self.path, self.document)
        except OSError as error:
            if not self._warned:
                self._warned = True
                warnings.warn(f'could not write {self.path} ({error}); the run goes on '
                              f'without its progress file.', stacklevel=2)
            return False
        self._last_write = time.monotonic()
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish('failed' if exc_type is not None else 'done')
        return False


class RankReports:
    """Which items each MPI rank has finished, brought to rank 0 while they run.

    Every rank calls :meth:`add` after each of its items and :meth:`close` once after its
    last one (before the gather). Rank 0's ``on_report(indices, ok)`` is called with the item
    indices (an int array) and whether each succeeded (a bool array): at once for its own
    items, and for the other ranks' as their messages arrive, which :meth:`add` and
    :meth:`close` both look for. :meth:`close` on rank 0 returns when every rank has closed,
    so ``on_report`` has then seen every item.

    A worker sends about once every ``min_interval`` seconds, without waiting for rank 0 to
    receive. With one rank, or no communicator, nothing is sent."""

    def __init__(self, comm, on_report, min_interval=MIN_INTERVAL, tag=REPORT_TAG):
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        self.size = comm.Get_size() if comm is not None else 1
        self.on_report = on_report
        self.min_interval = min_interval
        self.tag = tag
        self._indices, self._ok = [], []
        self._last_send = time.monotonic()
        self._requests = []
        self._open_ranks = self.size - 1
        self._closed = False

    def add(self, index, ok=True):
        if self.rank == 0:
            self.on_report(np.array([index], dtype=int), np.array([bool(ok)]))
            self._receive()
            return
        self._indices.append(int(index))
        self._ok.append(bool(ok))
        if time.monotonic() - self._last_send >= self.min_interval:
            self._send(final=False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.size == 1:
            return
        if self.rank != 0:
            self._send(final=True)
            for request in self._requests:
                request.wait()
            self._requests = []
            return
        while self._open_ranks:
            if not self._receive():
                time.sleep(0.01)

    def _send(self, final):
        message = (np.array(self._indices, dtype=int), np.array(self._ok, dtype=bool), final)
        self._requests = [r for r in self._requests if not r.Test()]
        self._requests.append(self.comm.isend(message, dest=0, tag=self.tag))
        self._indices, self._ok = [], []
        self._last_send = time.monotonic()

    def _receive(self):
        """Take every waiting report; returns whether there was one."""
        if self.size == 1:
            return False
        received = False
        # from any rank: mpi4py's default source
        while self._open_ranks and self.comm.iprobe(tag=self.tag):
            indices, ok, final = self.comm.recv(tag=self.tag)
            received = True
            if len(indices):
                self.on_report(indices, ok)
            if final:
                self._open_ranks -= 1
        return received
