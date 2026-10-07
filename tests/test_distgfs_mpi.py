"""Runs distgfs under MPI, by default with one controller and two workers.

Each test starts tests/mpi_distgfs_driver.py with an MPI launcher in a
subprocess.  The launcher is the first of these that starts a real
two-process world for this interpreter's mpi4py:

- the command in the environment variable DISTGFS_TEST_MPIEXEC, if set;
- /usr/bin/mpirun.openmpi -x PYTHONPATH;
- mpiexec.

A launcher that starts each process as a separate one-process world is
passed over.  This happens with the MPICH launcher on a system whose MPICH
library expects PMIx, which that launcher does not provide.  If no
launcher works, the tests are skipped.

MPI4PY_MPIABI is deliberately left alone: forcing mpi4py onto OpenMPI
crashes processes that also load an MPICH-linked library such as NEURON.
The spawn test is the exception.  The system MPICH may be unable to spawn
processes, so that test runs mpi4py on Open MPI (MPI4PY_MPIABI=openmpi)
under /usr/bin/mpirun.openmpi, and is skipped if spawning does not work
there.
"""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter

import ast

import h5py
import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
DRIVER = os.path.join(TESTS_DIR, "mpi_distgfs_driver.py")

sys.path.insert(0, TESTS_DIR)
import mpi_distgfs_driver as driver  # noqa: E402

PROBE = "from mpi4py import MPI; print(MPI.COMM_WORLD.size)"


def find_launcher():
    """Return the command of a working MPI launcher, or None."""
    candidates = []
    override = os.environ.get("DISTGFS_TEST_MPIEXEC")
    if override:
        candidates.append(shlex.split(override))
    if os.path.exists("/usr/bin/mpirun.openmpi"):
        candidates.append(["/usr/bin/mpirun.openmpi", "-x", "PYTHONPATH"])
    mpiexec = shutil.which("mpiexec")
    if mpiexec is not None:
        candidates.append([mpiexec])
    for command in candidates:
        try:
            probe = subprocess.run(
                command + ["-n", "2", sys.executable, "-c", PROBE],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0 and probe.stdout.split() == ["2", "2"]:
            return command
    return None


@pytest.fixture(scope="module")
def launcher():
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("needs mpi4py")
    found = find_launcher()
    if found is None:
        pytest.skip("no MPI launcher starts a two-process world for mpi4py")
    return found


SPAWN_LAUNCHER = [
    "/usr/bin/mpirun.openmpi",
    "--use-hwthread-cpus",
    "--oversubscribe",
    "--bind-to",
    "none",
    "-x",
    "PYTHONPATH",
    "-x",
    "MPI4PY_MPIABI",
]

SPAWN_PROBE = (
    "import sys; from mpi4py import MPI; "
    "c = MPI.COMM_SELF.Spawn(sys.executable, ['-c', "
    "'from mpi4py import MPI; MPI.Comm.Get_parent().Disconnect()'], maxprocs=1); "
    "c.Disconnect(); print('spawned')"
)


def driver_env(**extra):
    """Environment for the driver: the repository copy of distgfs must win
    over any installed copy, on every rank and in spawned processes."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO_DIR, TESTS_DIR, env.get("PYTHONPATH")) if p
    )
    env.update(extra)
    return env


def run_driver(launcher, fpath, n, *args, env=None):
    proc = subprocess.run(
        launcher + ["-n", str(n), sys.executable, DRIVER, fpath, *args],
        env=env or driver_env(),
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def rank_results(fpath, n):
    """The value distgfs.run() returned on each of the n ranks."""
    results = []
    for rank in range(n):
        with open(f"{fpath}.rank{rank}") as f:
            results.append(ast.literal_eval(f.read()))
    return results


def objective_counts(fpath):
    with h5py.File(fpath, "r") as f:
        return {
            problem_id: len(f["mpi_loop"][str(problem_id)]["objectives"])
            for problem_id in driver.PROBLEM_IDS
        }


def pid_calls(fpath):
    with open(fpath + ".pids") as f:
        return Counter(tuple(line.split()) for line in f)


def expected_calls(mode, repeat=1):
    if mode == "all":
        expected = [("0,1,2", str(i)) for i in range(driver.N_ITER)]
    else:
        expected = [
            (str(problem_id), str(i))
            for problem_id in driver.PROBLEM_IDS
            for i in range(driver.N_ITER)
        ]
    return Counter(expected * repeat)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mode", ["all", "per_problem"])
def test_mpi_controller_loop(tmp_path, launcher, mode):
    """Each problem gets N_ITER evaluations with distinct pids, and every
    rank returns the controller's best result."""
    fpath = str(tmp_path / "mpi.h5")
    run_driver(launcher, fpath, 3, "--mode", mode)
    assert objective_counts(fpath) == {p: driver.N_ITER for p in driver.PROBLEM_IDS}
    assert pid_calls(fpath) == expected_calls(mode)
    results = rank_results(fpath, 3)
    assert set(results[0]) == driver.PROBLEM_IDS
    assert results[1] == results[0] and results[2] == results[0]


@pytest.mark.timeout(180)
def test_mpi_max_workers(tmp_path, launcher):
    """With max_workers=2 of three worker ranks, the work still completes,
    and the rank left out returns None."""
    fpath = str(tmp_path / "mpi.h5")
    run_driver(launcher, fpath, 4, "--max-workers", "2")
    assert objective_counts(fpath) == {p: driver.N_ITER for p in driver.PROBLEM_IDS}
    results = rank_results(fpath, 4)
    assert results.count(None) == 1
    best = [r for r in results if r is not None]
    assert len(best) == 3 and all(r == best[0] for r in best)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mode", ["all", "per_problem"])
def test_mpi_objective_failure(tmp_path, launcher, mode):
    """An objective that raises: the controller gets the objective's own
    exception after a checkpoint is written, the workers return None, and
    the job ends."""
    fpath = str(tmp_path / "mpi.h5")
    run_driver(launcher, fpath, 3, "--mode", mode, "--fail-pid", "3")
    with open(fpath + ".error") as f:
        assert f.read().strip() == "ObjectiveError objective failed at pid 3"
    counts = objective_counts(fpath)
    assert all(n <= driver.N_ITER - 1 for n in counts.values())
    assert any(n >= 1 for n in counts.values())
    assert rank_results(fpath, 3) == ["error", None, None]


@pytest.fixture(scope="module")
def spawn_launcher():
    if not os.path.exists(SPAWN_LAUNCHER[0]):
        pytest.skip("needs /usr/bin/mpirun.openmpi")
    try:
        probe = subprocess.run(
            SPAWN_LAUNCHER + ["-n", "1", sys.executable, "-c", SPAWN_PROBE],
            env=driver_env(MPI4PY_MPIABI="openmpi"),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("the spawn probe did not finish")
    if probe.returncode != 0 or "spawned" not in probe.stdout:
        pytest.skip("mpi4py on Open MPI cannot spawn processes here")
    return SPAWN_LAUNCHER


@pytest.mark.timeout(240)
def test_mpi_spawn(tmp_path, spawn_launcher):
    """Two brokers each spawn a group of two processes, whose objective is
    made by obj_fun_init_name; both processes of a group evaluate every
    task, and mean_reduce combines their results."""
    fpath = str(tmp_path / "mpi.h5")
    run_driver(
        spawn_launcher,
        fpath,
        3,
        "--spawn",
        "--nprocs-per-worker",
        "2",
        "--obj-fun-init",
        env=driver_env(MPI4PY_MPIABI="openmpi"),
    )
    assert objective_counts(fpath) == {p: driver.N_ITER for p in driver.PROBLEM_IDS}
    assert pid_calls(fpath) == expected_calls("all", repeat=2)
    results = rank_results(fpath, 3)
    assert set(results[0]) == driver.PROBLEM_IDS
    assert results[1] == results[0] and results[2] == results[0]
