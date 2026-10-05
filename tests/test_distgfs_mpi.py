"""Runs the controller loop under MPI, with one controller and two workers.

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
"""

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter

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


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mode", ["all", "per_problem"])
def test_mpi_controller_loop(tmp_path, launcher, mode):
    fpath = str(tmp_path / "mpi.h5")
    env = dict(os.environ)
    # The repository copy of distgfs must win over any installed copy, on
    # the controller and on the workers.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO_DIR, TESTS_DIR, env.get("PYTHONPATH")) if p
    )
    proc = subprocess.run(
        launcher + ["-n", "3", sys.executable, DRIVER, mode, fpath],
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    with h5py.File(fpath, "r") as f:
        for problem_id in driver.PROBLEM_IDS:
            objectives = f["mpi_loop"][str(problem_id)]["objectives"]
            assert len(objectives) == driver.N_ITER

    with open(fpath + ".pids") as f:
        calls = Counter(tuple(line.split()) for line in f)
    if mode == "all":
        expected = [("0,1,2", str(i)) for i in range(driver.N_ITER)]
    else:
        expected = [
            (str(problem_id), str(i))
            for problem_id in driver.PROBLEM_IDS
            for i in range(driver.N_ITER)
        ]
    assert calls == Counter(expected)
