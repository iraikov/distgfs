"""Small distgfs optimization run under MPI by tests/test_distgfs_mpi.py.

Usage: mpiexec -n 3 python mpi_distgfs_driver.py <all|per_problem> <h5 path>

Three problems are optimized for N_ITER evaluations each, with two tasks
in flight, and a checkpoint is written to the given path.  Every task
appends a line "<problem ids> <pid>" to <h5 path>.pids, so that the test
can check the pid values the objective received.
"""

import sys

import distgfs

N_ITER = 8
PROBLEM_IDS = {0, 1, 2}


def obj_fun(pp, pid):
    with open(sys.argv[2] + ".pids", "a") as f:
        f.write(f"{','.join(map(str, sorted(pp)))} {pid}\n")
    return {
        problem_id: -((params["x"] - 0.3) ** 2) for problem_id, params in pp.items()
    }


if __name__ == "__main__":
    mode, fpath = sys.argv[1], sys.argv[2]
    distgfs.run(
        {
            "opt_id": "mpi_loop",
            "obj_fun_name": "mpi_distgfs_driver.obj_fun",
            "problem_parameters": {"y": 0.1},
            "space": {"x": [0.1, 0.7]},
            "problem_ids": PROBLEM_IDS,
            "per_problem_tasks": mode == "per_problem",
            "n_iter": N_ITER,
            "n_max_tasks": 2,
            "file_path": fpath,
            "save": True,
            "save_iter": 3,
        }
    )
