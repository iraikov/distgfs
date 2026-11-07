"""Small distgfs optimization run under MPI by tests/test_distgfs_mpi.py.

Usage: mpiexec -n <ranks> python mpi_distgfs_driver.py <h5 path> [options]

Three problems are optimized for N_ITER evaluations each, with two tasks
in flight, and a checkpoint is written to the given path.  Every task
appends a line "<problem ids> <pid>" to <h5 path>.pids, so that the test
can check the pid values the objective received.  Every rank writes the
value distgfs.run() returned to <h5 path>.rank<world rank>.

Options:

- --mode all|per_problem: one task for all problems, or one per problem;
- --spawn: spawn the worker groups instead of splitting the world;
- --nprocs-per-worker N: processes in each worker group; every process
  evaluates the objective, and mean_reduce() combines their results;
- --max-workers N: limit the number of worker ranks;
- --fail-pid N: the objective raises on pid N.  The controller catches the
  exception and writes "<type> <message>" to <h5 path>.error;
- --obj-fun-init: create the objective with obj_fun_init_name instead of
  naming it with obj_fun_name.
"""

import argparse
import sys

import distgfs

N_ITER = 8
PROBLEM_IDS = {0, 1, 2}


class ObjectiveError(Exception):
    pass


def make_obj_fun(fpath, fail_pid):
    def obj_fun(pp, pid):
        with open(fpath + ".pids", "a") as f:
            f.write(f"{','.join(map(str, sorted(pp)))} {pid}\n")
        if pid == fail_pid:
            raise ObjectiveError(f"objective failed at pid {pid}")
        return {
            problem_id: -((params["x"] - 0.3) ** 2) for problem_id, params in pp.items()
        }

    return obj_fun


def obj_fun_init(fpath, fail_pid=None, worker=None):
    """Objective factory named by obj_fun_init_name."""
    return make_obj_fun(fpath, fail_pid)


def mean_reduce(values):
    """Average the results of the processes of a worker group."""
    return {
        problem_id: sum(value[problem_id] for value in values) / len(values)
        for problem_id in values[0]
    }


def obj_fun(pp, pid):
    """Objective named by obj_fun_name; reads its settings from argv."""
    args = parse_args()
    return make_obj_fun(args.fpath, args.fail_pid)(pp, pid)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath")
    parser.add_argument("--mode", choices=["all", "per_problem"], default="all")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--nprocs-per-worker", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=-1)
    parser.add_argument("--fail-pid", type=int, default=None)
    parser.add_argument("--obj-fun-init", action="store_true")
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def main():
    from mpi4py import MPI

    # The objective runs from this file imported as a module, not as
    # __main__, so its exception class is the module's.
    from mpi_distgfs_driver import ObjectiveError

    args = parse_args()
    params = {
        "opt_id": "mpi_loop",
        "problem_parameters": {"y": 0.1},
        "space": {"x": [0.1, 0.7]},
        "problem_ids": PROBLEM_IDS,
        "per_problem_tasks": args.mode == "per_problem",
        "n_iter": N_ITER,
        "n_max_tasks": 2,
        "file_path": args.fpath,
        "save": True,
        "save_iter": 3,
    }
    if args.obj_fun_init:
        params["obj_fun_init_name"] = "mpi_distgfs_driver.obj_fun_init"
        params["obj_fun_init_args"] = {
            "fpath": args.fpath,
            "fail_pid": args.fail_pid,
        }
    else:
        params["obj_fun_name"] = "mpi_distgfs_driver.obj_fun"
    if args.nprocs_per_worker > 1:
        params["reduce_fun_name"] = "mpi_distgfs_driver.mean_reduce"
    try:
        best = distgfs.run(
            params,
            spawn_workers=args.spawn,
            nprocs_per_worker=args.nprocs_per_worker,
            max_workers=args.max_workers,
        )
    except ObjectiveError as e:
        with open(args.fpath + ".error", "w") as f:
            f.write(f"{type(e).__name__} {e}\n")
        best = "error"
    with open(f"{args.fpath}.rank{MPI.COMM_WORLD.rank}", "w") as f:
        f.write(repr(best))


if __name__ == "__main__":
    main()
