import math
import os
import tempfile

import h5py
import distgfs


def levi(x, y):
    """
    Levi's function (see https://en.wikipedia.org/wiki/Test_functions_for_optimization).
    Has a global _minimum_ of 0 at x=1, y=1.
    """
    a = math.sin(3.0 * math.pi * x) ** 2
    b = (x - 1) ** 2 * (1 + math.sin(3.0 * math.pi * y) ** 2)
    c = (y - 1) ** 2 * (1 + math.sin(2.0 * math.pi * y) ** 2)
    return a + b + c


def obj_fun(pp, pid):
    """Objective function to be _maximized_ by GFS."""
    x = pp["x"]
    y = pp["y"]

    res = levi(0.4 * x, y)
    # Since Dlib maximizes, but we want to find the minimum,
    # we negate the result before passing it to the Dlib optimizer.
    return -res


def obj_fun_multi_str(pp, pid):
    """Multi-problem objective with string problem ids; returns {id: float}."""
    results = {}
    for problem_id, params in pp.items():
        x = params["x"]
        y = params["y"]
        results[problem_id] = -levi(0.4 * x, y)
    return results


def obj_fun_multi_int(pp, pid):
    """Multi-problem objective with int problem ids; iterates only over pp keys."""
    results = {}
    for problem_id, params in pp.items():
        x = params["x"]
        y = params["y"]
        results[problem_id] = -levi(0.4 * x, y)
    return results


def test_basic():
    # For this example, we pretend that we want to keep 'y' fixed at 1.0
    # while optimizing 'x' in the range -4.5 to 4.5
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}

    # Create an optimizer parameter set
    distgfs_params = {
        "opt_id": "distgfs_levi",
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "test_distgfs",
        "problem_parameters": problem_parameters,
        "space": space,
        "n_iter": 50,
        "n_max_tasks": 1,
    }

    params, val = distgfs.run(distgfs_params, verbose=True)
    params_dict = dict(params)
    assert math.isclose(params_dict["x"], 1.0 / 0.4, rel_tol=1e-3)
    assert math.isclose(val, 0.0, abs_tol=1e-5)


def test_per_problem_tasks_levi():
    """per_problem_tasks=True dispatches one task per problem; both ids must converge."""
    distgfs_params = {
        "opt_id": "distgfs_levi_pp",
        "obj_fun_name": "obj_fun_multi_int",
        "obj_fun_module": "test_distgfs",
        "problem_parameters": {"y": 1.0},
        "space": {"x": [-4.5, 4.5]},
        "problem_ids": {0, 1},
        "per_problem_tasks": True,
        "n_iter": 20,
        "n_max_tasks": 1,
    }
    best = distgfs.run(distgfs_params, verbose=False)
    assert isinstance(best, dict)
    assert set(best.keys()) == {0, 1}
    for pid in (0, 1):
        prms, val = best[pid]
        assert math.isclose(val, 0.0, abs_tol=1e-3)


def test_per_problem_tasks_checkpoint():
    """per_problem_tasks checkpoint must record both problem groups in the H5 file."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        fpath = tmp.name
    os.unlink(fpath)

    try:
        distgfs_params = {
            "opt_id": "distgfs_levi_pp_ckpt",
            "obj_fun_name": "obj_fun_multi_int",
            "obj_fun_module": "test_distgfs",
            "problem_parameters": {"y": 1.0},
            "space": {"x": [-4.5, 4.5]},
            "problem_ids": {0, 1},
            "per_problem_tasks": True,
            "n_iter": 10,
            "n_max_tasks": 1,
            "file_path": fpath,
            "save": True,
            "save_iter": 5,
        }
        distgfs.run(distgfs_params, verbose=False)

        with h5py.File(fpath, "r") as f:
            grp = f["distgfs_levi_pp_ckpt"]
            assert "problem_ids" in grp
            for pid in (0, 1):
                assert str(pid) in grp
                assert len(grp[str(pid)]["objectives"]) == 10
    finally:
        if os.path.exists(fpath):
            os.unlink(fpath)


def test_per_problem_tasks_balanced():
    """All K problems must receive exactly n_iter evaluations (round-robin fairness)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        fpath = tmp.name
    os.unlink(fpath)

    n_iter = 10
    problem_ids = {0, 1, 2, 3}
    try:
        distgfs_params = {
            "opt_id": "distgfs_levi_pp_bal",
            "obj_fun_name": "obj_fun_multi_int",
            "obj_fun_module": "test_distgfs",
            "problem_parameters": {"y": 1.0},
            "space": {"x": [-4.5, 4.5]},
            "problem_ids": problem_ids,
            "per_problem_tasks": True,
            "n_iter": n_iter,
            "n_max_tasks": 1,
            "file_path": fpath,
            "save": True,
            "save_iter": n_iter,
        }
        distgfs.run(distgfs_params, verbose=False)

        with h5py.File(fpath, "r") as f:
            grp = f["distgfs_levi_pp_bal"]
            for pid in problem_ids:
                assert len(grp[str(pid)]["objectives"]) == n_iter
    finally:
        if os.path.exists(fpath):
            os.unlink(fpath)


def test_string_problem_ids():
    """String problem_ids must be stored and loaded correctly via HDF5."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        fpath = tmp.name
    os.unlink(fpath)

    try:
        distgfs_params = {
            "opt_id": "test_str_pids",
            "obj_fun_name": "obj_fun_multi_str",
            "obj_fun_module": "test_distgfs",
            "problem_parameters": {"y": 1.0},
            "space": {"x": [-4.5, 4.5]},
            "problem_ids": {"A", "B"},
            "n_iter": 10,
            "n_max_tasks": 1,
            "file_path": fpath,
            "save": True,
            "save_iter": 5,
        }
        distgfs.run(distgfs_params, verbose=False)

        with h5py.File(fpath, "r") as f:
            opt_grp = f["test_str_pids"]
            assert "problem_ids" in opt_grp
            raw = opt_grp["problem_ids"][:]
            stored = {
                pid.decode("utf-8") if isinstance(pid, bytes) else str(pid)
                for pid in raw
            }
            assert stored == {"A", "B"}
            assert "A" in opt_grp
            assert "B" in opt_grp
    finally:
        if os.path.exists(fpath):
            os.unlink(fpath)
