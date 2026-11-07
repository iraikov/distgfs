import math
import os
import re
import shutil
import sys
import tempfile
import warnings
from types import SimpleNamespace

import distwq
import dlib
import h5py
import numpy as np
import pytest

from distwq.future import TaskController

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


RECORDED_PP = []


def obj_fun_record(pp, pid):
    """Objective that records every parameter dict it is called with."""
    RECORDED_PP.append(dict(pp))
    return -((pp["x"] - 0.3) ** 2)


def test_basic():
    # For this example, we pretend that we want to keep 'y' fixed at 1.0
    # while optimizing 'x' in the range -4.5 to 4.5
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}

    # Create an optimizer parameter set
    distgfs_params = {
        "opt_id": "distgfs_levi",
        "obj_fun_name": "test_distgfs.obj_fun",
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
        "obj_fun_name": "test_distgfs.obj_fun_multi_int",
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
            "obj_fun_name": "test_distgfs.obj_fun_multi_int",
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
            "obj_fun_name": "test_distgfs.obj_fun_multi_int",
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
            "obj_fun_name": "test_distgfs.obj_fun_multi_str",
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


def test_initial_evals_seed_fresh_run():
    """initial_evals seeds a fresh (non-resumed) run; best result never
    regresses below the seeded point, which is the true optimum here."""
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}
    seed_x = 1.0 / 0.4
    distgfs_params = {
        "opt_id": "distgfs_levi_seed",
        "obj_fun_name": "test_distgfs.obj_fun",
        "problem_parameters": problem_parameters,
        "space": space,
        "n_iter": 3,
        "n_max_tasks": 1,
        "initial_evals": {0: [dlib.function_evaluation(x=[seed_x], y=0.0)]},
    }

    params, val = distgfs.run(distgfs_params, verbose=False)
    params_dict = dict(params)
    assert math.isclose(params_dict["x"], seed_x, rel_tol=1e-3)
    assert math.isclose(val, 0.0, abs_tol=1e-5)


def test_initial_evals_merge_with_checkpoint():
    """initial_evals passed alongside a resumed checkpoint are merged with
    the resumed evaluations and are persisted on the next save (bookkeeping
    fix: seeded evals were never on disk, so they must not be excluded from
    the first save_evals() call after construction)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        fpath = tmp.name
    os.unlink(fpath)

    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}
    n_iter_first = 5
    n_iter_second = 4
    n_seeded = 2

    try:
        distgfs_params_first = {
            "opt_id": "distgfs_levi_merge",
            "obj_fun_name": "test_distgfs.obj_fun",
            "problem_parameters": problem_parameters,
            "space": space,
            "n_iter": n_iter_first,
            "n_max_tasks": 1,
            "file_path": fpath,
            "save": True,
            "save_iter": n_iter_first,
        }
        distgfs.run(distgfs_params_first, verbose=False)

        distgfs_params_second = {
            "opt_id": "distgfs_levi_merge",
            "obj_fun_name": "test_distgfs.obj_fun",
            "problem_parameters": problem_parameters,
            "space": space,
            "n_iter": n_iter_second,
            "n_max_tasks": 1,
            "file_path": fpath,
            "save": True,
            "save_iter": n_iter_second,
            "initial_evals": {
                0: [
                    dlib.function_evaluation(x=[1.0 / 0.4], y=0.0),
                    dlib.function_evaluation(x=[-2.0], y=-5.0),
                ]
            },
        }
        distgfs.run(distgfs_params_second, verbose=False)

        with h5py.File(fpath, "r") as f:
            grp = f["distgfs_levi_merge"]
            n_objectives = len(grp["0"]["objectives"])
            assert n_objectives == n_iter_first + n_seeded + n_iter_second
    finally:
        if os.path.exists(fpath):
            os.unlink(fpath)


def test_initial_evals_dimension_mismatch_raises():
    """An initial_evals entry whose x dimensionality doesn't match the
    configured space must raise ValueError, not silently misalign params.

    Constructs DistGFSOptimizer directly rather than via distgfs.run():
    distwq's controller loop catches ValueError from the controller function
    and turns it into a graceful abort() rather than propagating it, so it
    cannot be observed with pytest.raises() through the distgfs.run() path.
    """
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}

    with pytest.raises(ValueError):
        distgfs.DistGFSOptimizer(
            opt_id="distgfs_levi_bad_dim",
            obj_fun=obj_fun,
            problem_parameters=problem_parameters,
            space=space,
            n_iter=3,
            initial_evals={0: [dlib.function_evaluation(x=[1.0, 2.0], y=0.0)]},
        )


def test_initial_feature_evals_requires_feature_dtypes():
    """initial_feature_evals without feature_dtypes configured must raise,
    since there is nowhere valid to store the seeded feature vectors."""
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}

    with pytest.raises(ValueError):
        distgfs.DistGFSOptimizer(
            opt_id="distgfs_levi_bad_feat",
            obj_fun=obj_fun,
            problem_parameters=problem_parameters,
            space=space,
            n_iter=3,
            initial_evals={0: [dlib.function_evaluation(x=[1.0 / 0.4], y=0.0)]},
            initial_feature_evals={0: [(-1, [0.0])]},
        )


def test_initial_evals_unknown_problem_id_raises():
    """initial_evals keyed by a problem_id outside the configured
    problem_ids set must raise ValueError rather than silently creating an
    orphaned, never-iterated entry."""
    with pytest.raises(ValueError):
        distgfs.DistGFSOptimizer(
            opt_id="distgfs_levi_pp_bad_pid",
            obj_fun=obj_fun_multi_int,
            problem_parameters={"y": 1.0},
            space={"x": [-4.5, 4.5]},
            problem_ids={0, 1},
            per_problem_tasks=True,
            n_iter=3,
            initial_evals={2: [dlib.function_evaluation(x=[1.0 / 0.4], y=0.0)]},
        )


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
F32 = np.dtype(np.float32)
F64 = np.dtype(np.float64)


def record_params(opt_id, fpath, n_iter=6, **extra):
    """Parameter set for obj_fun_record with a space and a fixed 'y' that
    are not exactly representable in float32."""
    params = {
        "opt_id": opt_id,
        "obj_fun_name": "test_distgfs.obj_fun_record",
        "problem_parameters": {"y": 0.1},
        "space": {"x": [0.1, 0.7]},
        "n_iter": n_iter,
        "n_max_tasks": 1,
        "file_path": fpath,
        "save": True,
        "save_iter": n_iter,
    }
    params.update(extra)
    return params


def resume_params(opt_id, fpath, n_iter=3, **extra):
    """Parameter set that resumes 'opt_id' from 'fpath' with nothing but
    the file to describe the problem."""
    params = record_params(opt_id, fpath, n_iter=n_iter, **extra)
    for key in ("problem_parameters", "space"):
        if key not in extra:
            del params[key]
    return params


def assert_group_dtype(grp, dtype, problem_ids=(0,)):
    """Check that every parameter-related field in 'grp' has 'dtype'."""
    spec_dt = grp["parameter_spec"].dtype
    assert spec_dt["lower"] == dtype
    assert spec_dt["upper"] == dtype
    assert grp["problem_parameters"].dtype["value"] == dtype
    for pid in problem_ids:
        params_dt = grp[str(pid)]["parameters"].dtype
        for name in params_dt.names:
            assert params_dt[name] == dtype
    assert distgfs.h5_parameter_dtype(grp) == dtype


def stored_problem_parameters(grp):
    """Read the fixed problem parameters of 'grp' as a name -> value dict."""
    names = {
        v: k for k, v in h5py.check_enum_dtype(grp["parameter_enum"].dtype).items()
    }
    return {names[idx]: val for idx, val in grp["problem_parameters"][:]}


def test_parameter_dtype_default_float64(tmp_path):
    fpath = str(tmp_path / "ckpt.h5")
    distgfs.run(record_params("pd_default", fpath))
    with h5py.File(fpath, "r") as f:
        assert_group_dtype(f["pd_default"], F64)
    assert distgfs.gfsopt_dict["pd_default"].parameter_dtype == F64


def test_parameter_dtype_exact_round_trip(tmp_path):
    """Stored parameters equal the evaluated ones bit for bit."""
    fpath = str(tmp_path / "ckpt.h5")
    RECORDED_PP.clear()
    distgfs.run(record_params("pd_exact", fpath, n_iter=8))
    evaluated = np.sort(np.array([pp["x"] for pp in RECORDED_PP]))
    assert len(evaluated) == 8
    assert np.any(evaluated.astype(np.float32).astype(np.float64) != evaluated)
    with h5py.File(fpath, "r") as f:
        grp = f["pd_exact"]
        stored = np.sort(grp["0"]["parameters"]["x"])
        assert np.array_equal(stored, evaluated)
        assert stored_problem_parameters(grp)["y"] == 0.1


def test_parameter_dtype_exact_bounds_on_resume(tmp_path):
    fpath = str(tmp_path / "ckpt.h5")
    distgfs.run(record_params("pd_bounds", fpath))
    gfsopt = distgfs.DistGFSOptimizer(
        opt_id="pd_bounds", obj_fun=obj_fun_record, file_path=fpath
    )
    assert list(gfsopt.spec.lower) == [0.1]
    assert list(gfsopt.spec.upper) == [0.7]
    assert gfsopt.parameter_dtype == F64
    assert gfsopt.problem_parameters == {"y": 0.1}


def test_parameter_dtype_float32_opt_in(tmp_path):
    fpath = str(tmp_path / "ckpt.h5")
    distgfs.run(record_params("pd_f32", fpath, parameter_dtype="float32"))
    with h5py.File(fpath, "r") as f:
        assert_group_dtype(f["pd_f32"], F32)


@pytest.mark.parametrize("value", ["float16", "int32", "nonsense", ">f8", "<f4,<f4"])
def test_parameter_dtype_invalid_raises(value):
    with pytest.raises(ValueError, match="parameter_dtype"):
        distgfs.DistGFSOptimizer(
            opt_id="pd_invalid",
            obj_fun=obj_fun_record,
            problem_parameters={"y": 0.1},
            space={"x": [0.1, 0.7]},
            parameter_dtype=value,
        )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("float32", F32),
        ("float64", F64),
        (np.float32, F32),
        (np.dtype("f8"), F64),
        (None, F64),
    ],
)
def test_parameter_dtype_valid_values(value, expected):
    assert distgfs.validate_parameter_dtype(value) == expected


def test_parameter_dtype_resume_float32_with_default(tmp_path):
    """A float32 checkpoint resumed with the default setting keeps float32,
    warns once about the mismatch, and keeps appending."""
    fpath = str(tmp_path / "ckpt.h5")
    distgfs.run(record_params("pd_resume32", fpath, parameter_dtype="float32"))

    with pytest.warns(UserWarning, match="float32"):
        gfsopt = distgfs.DistGFSOptimizer(
            opt_id="pd_resume32", obj_fun=obj_fun_record, file_path=fpath
        )
    assert gfsopt.parameter_dtype == F32

    with pytest.warns(UserWarning, match="use a new file_path or opt_id"):
        distgfs.run(resume_params("pd_resume32", fpath, n_iter=3))
    assert distgfs.gfsopt_dict["pd_resume32"].parameter_dtype == F32
    with h5py.File(fpath, "r") as f:
        grp = f["pd_resume32"]
        assert len(grp["0"]["parameters"]) == 6 + 3
        assert_group_dtype(grp, F32)


def test_resume_v1_4_0_checkpoint(tmp_path):
    """A checkpoint written by distgfs 1.4.0 loads and resumes in float32.

    tests/data/distgfs_v1_4_0.h5 was produced by the 1.4.0 release running
    5 iterations of opt_id 'distgfs_v1_4_0' with space {"x": [0.1, 0.7]}
    and problem_parameters {"y": 0.1}. Version 1.4.0 also recorded the last
    evaluated 'x' among the fixed problem parameters.
    """
    fpath = str(tmp_path / "v140.h5")
    shutil.copy(os.path.join(DATA_DIR, "distgfs_v1_4_0.h5"), fpath)

    raw_spec, spec, evals, _, _, info, _ = distgfs.h5_load_all(fpath, "distgfs_v1_4_0")
    assert info["parameter_dtype"] == F32
    assert info["params"] == ["x"]
    assert len(evals[0]) == 5
    assert info["problem_parameters"]["y"] == float(np.float32(0.1))

    RECORDED_PP.clear()
    with pytest.warns(UserWarning, match="float32"):
        distgfs.run(resume_params("distgfs_v1_4_0", fpath, n_iter=3))
    assert all(pp["y"] == float(np.float32(0.1)) for pp in RECORDED_PP)
    with h5py.File(fpath, "r") as f:
        grp = f["distgfs_v1_4_0"]
        assert len(grp["0"]["parameters"]) == 5 + 3
        assert len(grp["0"]["objectives"]) == 5 + 3
        assert_group_dtype(grp, F32)


def test_parameter_dtype_multi_problem(tmp_path):
    fpath = str(tmp_path / "ckpt.h5")
    distgfs.run(
        {
            "opt_id": "pd_multi",
            "obj_fun_name": "test_distgfs.obj_fun_multi_int",
            "problem_parameters": {"y": 1.0},
            "space": {"x": [-4.5, 4.5]},
            "problem_ids": {0, 1},
            "per_problem_tasks": True,
            "n_iter": 4,
            "n_max_tasks": 1,
            "file_path": fpath,
            "save": True,
            "save_iter": 4,
        }
    )
    with h5py.File(fpath, "r") as f:
        assert_group_dtype(f["pd_multi"], F64, problem_ids=(0, 1))


def test_problem_parameters_caller_values_win_on_resume(tmp_path):
    """On resume the objective receives the caller's fixed parameters, not
    values rounded to the checkpoint's precision."""
    fpath = str(tmp_path / "ckpt.h5")
    distgfs.run(record_params("pp_resume", fpath, parameter_dtype="float32"))

    RECORDED_PP.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distgfs.run(
            resume_params(
                "pp_resume",
                fpath,
                problem_parameters={"y": 0.1},
                parameter_dtype="float32",
            )
        )
    assert not [w for w in caught if "problem_parameters" in str(w.message)]
    assert RECORDED_PP and all(pp["y"] == 0.1 for pp in RECORDED_PP)

    RECORDED_PP.clear()
    with pytest.warns(UserWarning, match="problem_parameters.*y: given 0.2"):
        distgfs.run(
            resume_params(
                "pp_resume",
                fpath,
                problem_parameters={"y": 0.2},
                parameter_dtype="float32",
            )
        )
    assert RECORDED_PP and all(pp["y"] == 0.2 for pp in RECORDED_PP)
    with h5py.File(fpath, "r") as f:
        stored = stored_problem_parameters(f["pp_resume"])
    assert stored == {"y": np.float32(0.1)}


def test_problem_parameters_stored_values_on_resume(tmp_path):
    """Without caller values, a float64 checkpoint supplies the exact
    stored fixed parameters."""
    fpath = str(tmp_path / "ckpt.h5")
    distgfs.run(record_params("pp_stored", fpath))
    RECORDED_PP.clear()
    distgfs.run(resume_params("pp_stored", fpath))
    assert RECORDED_PP and all(pp["y"] == 0.1 for pp in RECORDED_PP)


def test_compare_problem_parameters():
    stored = {"y": float(np.float32(0.1)), "z": 1.0, "w": float("nan"), "x": 0.0}
    given = {"y": 0.1, "w": float("nan"), "v": 2.0, "x": 5.0}
    diffs = distgfs.compare_problem_parameters(given, stored, ["x"], F32)
    assert diffs == ["not given: ['z']", "not stored: ['v']"]

    diffs = distgfs.compare_problem_parameters({"y": 0.1}, {"y": 0.1}, [], F64)
    assert diffs == []
    diffs = distgfs.compare_problem_parameters(
        {"y": 0.1}, {"y": float(np.float32(0.1))}, [], F64
    )
    assert len(diffs) == 1 and diffs[0].startswith("y:")
    diffs = distgfs.compare_problem_parameters({"y": "abc"}, {"y": 0.1}, [], F64)
    assert len(diffs) == 1 and diffs[0].startswith("y:")


def test_caller_problem_parameters_not_modified(tmp_path):
    """The caller's dict is left unchanged, and only the fixed parameters
    are stored as problem parameters."""
    fpath = str(tmp_path / "ckpt.h5")
    params = record_params("pp_unmodified", fpath)
    problem_parameters = params["problem_parameters"]
    distgfs.run(params)
    assert problem_parameters == {"y": 0.1}
    with h5py.File(fpath, "r") as f:
        assert stored_problem_parameters(f["pp_unmodified"]) == {"y": 0.1}


def test_problem_parameters_padding_rows_ignored(tmp_path):
    """Zero-filled rows after the real problem parameter rows do not
    overwrite the real values when loading."""
    fpath = str(tmp_path / "ckpt.h5")
    spec = dlib.function_spec(bound1=[0.1], bound2=[0.7], is_integer=[False])
    with h5py.File(fpath, "w") as f:
        distgfs.h5_init_types(f, "pad", None, None, ["x"], {"y": 0.1}, spec)
        grp = f["pad"]
        grp["solver_epsilon"] = 0.0005
        grp["relative_noise_magnitude"] = 0.001
        dset = grp["problem_parameters"]
        assert dset.shape == (1,)
        y_idx = h5py.check_enum_dtype(grp["parameter_enum"].dtype)["y"]
        dset.resize((2,))
        dset[1] = np.array((y_idx, 0.0), dtype=dset.dtype)
    _, _, info = distgfs.h5_load_raw(fpath, "pad")
    assert info["problem_parameters"]["y"] == 0.1
    assert info["parameter_dtype"] == F64


def test_new_opt_id_in_existing_file(tmp_path):
    """A new opt_id in an existing float32 checkpoint starts a fresh
    float64 group and leaves the old group untouched."""
    fpath = str(tmp_path / "v140.h5")
    shutil.copy(os.path.join(DATA_DIR, "distgfs_v1_4_0.h5"), fpath)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        distgfs.run(record_params("pd_new_group", fpath, n_iter=4))
    assert not [w for w in caught if "use a new file_path" in str(w.message)]
    with h5py.File(fpath, "r") as f:
        assert_group_dtype(f["pd_new_group"], F64)
        assert len(f["pd_new_group"]["0"]["parameters"]) == 4
        assert_group_dtype(f["distgfs_v1_4_0"], F32)
        assert len(f["distgfs_v1_4_0"]["0"]["parameters"]) == 5

    with pytest.raises(ValueError, match="no optimization group"):
        distgfs.DistGFSOptimizer(
            opt_id="pd_missing", obj_fun=obj_fun_record, file_path=fpath
        )


def test_h5_parameter_dtype_mixed_fields_raises(tmp_path):
    fpath = str(tmp_path / "mixed.h5")
    with h5py.File(fpath, "w") as f:
        grp = f.create_group("mixed")
        grp["parameter_space_type"] = np.dtype([("x", np.float32), ("z", np.float64)])
        with pytest.raises(ValueError, match="mixed types"):
            distgfs.h5_parameter_dtype(grp)


# ---------------------------------------------------------------------------
# Controller loop
# ---------------------------------------------------------------------------


class FakeController:
    """Stand-in for distwq.MPIController that mimics MPI timing in one process.

    It is used underneath a real distwq.future.TaskController (see
    fake_session()), so that the controller loop runs on real futures.

    Each task runs as soon as it is submitted, but its result is held back,
    as if a worker were still computing it.  Every call to process() releases
    held results according to 'pattern', a cycle of counts ("all" releases
    everything; 0 is a wakeup that brings no result, like a READY message).
    'reverse' releases the newest held results first.  With
    'release_on_submit', submit_call() releases results before dispatching,
    as distwq's submit_call() does through its own process() call.

    'wrap', if given, is applied to every result before it is held, to
    imitate the result types of distwq.  A task that raises is recorded as
    distwq records a captured failure: its result is the description from
    distwq.task_failure(), and its id is added to failed_tasks.

    A blocking process() with nothing held raises AssertionError, turning
    a hang into a test failure.
    """

    def __init__(
        self,
        pattern=("all",),
        reverse=False,
        release_on_submit=False,
        limit=None,
        wrap=None,
    ):
        self.pattern = list(pattern)
        self.wrap = wrap
        self.n_wakeups = 0
        self.reverse = reverse
        self.release_on_submit = release_on_submit
        self.limit = limit
        self.held = []
        self.ready = []
        self.count = 0
        self.inflight_at_submit = []
        self.submitted = []
        self.blocked_with_ready = 0
        self.failed_tasks = set()
        # Read by TaskController to tell whether a task is still known.
        self.waiting = {}
        self.assigned = {}
        self.result_queue = []

    @property
    def results(self):
        return {task_id for task_id, _ in self.held + self.ready}

    def _outstanding(self):
        return len(self.held) + len(self.ready)

    def _release(self):
        n = self.pattern[self.n_wakeups % len(self.pattern)]
        self.n_wakeups += 1
        n = len(self.held) if n == "all" else min(n, len(self.held))
        for _ in range(n):
            self.ready.append(self.held.pop() if self.reverse else self.held.pop(0))

    def submit_call(
        self, name_to_call, args=(), kwargs=None, module_name="__main__", time_est=1
    ):
        if self.release_on_submit:
            self._release()
        outstanding = self._outstanding()
        if self.limit is not None:
            assert outstanding < self.limit, "in-flight limit exceeded"
        self.inflight_at_submit.append(outstanding)
        fun = getattr(sys.modules[module_name], name_to_call)
        _opt_id, index, vals_dict = args
        self.submitted.append((index, tuple(vals_dict)))
        task_id = self.count
        self.count += 1
        try:
            result = fun(*args, **(kwargs or {}))
        except Exception as e:
            result = distwq.task_failure(e)
            self.failed_tasks.add(task_id)
        else:
            if self.wrap is not None:
                result = self.wrap(result)
        self.held.append((task_id, result))
        return task_id

    def process(self, limit=1000, block=False):
        if block:
            assert self.held or self.ready, "blocking with nothing outstanding"
            if self.ready:
                self.blocked_with_ready += 1
        self._release()

    def probe_all_next_results(self):
        ret = self.ready
        self.ready = []
        return ret

    def summary(self):
        return ""


def fake_session(ctl):
    """Return a real TaskController that drives the FakeController 'ctl'."""
    return TaskController(SimpleNamespace(_controller=ctl, worker_config=None))


RECORDED_CALLS = []


def obj_fun_pids(pp, pid):
    """Single-problem objective that records the 'pid' it receives."""
    RECORDED_CALLS.append((pid, None))
    return -((pp["x"] - 0.3) ** 2)


def obj_fun_multi_pids(pp, pid):
    """Multi-problem objective that records 'pid' and the problem ids."""
    RECORDED_CALLS.append((pid, tuple(pp)))
    return {
        problem_id: -((params["x"] - 0.3) ** 2) for problem_id, params in pp.items()
    }


def obj_fun_feature(pp, pid):
    """Objective whose feature is a copy of x, to check row alignment."""
    feature = np.array([(pp["x"],)], dtype=np.dtype([("xcopy", np.float64)]))
    return -((pp["x"] - 0.3) ** 2), feature


def loop_params(opt_id, n_iter, n_max_tasks=1, **extra):
    params = {
        "opt_id": opt_id,
        "obj_fun_name": "test_distgfs.obj_fun_pids",
        "problem_parameters": {"y": 0.1},
        "space": {"x": [0.1, 0.7]},
        "n_iter": n_iter,
        "n_max_tasks": n_max_tasks,
    }
    params.update(extra)
    return params


def multi_params(opt_id, n_iter, problem_ids, per_problem, n_max_tasks=1, **extra):
    extra.setdefault("obj_fun_name", "test_distgfs.obj_fun_multi_pids")
    return loop_params(
        opt_id,
        n_iter,
        n_max_tasks,
        problem_ids=problem_ids,
        per_problem_tasks=per_problem,
        **extra,
    )


def n_evals(opt_id, problem_id=0):
    optimizer = distgfs.gfsopt_dict[opt_id].optimizer_dict[problem_id]
    return len(optimizer.get_function_evaluations()[1][0])


@pytest.mark.timeout(60)
def test_loop_batch_exact_count():
    """Batch mode submits exactly n_iter tasks, and only when none are in
    flight, even when n_max_tasks does not divide n_iter."""
    ctl = FakeController(limit=2)
    distgfs.gfsctrl(
        fake_session(ctl), loop_params("loop_batch", n_iter=5, n_max_tasks=2)
    )
    assert ctl.inflight_at_submit == [0, 1, 0, 1, 0]
    assert n_evals("loop_batch") == 5
    assert ctl._outstanding() == 0


@pytest.mark.timeout(60)
def test_loop_batch_limit_larger_than_total():
    ctl = FakeController()
    distgfs.gfsctrl(
        fake_session(ctl), loop_params("loop_batch_big", n_iter=5, n_max_tasks=10)
    )
    assert ctl.inflight_at_submit == [0, 1, 2, 3, 4]
    assert n_evals("loop_batch_big") == 5


@pytest.mark.timeout(60)
def test_loop_unique_pids_all_problems():
    """Every task of a batch gets its own pid, 0 .. n_iter-1."""
    RECORDED_CALLS.clear()
    distgfs.run(loop_params("loop_pids", n_iter=7, n_max_tasks=3))
    assert sorted(pid for pid, _ in RECORDED_CALLS) == list(range(7))


@pytest.mark.timeout(60)
def test_loop_unique_pids_per_problem():
    RECORDED_CALLS.clear()
    distgfs.run(
        multi_params("loop_pids_pp", 4, {0, 1}, per_problem=True, n_max_tasks=3)
    )
    for problem_id in (0, 1):
        pids = sorted(pid for pid, ids in RECORDED_CALLS if ids == (problem_id,))
        assert pids == list(range(4))


@pytest.mark.timeout(60)
@pytest.mark.parametrize("pattern", [("all",), (0, 0, 1)])
def test_loop_save_count(tmp_path, monkeypatch, pattern):
    """Checkpoints are written once per save_iter new results plus once at
    the end, however many times the controller wakes up without a result."""
    calls = []
    original = distgfs.DistGFSOptimizer.save_evals

    def counting_save(self):
        calls.append(1)
        original(self)

    monkeypatch.setattr(distgfs.DistGFSOptimizer, "save_evals", counting_save)
    fpath = str(tmp_path / "ckpt.h5")
    ctl = FakeController(pattern=pattern)
    distgfs.gfsctrl(
        fake_session(ctl),
        loop_params("loop_save", n_iter=20, file_path=fpath, save=True, save_iter=5),
    )
    assert len(calls) == 4
    with h5py.File(fpath, "r") as f:
        assert len(f["loop_save"]["0"]["objectives"]) == 20


@pytest.mark.timeout(60)
def test_loop_results_during_submit():
    """Results that arrive while a task is being submitted are collected
    before the controller blocks again, and the work still completes."""
    ctl = FakeController(pattern=(1,), release_on_submit=True, limit=2)
    distgfs.gfsctrl(
        fake_session(ctl),
        multi_params("loop_submit", 6, {0, 1, 2}, per_problem=True, n_max_tasks=2),
    )
    assert ctl.blocked_with_ready == 0
    for problem_id in (0, 1, 2):
        assert n_evals("loop_submit", problem_id) == 6


@pytest.mark.timeout(60)
def test_loop_rolling_fairness_out_of_order():
    """Rolling mode refills free slots while other tasks are running, keeps
    within n_max_tasks, and gives every problem exactly n_iter tasks."""
    ctl = FakeController(pattern=(1,), reverse=True, limit=2)
    distgfs.gfsctrl(
        fake_session(ctl),
        multi_params("loop_fair", 4, {0, 1, 2}, per_problem=True, n_max_tasks=2),
    )
    assert max(ctl.inflight_at_submit) == 1
    assert any(n > 0 for n in ctl.inflight_at_submit)
    for problem_id in (0, 1, 2):
        assert sum(1 for _, ids in ctl.submitted if ids == (problem_id,)) == 4
        assert n_evals("loop_fair", problem_id) == 4


@pytest.mark.timeout(60)
def test_loop_round_robin_order():
    """With one slot, problems take turns instead of one problem running
    all of its tasks first."""
    ctl = FakeController()
    distgfs.gfsctrl(
        fake_session(ctl),
        multi_params("loop_rr", 3, {2, 0, 1}, per_problem=True, n_max_tasks=1),
    )
    assert [ids for _, ids in ctl.submitted] == [(0,), (1,), (2,)] * 3


@pytest.mark.timeout(60)
def test_loop_out_of_order_features(tmp_path):
    """Features stay aligned with their parameters when results arrive in
    reverse order."""
    fpath = str(tmp_path / "ckpt.h5")
    ctl = FakeController(reverse=True)
    distgfs.gfsctrl(
        fake_session(ctl),
        loop_params(
            "loop_feat",
            n_iter=6,
            n_max_tasks=3,
            obj_fun_name="test_distgfs.obj_fun_feature",
            feature_dtypes=[("xcopy", np.float64)],
            file_path=fpath,
            save=True,
            save_iter=6,
        ),
    )
    with h5py.File(fpath, "r") as f:
        grp = f["loop_feat"]["0"]
        xs = grp["parameters"]["x"][:]
        assert len(xs) == 6
        assert np.array_equal(grp["features"]["xcopy"][:], xs)


@pytest.mark.timeout(60)
def test_loop_zero_iterations_saves_seeds(tmp_path):
    fpath = str(tmp_path / "ckpt.h5")
    ctl = FakeController()
    distgfs.gfsctrl(
        fake_session(ctl),
        loop_params(
            "loop_zero",
            n_iter=0,
            file_path=fpath,
            save=True,
            initial_evals={0: [dlib.function_evaluation(x=[0.3], y=0.0)]},
        ),
    )
    assert ctl.submitted == []
    with h5py.File(fpath, "r") as f:
        assert len(f["loop_zero"]["0"]["objectives"]) == 1


@pytest.mark.timeout(60)
@pytest.mark.parametrize("per_problem", [False, True])
def test_loop_default_n_max_tasks_terminates(per_problem):
    """Without MPI there are no workers; leaving n_max_tasks unset must
    still run the optimization."""
    params = multi_params("loop_default", 3, {0, 1}, per_problem=per_problem)
    del params["n_max_tasks"]
    best = distgfs.run(params)
    assert set(best) == {0, 1}
    for problem_id in (0, 1):
        assert n_evals("loop_default", problem_id) == 3


def test_save_iter_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="save_iter"):
        distgfs.DistGFSOptimizer(
            opt_id="bad_save_iter",
            obj_fun=obj_fun_record,
            problem_parameters={"y": 0.1},
            space={"x": [0.1, 0.7]},
            file_path=str(tmp_path / "ckpt.h5"),
            save=True,
            save_iter=0,
        )
    distgfs.DistGFSOptimizer(
        opt_id="unsaved_save_iter",
        obj_fun=obj_fun_record,
        problem_parameters={"y": 0.1},
        space={"x": [0.1, 0.7]},
        save_iter=0,
    )


def test_empty_problem_ids_raises():
    with pytest.raises(ValueError, match="problem_ids"):
        distgfs.DistGFSOptimizer(
            opt_id="empty_pids",
            obj_fun=obj_fun_multi_int,
            problem_parameters={"y": 1.0},
            space={"x": [-4.5, 4.5]},
            problem_ids=set(),
        )


@pytest.mark.parametrize(
    "result, message",
    [({0: -1.0}, "missing"), ({0: -1.0, 1: -1.0, 2: -1.0}, "unexpected")],
)
def test_update_result_value_checks_problem_ids(result, message):
    gfsopt = distgfs.DistGFSOptimizer(
        opt_id="check_pids",
        obj_fun=obj_fun_multi_int,
        problem_parameters={"y": 1.0},
        space={"x": [-4.5, 4.5]},
        problem_ids={0, 1},
    )
    for problem_id in (0, 1):
        gfsopt.evals[problem_id][7] = gfsopt.optimizer_dict[problem_id].get_next_x()
    with pytest.raises(ValueError, match=message):
        gfsopt.update_result_value(7, result, expected_ids=(0, 1))


@pytest.mark.parametrize(
    "ids, expected",
    [
        ({3, 1, 2}, [1, 2, 3]),
        ({"b", "a"}, ["a", "b"]),
        ([2, "a", 1], [2, "a", 1]),
        ([1, 0, 1, 0], [0, 1]),
        (["a", 1, "a"], ["a", 1]),
    ],
)
def test_ordered_problem_ids(ids, expected):
    assert distgfs._ordered_problem_ids(ids) == expected


# ---------------------------------------------------------------------------
# Result forms: the objective's mapping, or a CollectiveTaskResult from a
# worker group
# ---------------------------------------------------------------------------


CollectiveTaskResult = distwq.CollectiveTaskResult


def must_not_reduce(values, **kwargs):
    raise AssertionError("reduce_fun must not be called")


def test_result_plain_dict_used_as_is():
    res = {0: -1.0}
    assert distgfs._task_result_dict(1, res, must_not_reduce) is res


@pytest.mark.parametrize(
    "values",
    [[{0: -1.0}], [{0: -1.0}, None, None], [None, {0: -1.0}]],
)
def test_result_group_single_value(values):
    """Without reduce_fun, the one result that is not None is used."""
    res = distgfs._task_result_dict(1, CollectiveTaskResult(values))
    assert res == {0: -1.0}


def test_result_group_several_values_need_reduce_fun():
    res = CollectiveTaskResult([{0: -1.0}, None, {0: -2.0}])
    with pytest.raises(ValueError, match="2 sub-worker results.*reduce_fun"):
        distgfs._task_result_dict(1, res)


@pytest.mark.parametrize(
    "values, message",
    [([None, None], "returned None"), ([], "no results")],
)
@pytest.mark.parametrize("reduce_fun", [None, must_not_reduce])
def test_result_group_without_values_raises(values, message, reduce_fun):
    with pytest.raises(ValueError, match=message):
        distgfs._task_result_dict(1, CollectiveTaskResult(values), reduce_fun)


def test_result_group_reduce_fun_receives_present_values():
    """reduce_fun gets the results that are not None, in group order, and
    the keyword arguments in reduce_fun_args."""
    calls = []

    def reduce_fun(values, **kwargs):
        calls.append((values, kwargs))
        return {0: sum(v[0] for v in values)}

    first, second = {0: -1.0}, {0: -2.0}
    res = CollectiveTaskResult([first, None, second])
    out = distgfs._task_result_dict(1, res, reduce_fun, {"scale": 2})
    assert out == {0: -3.0}
    assert calls == [([first, second], {"scale": 2})]


def test_result_group_reduce_fun_on_single_value():
    """reduce_fun is still called when only one result is present."""
    res = CollectiveTaskResult([{0: -1.0}, None])
    out = distgfs._task_result_dict(1, res, lambda values: {0: len(values)})
    assert out == {0: 1}


@pytest.mark.parametrize("returned", [None, [{0: -1.0}], -1.0])
def test_result_reduce_fun_not_a_dict_raises(returned):
    res = CollectiveTaskResult([{0: -1.0}, {0: -2.0}])
    with pytest.raises(TypeError, match="reduce_fun"):
        distgfs._task_result_dict(1, res, lambda values: returned)


@pytest.mark.parametrize(
    "res",
    [
        (({0: -1.0}),),
        None,
        -1.0,
        "abc",
        [{0: -1.0}],
        distwq.TaskResult({0: -1.0}),
    ],
)
def test_result_stray_types_raise(res):
    """Only a mapping or a CollectiveTaskResult is accepted.  A list is not
    taken for a group's results, and a TaskResult never reaches distgfs,
    because the futures layer unwraps it."""
    with pytest.raises(TypeError, match="task 1"):
        distgfs._task_result_dict(1, res, must_not_reduce)


def two_problem_optimizer(opt_id):
    gfsopt = distgfs.DistGFSOptimizer(
        opt_id=opt_id,
        obj_fun=obj_fun_multi_int,
        problem_parameters={"y": 1.0},
        space={"x": [-4.5, 4.5]},
        problem_ids={0, 1},
    )
    for problem_id in (0, 1):
        gfsopt.evals[problem_id][7] = gfsopt.optimizer_dict[problem_id].get_next_x()
    return gfsopt


def test_update_result_value_collective():
    gfsopt = two_problem_optimizer("update_collective")
    gfsopt.update_result_value(
        7, CollectiveTaskResult([{0: -1.5, 1: -2.5}, None]), expected_ids=(0, 1)
    )
    for problem_id, y in ((0, -1.5), (1, -2.5)):
        ys = gfsopt.optimizer_dict[problem_id].get_function_evaluations()[1][0]
        assert [e.y for e in ys] == [y]


def test_update_result_value_collective_missing_id():
    """The problem-id check applies to the combined result."""
    gfsopt = two_problem_optimizer("update_collective_missing")
    with pytest.raises(ValueError, match="missing"):
        gfsopt.update_result_value(
            7, CollectiveTaskResult([{0: -1.5}]), expected_ids=(0, 1)
        )


WRAPS = {
    # Unwrapped by the TaskController, as for an ungrouped worker.
    "task_result": distwq.TaskResult,
    "collective_with_none": lambda r: CollectiveTaskResult([r, None]),
}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("wrap", sorted(WRAPS))
@pytest.mark.parametrize("per_problem", [False, True])
def test_loop_wrapped_results(wrap, per_problem):
    """The controller loop handles every result form in both modes."""
    opt_id = f"loop_wrap_{wrap}_{per_problem}"
    ctl = FakeController(pattern=(1,), reverse=True, wrap=WRAPS[wrap])
    distgfs.gfsctrl(
        fake_session(ctl),
        multi_params(opt_id, 5, {0, 1}, per_problem=per_problem, n_max_tasks=2),
    )
    for problem_id in (0, 1):
        assert n_evals(opt_id, problem_id) == 5


@pytest.mark.timeout(60)
def test_loop_wrapped_results_with_features(tmp_path):
    """Features stay aligned with their parameters when results arrive in
    reverse order as CollectiveTaskResults."""
    fpath = str(tmp_path / "ckpt.h5")
    ctl = FakeController(reverse=True, wrap=lambda r: CollectiveTaskResult([None, r]))
    distgfs.gfsctrl(
        fake_session(ctl),
        loop_params(
            "loop_wrap_feat",
            n_iter=6,
            n_max_tasks=3,
            obj_fun_name="test_distgfs.obj_fun_feature",
            feature_dtypes=[("xcopy", np.float64)],
            file_path=fpath,
            save=True,
            save_iter=6,
        ),
    )
    with h5py.File(fpath, "r") as f:
        grp = f["loop_wrap_feat"]["0"]
        xs = grp["parameters"]["x"][:]
        assert len(xs) == 6
        assert np.array_equal(grp["features"]["xcopy"][:], xs)


def mean_reduce(values, offset=0.0):
    """Average the per-process results of a single-problem group."""
    return {0: sum(v[0] for v in values) / len(values) + offset}


@pytest.mark.timeout(60)
def test_loop_reduce_fun_on_group_results():
    """With reduce_fun, the optimizer records the reduced value of each
    task's group results."""
    ctl = FakeController(
        wrap=lambda r: CollectiveTaskResult([r, {0: r[0] + 1.0}, None])
    )
    distgfs.gfsctrl(
        fake_session(ctl),
        loop_params(
            "loop_reduce",
            n_iter=6,
            n_max_tasks=2,
            reduce_fun=mean_reduce,
            reduce_fun_args={"offset": 0.25},
        ),
    )
    evals = distgfs.gfsopt_dict["loop_reduce"].optimizer_dict[0]
    evals = evals.get_function_evaluations()[1][0]
    assert len(evals) == 6
    for e in evals:
        assert math.isclose(e.y, -((e.x[0] - 0.3) ** 2) + 0.5 + 0.25)


# ---------------------------------------------------------------------------
# gfsinit: controller and worker setup
# ---------------------------------------------------------------------------


CONTROLLER_INIT_CALLS = []
OBJ_FUN_INIT_CALLS = []


def controller_init(**kwargs):
    CONTROLLER_INIT_CALLS.append(kwargs)


def obj_fun_init(worker=None, **kwargs):
    """Objective factory that records how it was called."""
    OBJ_FUN_INIT_CALLS.append((worker, kwargs))
    return obj_fun_pids


def init_params(opt_id, **extra):
    params = loop_params(
        opt_id,
        n_iter=2,
        controller_init_fun_name="test_distgfs.controller_init",
        controller_init_fun_args={"label": opt_id},
        reduce_fun_name="test_distgfs.mean_reduce",
        reduce_fun_args={"offset": 0.25},
    )
    params.update(extra)
    return params


@pytest.fixture
def init_calls():
    CONTROLLER_INIT_CALLS.clear()
    OBJ_FUN_INIT_CALLS.clear()


def set_roles(monkeypatch, is_controller, is_worker):
    monkeypatch.setattr(distwq, "is_controller", is_controller)
    monkeypatch.setattr(distwq, "is_worker", is_worker)


def test_gfsinit_single_process_does_both(init_calls):
    """A single process is controller and worker: it runs the controller
    init, resolves reduce_fun_name, and resolves the objective."""
    assert distwq.is_controller and distwq.is_worker
    gfsopt = distgfs.gfsinit(init_params("init_single"))
    assert CONTROLLER_INIT_CALLS == [{"label": "init_single"}]
    assert gfsopt.reduce_fun is mean_reduce
    assert gfsopt.reduce_fun_args == {"offset": 0.25}
    assert gfsopt.eval_fun.args[0] is obj_fun_pids


def test_gfsinit_controller_only(monkeypatch, init_calls):
    """An MPI controller rank sets up the controller side only."""
    set_roles(monkeypatch, is_controller=True, is_worker=False)
    gfsopt = distgfs.gfsinit(init_params("init_controller"))
    assert CONTROLLER_INIT_CALLS == [{"label": "init_controller"}]
    assert gfsopt.reduce_fun is mean_reduce
    assert gfsopt.eval_fun.args[0] is None


def test_gfsinit_worker_only(monkeypatch, init_calls):
    """An MPI worker rank resolves the objective only."""
    set_roles(monkeypatch, is_controller=False, is_worker=True)
    gfsopt = distgfs.gfsinit(init_params("init_worker"))
    assert CONTROLLER_INIT_CALLS == []
    assert gfsopt.reduce_fun is None
    assert gfsopt.eval_fun.args[0] is obj_fun_pids


def test_gfsinit_obj_fun_init(init_calls):
    """The objective factory receives its arguments and the worker; its
    arguments may be omitted."""
    params = init_params(
        "init_factory",
        obj_fun_init_name="test_distgfs.obj_fun_init",
        obj_fun_init_args={"scale": 2},
    )
    del params["obj_fun_name"]
    distgfs.gfsinit(params, worker="w")
    assert OBJ_FUN_INIT_CALLS == [("w", {"scale": 2})]

    del params["obj_fun_init_args"]
    params["opt_id"] = "init_factory_noargs"
    distgfs.gfsinit(params)
    assert OBJ_FUN_INIT_CALLS[-1] == (None, {})


def test_gfsinit_unknown_reduce_fun_name_raises(init_calls):
    with pytest.raises(ImportError, match="no_such_reduce"):
        distgfs.gfsinit(
            init_params("init_bad_reduce", reduce_fun_name="no_such_reduce")
        )


@pytest.mark.timeout(60)
def test_loop_reduce_fun_name_single_process(init_calls):
    """reduce_fun_name now takes effect without MPI: group results are
    reduced by the named function."""
    ctl = FakeController(wrap=lambda r: CollectiveTaskResult([r, {0: r[0] + 1.0}]))
    distgfs.gfsctrl(fake_session(ctl), init_params("loop_reduce_name", n_iter=4))
    evals = distgfs.gfsopt_dict["loop_reduce_name"].optimizer_dict[0]
    evals = evals.get_function_evaluations()[1][0]
    assert len(evals) == 4
    for e in evals:
        assert math.isclose(e.y, -((e.x[0] - 0.3) ** 2) + 0.5 + 0.25)
    assert CONTROLLER_INIT_CALLS == [{"label": "loop_reduce_name"}]


# ---------------------------------------------------------------------------
# Function paths ("module.name") and the removed *_module keys
# ---------------------------------------------------------------------------


class PathHolder:
    """Module-level class whose method is reached through an attribute path."""

    @staticmethod
    def method():
        return "held"


def test_resolve_module_function():
    assert distgfs.resolve_object_path("test_distgfs.obj_fun_pids") is obj_fun_pids


def test_resolve_package_module():
    """The longest importable prefix is the module: os.path, not os."""
    assert distgfs.resolve_object_path("os.path.join") is os.path.join


def test_resolve_attribute_chain():
    assert distgfs.resolve_object_path("test_distgfs.PathHolder.method")() == "held"


def test_resolve_main(monkeypatch):
    """A name without an importable module part is looked up in __main__,
    including attribute chains; '__main__.' may also be given."""
    main = sys.modules["__main__"]
    monkeypatch.setattr(main, "distgfs_main_fun", obj_fun_pids, raising=False)
    monkeypatch.setattr(main, "DistgfsMainHolder", PathHolder, raising=False)
    assert distgfs.resolve_object_path("distgfs_main_fun") is obj_fun_pids
    assert distgfs.resolve_object_path("__main__.distgfs_main_fun") is obj_fun_pids
    assert distgfs.resolve_object_path("DistgfsMainHolder.method")() == "held"


def test_resolve_missing_attribute_raises():
    with pytest.raises(ImportError, match="'test_distgfs' has no attribute 'no_such'"):
        distgfs.resolve_object_path("test_distgfs.no_such")


def test_resolve_missing_module_raises():
    with pytest.raises(ImportError, match="no module 'no_such_module_xyz'"):
        distgfs.resolve_object_path("no_such_module_xyz.obj_fun")


def test_resolve_missing_name_in_main_raises():
    with pytest.raises(ImportError, match="'__main__' has no attribute"):
        distgfs.resolve_object_path("no_such_main_fun_xyz")


def test_resolve_broken_module_import_propagates(tmp_path, monkeypatch):
    """A module that exists but fails to import one of its own dependencies
    reports that failure instead of being skipped."""
    (tmp_path / "distgfs_broken_mod_xyz.py").write_text(
        "import no_such_dependency_xyz\n\ndef fun():\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="no_such_dependency_xyz"):
        distgfs.resolve_object_path("distgfs_broken_mod_xyz.fun")


@pytest.mark.parametrize("path", ["", ".fun", "mod..fun", "fun.", "mod.1fun", "a b"])
def test_resolve_malformed_path_raises(path):
    with pytest.raises(ImportError, match="not a valid"):
        distgfs.resolve_object_path(path)


@pytest.mark.parametrize("path", [None, 3, ["test_distgfs", "obj_fun"]])
def test_resolve_non_string_raises(path):
    with pytest.raises(TypeError):
        distgfs.resolve_object_path(path)


def test_resolve_callable_rejects_data():
    with pytest.raises(TypeError, match="obj_fun_name 'test_distgfs.RECORDED_CALLS'"):
        distgfs.resolve_callable("test_distgfs.RECORDED_CALLS", "obj_fun_name")


@pytest.mark.parametrize("key", sorted(distgfs.REMOVED_MODULE_KEYS))
def test_removed_module_keys_raise(key, init_calls):
    """Each removed key is rejected, by gfsinit and by run(), with a hint
    naming the key that now carries the module."""
    params = loop_params(f"removed_{key}", n_iter=1)
    params[key] = "test_distgfs"
    hint = f"{distgfs.REMOVED_MODULE_KEYS[key]}='test_distgfs.<name>'"
    with pytest.raises(TypeError, match=re.escape(hint)):
        distgfs.gfsinit(dict(params))
    with pytest.raises(TypeError, match=key):
        distgfs.run(dict(params))


def test_obj_fun_name_with_module_runs():
    """An end-to-end run with a module path finds and calls the objective."""
    RECORDED_CALLS.clear()
    distgfs.run(loop_params("path_run", n_iter=3))
    assert len(RECORDED_CALLS) == 3


BROKER_CALLS = []


def broker_fun(broker, gfsopt_params, verbose):
    BROKER_CALLS.append((broker, gfsopt_params["opt_id"], verbose))
    return "broker done"


def test_gfsbroker_calls_named_function():
    BROKER_CALLS.clear()
    params = {"opt_id": "broker", "broker_fun_name": "test_distgfs.broker_fun"}
    assert distgfs.gfsbroker("B", params, True) == "broker done"
    assert BROKER_CALLS == [("B", "broker", True)]


class RecordingEnvironment:
    """Stand-in for distwq.future.DistributedEnvironment that records how
    run() constructs it, and acts as a worker rank that receives nothing."""

    created = []

    def __init__(self, config, worker_config, **kwargs):
        self.config = config
        self.worker_config = worker_config
        self.kwargs = kwargs
        self.worker_count = 0
        RecordingEnvironment.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def controller_session(self):
        return iter(())

    def broadcast(self, value=None):
        return "broadcast value"


@pytest.fixture
def recording_env(monkeypatch):
    RecordingEnvironment.created.clear()
    monkeypatch.setattr(distgfs, "DistributedEnvironment", RecordingEnvironment)
    return RecordingEnvironment.created


@pytest.mark.parametrize(
    "spawn, broker_name",
    [(False, None), (True, None), (True, "test_distgfs.broker_fun")],
)
def test_run_environment_settings(recording_env, spawn, broker_name):
    """run() gives distwq gfswork as the worker init function, distgfs's
    own broker trampoline when a broker function is named (spawn only),
    and worker arguments without the saving keys; the caller's dict is
    left as it is."""
    params = loop_params("env_run", n_iter=1, file_path="unused.h5", save=True)
    if broker_name is not None:
        params["broker_fun_name"] = broker_name
    before = dict(params)
    best = distgfs.run(params, spawn_workers=spawn, max_workers=3, nprocs_per_worker=2)
    assert best == "broadcast value"
    assert params == before
    (env,) = recording_env
    assert env.kwargs["worker_init"] is distgfs.gfswork
    expected_broker = None if broker_name is None else distgfs.gfsbroker
    assert env.kwargs["broker_init"] is expected_broker
    worker_params, verbose = env.kwargs["init_args"]
    assert "file_path" not in worker_params and "save" not in worker_params
    assert worker_params["opt_id"] == "env_run"
    assert verbose is False
    assert env.config.resources.max_workers == 3
    wc = env.worker_config
    assert wc.nprocs_per_worker == 2
    assert wc.broker_is_worker is (not spawn)
    expected_method = (
        distwq.GroupingMethod.GroupSpawn if spawn else distwq.GroupingMethod.GroupSplit
    )
    assert wc.grouping_method == expected_method


def test_run_broker_fun_needs_spawn(recording_env):
    params = loop_params("env_broker_split", n_iter=1)
    params["broker_fun_name"] = "test_distgfs.broker_fun"
    with pytest.raises(ValueError, match="spawn_workers"):
        distgfs.run(params)
    assert recording_env == []


def test_run_unknown_broker_fun_raises_before_setup(recording_env):
    params = loop_params("env_broker_missing", n_iter=1)
    params["broker_fun_name"] = "test_distgfs.no_such_broker"
    with pytest.raises(ImportError, match="no_such_broker"):
        distgfs.run(params, spawn_workers=True)
    assert recording_env == []


# ---------------------------------------------------------------------------
# Objective failures: checkpoint, then raise
# ---------------------------------------------------------------------------


class ObjectiveError(Exception):
    pass


FAIL_PID = 3


def obj_fun_fail(pp, pid):
    """Single-problem objective that raises on pid FAIL_PID."""
    if pid == FAIL_PID:
        raise ObjectiveError(f"objective failed at pid {pid}")
    return -((pp["x"] - 0.3) ** 2)


def obj_fun_multi_fail(pp, pid):
    """Multi-problem objective that raises on pid FAIL_PID of problem 1."""
    if pid == FAIL_PID and 1 in pp:
        raise ObjectiveError(f"objective failed at pid {pid}")
    return {
        problem_id: -((params["x"] - 0.3) ** 2) for problem_id, params in pp.items()
    }


def checkpoint_counts(fpath, opt_id, problem_ids):
    with h5py.File(fpath, "r") as f:
        return {
            problem_id: len(f[opt_id][str(problem_id)]["objectives"])
            for problem_id in problem_ids
        }


@pytest.mark.timeout(60)
@pytest.mark.parametrize("n_max_tasks", [1, 2])
def test_loop_objective_failure_saves_then_raises(tmp_path, n_max_tasks):
    """The objective's exception reaches the caller, after a checkpoint
    with every result received before it; the failed point is not
    recorded."""
    fpath = str(tmp_path / "ckpt.h5")
    ctl = FakeController(pattern=(1,))
    params = loop_params(
        "fail_batch",
        n_iter=8,
        n_max_tasks=n_max_tasks,
        obj_fun_name="test_distgfs.obj_fun_fail",
        file_path=fpath,
        save=True,
        save_iter=100,
    )
    with pytest.raises(ObjectiveError, match="pid 3"):
        distgfs.gfsctrl(fake_session(ctl), params)
    # Tasks for pids 0 .. 3 complete, in order, and the one for pid 3 fails.
    assert n_evals("fail_batch") == FAIL_PID
    assert checkpoint_counts(fpath, "fail_batch", [0]) == {0: FAIL_PID}


@pytest.mark.timeout(60)
def test_loop_objective_failure_keeps_results_of_same_wakeup(tmp_path):
    """Results that arrive together with a failure are still recorded and
    saved, whatever their order."""
    fpath = str(tmp_path / "ckpt.h5")
    ctl = FakeController(reverse=True)
    params = multi_params(
        "fail_rolling",
        6,
        {0, 1, 2},
        per_problem=True,
        n_max_tasks=12,
        obj_fun_name="test_distgfs.obj_fun_multi_fail",
        file_path=fpath,
        save=True,
        save_iter=100,
    )
    with pytest.raises(ObjectiveError):
        distgfs.gfsctrl(fake_session(ctl), params)
    # The first 12 tasks (pids 0 .. 3 of each problem) arrive together, and
    # only problem 1's pid 3 fails.
    expected = {0: 4, 1: 3, 2: 4}
    assert {p: n_evals("fail_rolling", p) for p in (0, 1, 2)} == expected
    assert checkpoint_counts(fpath, "fail_rolling", [0, 1, 2]) == expected


@pytest.mark.timeout(60)
def test_run_objective_failure_single_process(tmp_path):
    """Without MPI, run() raises the objective's own exception type, and
    the checkpoint holds the results before it."""
    fpath = str(tmp_path / "ckpt.h5")
    params = loop_params(
        "fail_run",
        n_iter=8,
        obj_fun_name="test_distgfs.obj_fun_fail",
        file_path=fpath,
        save=True,
        save_iter=2,
    )
    with pytest.raises(ObjectiveError, match="pid 3"):
        distgfs.run(params)
    assert checkpoint_counts(fpath, "fail_run", [0]) == {0: FAIL_PID}
