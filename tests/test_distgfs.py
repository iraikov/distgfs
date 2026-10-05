import math
import os
import shutil
import sys
import tempfile
import warnings

import dlib
import h5py
import numpy as np
import pytest

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


def test_initial_evals_seed_fresh_run():
    """initial_evals seeds a fresh (non-resumed) run; best result never
    regresses below the seeded point, which is the true optimum here."""
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}
    seed_x = 1.0 / 0.4
    distgfs_params = {
        "opt_id": "distgfs_levi_seed",
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "test_distgfs",
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
            "obj_fun_name": "obj_fun",
            "obj_fun_module": "test_distgfs",
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
            "obj_fun_name": "obj_fun",
            "obj_fun_module": "test_distgfs",
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
        "obj_fun_name": "obj_fun_record",
        "obj_fun_module": "test_distgfs",
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
            "obj_fun_name": "obj_fun_multi_int",
            "obj_fun_module": "test_distgfs",
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

    Each task runs as soon as it is submitted, but its result is held back,
    as if a worker were still computing it.  Every call to process() releases
    held results according to 'pattern', a cycle of counts ("all" releases
    everything; 0 is a wakeup that brings no result, like a READY message).
    'reverse' releases the newest held results first.  With
    'release_on_submit', submit_call() releases results before dispatching,
    as distwq's submit_call() does through its own process() call.

    A blocking process() with nothing held raises AssertionError, turning
    a hang into a test failure.
    """

    def __init__(
        self,
        pattern=("all",),
        reverse=False,
        release_on_submit=False,
        limit=None,
        extra_ids=(),
    ):
        self.pattern = list(pattern)
        self.n_wakeups = 0
        self.reverse = reverse
        self.release_on_submit = release_on_submit
        self.limit = limit
        self.extra_ids = list(extra_ids)
        self.held = []
        self.ready = []
        self.count = 0
        self.inflight_at_submit = []
        self.submitted = []
        self.blocked_with_ready = 0

    def _outstanding(self):
        return len(self.held) + len(self.ready)

    def _release(self):
        n = self.pattern[self.n_wakeups % len(self.pattern)]
        self.n_wakeups += 1
        n = len(self.held) if n == "all" else min(n, len(self.held))
        for _ in range(n):
            self.ready.append(self.held.pop() if self.reverse else self.held.pop(0))

    def submit_call(self, name_to_call, args=(), kwargs=None, module_name="__main__"):
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
        self.held.append((task_id, fun(*args, **(kwargs or {}))))
        return task_id

    def process(self, limit=1000, block=False):
        if block:
            assert self.held or self.ready, "blocking with nothing outstanding"
            if self.ready:
                self.blocked_with_ready += 1
        self._release()

    def probe_all_next_results(self):
        ret = [(task_id, {}) for task_id in self.extra_ids] + self.ready
        self.extra_ids = []
        self.ready = []
        return ret

    def info(self):
        pass


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
        "obj_fun_name": "obj_fun_pids",
        "obj_fun_module": "test_distgfs",
        "problem_parameters": {"y": 0.1},
        "space": {"x": [0.1, 0.7]},
        "n_iter": n_iter,
        "n_max_tasks": n_max_tasks,
    }
    params.update(extra)
    return params


def multi_params(opt_id, n_iter, problem_ids, per_problem, n_max_tasks=1, **extra):
    return loop_params(
        opt_id,
        n_iter,
        n_max_tasks,
        obj_fun_name="obj_fun_multi_pids",
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
    distgfs.gfsctrl(ctl, loop_params("loop_batch", n_iter=5, n_max_tasks=2))
    assert ctl.inflight_at_submit == [0, 1, 0, 1, 0]
    assert n_evals("loop_batch") == 5
    assert ctl._outstanding() == 0


@pytest.mark.timeout(60)
def test_loop_batch_limit_larger_than_total():
    ctl = FakeController()
    distgfs.gfsctrl(ctl, loop_params("loop_batch_big", n_iter=5, n_max_tasks=10))
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
        ctl,
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
        ctl, multi_params("loop_submit", 6, {0, 1, 2}, per_problem=True, n_max_tasks=2)
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
        ctl, multi_params("loop_fair", 4, {0, 1, 2}, per_problem=True, n_max_tasks=2)
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
        ctl, multi_params("loop_rr", 3, {2, 0, 1}, per_problem=True, n_max_tasks=1)
    )
    assert [ids for _, ids in ctl.submitted] == [(0,), (1,), (2,)] * 3


@pytest.mark.timeout(60)
def test_loop_out_of_order_features(tmp_path):
    """Features stay aligned with their parameters when results arrive in
    reverse order."""
    fpath = str(tmp_path / "ckpt.h5")
    ctl = FakeController(reverse=True)
    distgfs.gfsctrl(
        ctl,
        loop_params(
            "loop_feat",
            n_iter=6,
            n_max_tasks=3,
            obj_fun_name="obj_fun_feature",
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
        ctl,
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


@pytest.mark.timeout(60)
def test_loop_unknown_task_id_raises():
    ctl = FakeController(extra_ids=[999])
    with pytest.raises(RuntimeError, match="999"):
        distgfs.gfsctrl(ctl, loop_params("loop_unknown", n_iter=3))


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
