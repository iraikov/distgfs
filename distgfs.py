import copy
import importlib
import logging
import os
import sys
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import distwq
import dlib
import numpy as np
from dlib import function_evaluation, function_spec
from h5py._hl.dataset import Dataset
from h5py._hl.files import File
from h5py._hl.group import Group
from numpy import ndarray

try:
    import h5py
except ImportError as e:
    warnings.warn(
        f"distgfs: unable to import h5py: {e}", category=warnings.ImportWarning
    )

gfsopt_dict = {}


def validate_inputs(
    problem_parameters: Dict[str, float],
    space: Dict[str, List[float]],
    save: bool,
    file_path: Optional[str],
) -> None:
    # Verify inputs
    if file_path is None:
        if problem_parameters is None or space is None:
            raise ValueError(
                "You must specify at least file name `file_path` or problem "
                "parameters `problem_parameters` "
                "along with a hyperparameter space `space`."
            )
        if save:
            raise ValueError(
                "If you want to save you must specify a file name `file_path`."
            )
    else:
        if not os.path.isfile(file_path):
            if problem_parameters is None or space is None:
                raise FileNotFoundError(file_path)


class DistGFSOptimizer:
    def __init__(
        self,
        opt_id: str,
        obj_fun: Callable,
        reduce_fun: Optional[Callable] = None,
        reduce_fun_args: Dict[str, Any] = dict(),
        problem_ids: Optional[List[int]] = None,
        problem_parameters: Optional[Dict[str, float]] = None,
        space: Optional[Dict[str, List[float]]] = None,
        feature_dtypes: Optional[List[Tuple[str, Any]]] = None,
        constraint_names: Optional[List[str]] = None,
        solver_epsilon: float = 0.0005,
        relative_noise_magnitude: float = 0.001,
        seed: Optional[Any] = None,
        n_iter: int = 100,
        n_max_tasks: int = -1,
        save_iter: int = 10,
        file_path: Optional[str] = None,
        save: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """`Creates an optimizer based on the Global Function Search
        <http://dlib.net/optimization.html#global_function_search>`_
        (GFS) optimizer in dlib. Supports distributed optimization
        runs via mpi4py. Based on GFSOPtimizer by https://github.com/tsoernes

        :param string opt_id: optimization group id
            An identifier to associate with this class of optimization runs.
        :param func obj_fun: function to maximize.
            Must take as argument every parameter specified in
            both 'problem_parameters' and 'space', in addition to 'pid',
            and return the result as float.
            'pid' specifies simulation run number.
            If you want to minimize instead,
            simply negate the result in the objective function before returning it.
        :param set problem_ids (optional): Set of problem ids.
            For solving sets of related problems with the same set of parameters.
            If this parameter is not None, it is expected that the objective function
            will return a dictionary of the form { problem_id: value }
        :param dict problem_parameters: Problem parameters.
            All hyperparameters and their values for the objective
            function, including those not being optimized over. E.g: ``{'beta': 0.44}``.
            Can be an empty dict.
            Can include hyperparameters being optimized over, but does not need to.
            If a hyperparameter is specified in both 'problem_parameters' and 'space',
            its value in 'problem_parameters' will be overridden.
        :param dict space: Hyperparameters to optimize over.
            Entries should be of the form:
            ``parameter: (Low_Bound, High_Bound)`` e.g:
            ``{'alpha': (0.65, 0.85), 'gamma': (1, 8)}``. If both bounds for a
            parameter are Ints, then only integers within the (inclusive) range
            will be sampled and tested.
        :param func reduce_fun: function to reduce multiple results per evaluation
            obtained from each distributed worker. Must take as argument a list
            of objective evaluations.
        :param int n_iter: (optional) Number of times to sample and test params.
        :param int save_iter: (optional) How often to save progress.
        :param str file_path: (optional) File name for restoring
            and/or saving results and settings.
        :param bool save: (optional) Save settings and progress periodically.
        :param float solver_epsilon: (optional) The accuracy to which local optima
            are determined before global exploration is resumed.
            See `Dlib <http://dlib.net/dlib/global_optimization/
            global_function_search_abstract.h.html#global_function_search>`_
            for further documentation. Default: 0.0005
        :param float relative_noise_magnitude: (optional) Should be increased for
            highly stochastic objective functions. Deterministic and continuous
            functions can use a value of 0. See `Dlib
            <http://dlib.net/dlib/global_optimization/upper_bound_function_abstract.h.html
            #upper_bound_function>`_
            for further documentation. Default: 0.001
        :param float seed: (optional) Sets the seed used for random
            sampling by the optimization algorithm. If None, the optimizer will always
            produce the same deterministic behavior.  Default: None
        """

        self.opt_id = opt_id
        self.verbose = verbose

        self.logger = logging.getLogger(opt_id)
        if self.verbose:
            self.logger.setLevel(logging.INFO)

        validate_inputs(problem_parameters, space, save, file_path)

        eps = solver_epsilon
        noise_mag = relative_noise_magnitude

        param_names, is_int, lo_bounds, hi_bounds = [], [], [], []
        if space is not None:
            for parm, conf in space.items():
                param_names.append(parm)
                lo, hi = conf
                is_int.append(isinstance(lo, int) and isinstance(hi, int))
                lo_bounds.append(lo)
                hi_bounds.append(hi)
        old_evals = {}
        if (file_path is not None) and os.path.isfile(file_path):
            (
                old_evals,
                old_feature_evals,
                old_constraint_evals,
                param_names,
                feature_dtypes,
                constraint_names,
                is_int,
                lo_bounds,
                hi_bounds,
                eps,
                noise_mag,
                problem_parameters,
                problem_ids,
            ) = init_from_h5(file_path, param_names, opt_id, self.logger)

        self.feature_dtypes = feature_dtypes
        self.feature_names = (
            None if feature_dtypes is None else [dtype[0] for dtype in feature_dtypes]
        )

        self.constraint_names = constraint_names

        spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)

        has_problem_ids = problem_ids is not None
        if not has_problem_ids:
            problem_ids = set([0])

        n_saved_evals = 0
        n_saved_features = 0
        n_saved_constraints = 0
        optimizer_dict = {}
        for problem_id in problem_ids:
            if problem_id in old_evals:
                optimizer = dlib.global_function_search(
                    [spec],
                    initial_function_evals=[old_evals[problem_id]],
                    relative_noise_magnitude=noise_mag,
                )
                n_saved_evals = len(old_evals[problem_id])
            else:
                optimizer = dlib.global_function_search([spec])
                optimizer.set_relative_noise_magnitude(noise_mag)
            optimizer.set_solver_epsilon(eps)
            if seed is not None:
                optimizer.set_seed(seed)
            optimizer_dict[problem_id] = optimizer

        self.optimizer_dict = optimizer_dict
        self.metadata = metadata
        self.problem_parameters, self.param_names, self.spec = (
            problem_parameters,
            param_names,
            spec,
        )
        self.eps, self.noise_mag, self.is_int = eps, noise_mag, is_int
        self.file_path, self.save = file_path, save

        self.n_iter = n_iter
        self.n_max_tasks = n_max_tasks
        self.n_saved_evals = n_saved_evals
        self.n_saved_features = n_saved_features
        self.n_saved_constraints = n_saved_constraints
        self.save_iter = save_iter

        if has_problem_ids:
            self.eval_fun = partial(
                eval_obj_fun_mp,
                obj_fun,
                self.problem_parameters,
                self.param_names,
                self.is_int,
                problem_ids,
            )
        else:
            self.eval_fun = partial(
                eval_obj_fun_sp,
                obj_fun,
                self.problem_parameters,
                self.param_names,
                self.is_int,
                0,
            )

        self.reduce_fun = reduce_fun
        self.reduce_fun_args = reduce_fun_args

        self.evals = {problem_id: {} for problem_id in problem_ids}
        self.feature_evals = None
        if self.feature_names is not None:
            self.feature_evals = {problem_id: [] for problem_id in problem_ids}
        self.constraint_evals = None
        if self.constraint_names is not None:
            self.constraint_evals = {problem_id: [] for problem_id in problem_ids}

        self.has_problem_ids = has_problem_ids
        self.problem_ids = problem_ids

    def save_evals(self) -> None:
        """Store results of finished evals to file; print best eval"""
        finished_feature_evals = None
        finished_constraint_evals = None
        eval_offset = self.n_saved_evals
        finished_evals = {
            problem_id: self.optimizer_dict[problem_id].get_function_evaluations()[1][
                0
            ][eval_offset:]
            for problem_id in self.problem_ids
        }
        if self.feature_dtypes is not None:
            feature_offset = self.n_saved_features
            finished_feature_evals = {
                problem_id: list(
                    [x[1] for x in self.feature_evals[problem_id][feature_offset:]]
                )
                for problem_id in self.problem_ids
            }
            self.n_saved_features += len(
                finished_feature_evals[next(iter(self.problem_ids))]
            )
        if self.constraint_names is not None:
            constraint_offset = self.n_saved_constraints
            finished_constraint_evals = {
                problem_id: list(
                    [
                        x[1]
                        for x in self.constraint_evals[problem_id][constraint_offset:]
                    ]
                )
                for problem_id in self.problem_ids
            }
            self.n_saved_constraints += len(
                finished_constraint_evals[next(iter(self.problem_ids))]
            )
        save_to_h5(
            self.opt_id,
            self.problem_ids,
            self.has_problem_ids,
            self.feature_dtypes,
            self.constraint_names,
            self.param_names,
            self.spec,
            finished_evals,
            finished_feature_evals,
            finished_constraint_evals,
            self.eps,
            self.noise_mag,
            self.problem_parameters,
            self.metadata,
            self.file_path,
            self.logger,
        )

        self.n_saved_evals += len(finished_evals[next(iter(self.problem_ids))])

    def get_best(self) -> Tuple[List[Tuple[str, float]], float]:
        best_results = {}
        for problem_id in self.problem_ids:
            best_eval = self.optimizer_dict[problem_id].get_best_function_eval()
            prms = list(zip(self.param_names, list(best_eval[0])))
            res = best_eval[1]
            best_results[problem_id] = (prms, res)
        if self.has_problem_ids:
            return best_results
        else:
            return best_results[problem_id]

    def print_best(self) -> None:
        best_results = self.get_best()
        if self.has_problem_ids:
            for problem_id in self.problem_ids:
                res, prms = best_results[problem_id]
                self.logger.info(f"Best eval so far for id {problem_id}: {res}@{prms}")
        else:
            res, prms = best_results
            self.logger.info(f"Best eval so far for: {res}@{prms}")

    def update_result_value(
        self, task_id: int, res: Dict[int, Tuple[float, ndarray]]
    ) -> None:
        rres = res
        if self.reduce_fun is not None:
            rres = self.reduce_fun(res, **self.reduce_fun_args)
        for problem_id in rres:
            eval_req = self.evals[problem_id][task_id]
            parameters = list(eval_req.x)
            resval = None
            feature = None
            constraint = None
            if (self.feature_names is None) and (self.constraint_names is None):
                resval = rres[problem_id]
            else:
                resval = rres[problem_id][0]
                if (self.feature_names is not None) and (
                    self.constraint_names is not None
                ):
                    feature = rres[problem_id][1]
                    constraint = rres[problem_id][2]
                    self.feature_evals[problem_id].append((task_id, feature))
                    self.constraint_evals[problem_id].append((task_id, constraint))
                elif self.feature_names is not None:
                    feature = rres[problem_id][1]
                    self.feature_evals[problem_id].append((task_id, feature))
                elif self.constraint_names is not None:
                    constraint = rres[problem_id][1]
                    self.constraint_evals[problem_id].append((task_id, constraint))

            self.logger.info(
                f"problem id {problem_id}: "
                f"task id {task_id}: "
                f"parameter coordinates {parameters}:{resval} "
                f"{'' if feature is None else feature}"
            )

            eval_req.set(resval)


def h5_get_group(h: Union[File, Group], groupname: str) -> Group:
    if groupname in h.keys():
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g


def h5_get_dataset(g: Group, dsetname: str, **kwargs) -> Dataset:
    if dsetname in g.keys():
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset


def h5_concat_dataset(dset: Dataset, data: ndarray) -> Dataset:
    dsize = dset.shape[0]
    newshape = (dsize + len(data),)
    dset.resize(newshape)
    dset[dsize:] = data
    return dset


def h5_init_types(
    f: File,
    opt_id: str,
    feature_dtypes: List[
        Union[Tuple[str, Tuple[Type[int], int]], Tuple[str, Tuple[Type[float], int]]]
    ],
    constraint_names: Optional[List[str]],
    param_names: List[str],
    problem_parameters: Dict[str, float],
    spec: function_spec,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    opt_grp = h5_get_group(f, opt_id)

    param_keys = set(param_names)
    param_keys.update(problem_parameters.keys())
    # create an HDF5 enumerated type for the parameter label
    param_mapping = {name: idx for (idx, name) in enumerate(param_keys)}

    feature_keys = None
    if feature_dtypes is not None:
        feature_keys = [feature_dtype[0] for feature_dtype in feature_dtypes]

    # create HDF5 types for features, if any
    feature_mapping = None
    if feature_keys is not None:
        feature_mapping = {name: idx for (idx, name) in enumerate(feature_keys)}

    constraint_mapping = None
    if constraint_names is not None:
        constraint_mapping = {name: idx for (idx, name) in enumerate(constraint_names)}

    objective_names = ["y"]
    objective_mapping = {name: idx for (idx, name) in enumerate(objective_names)}
    dt = h5py.enum_dtype(objective_mapping, basetype=np.uint16)
    opt_grp["objective_enum"] = dt
    dt = np.dtype({"names": objective_names, "formats": [np.float32]})
    opt_grp["objective_type"] = dt
    dt = np.dtype([("objective", opt_grp["objective_enum"])])
    opt_grp["objective_spec_type"] = dt
    dset = h5_get_dataset(
        opt_grp,
        "objective_spec",
        maxshape=(len(objective_names),),
        dtype=opt_grp["objective_spec_type"].dtype,
    )
    dset.resize((len(objective_names),))
    a = np.zeros(len(objective_names), dtype=opt_grp["objective_spec_type"].dtype)
    for idx, parm in enumerate(objective_names):
        a[idx]["objective"] = objective_mapping[parm]
    dset[:] = a

    if feature_mapping is not None:
        dt = h5py.enum_dtype(feature_mapping, basetype=np.uint16)
        opt_grp["feature_enum"] = dt

        dt = np.dtype([("feature", opt_grp["feature_enum"])])
        opt_grp["feature_spec_type"] = dt

        dt = np.dtype(feature_dtypes)
        opt_grp["feature_type"] = dt

        dset = h5_get_dataset(
            opt_grp,
            "feature_spec",
            maxshape=(len(feature_keys),),
            dtype=opt_grp["feature_spec_type"].dtype,
        )
        dset.resize((len(feature_keys),))
        a = np.zeros(len(feature_keys), dtype=opt_grp["feature_spec_type"].dtype)
        for idx, parm in enumerate(feature_keys):
            a[idx]["feature"] = feature_mapping[parm]
        dset[:] = a

    if constraint_mapping is not None:
        dt = h5py.enum_dtype(constraint_mapping, basetype=np.uint16)
        opt_grp["constraint_enum"] = dt

        dt = np.dtype([("constraint", opt_grp["constraint_enum"])])
        opt_grp["constraint_spec_type"] = dt

        dt = np.dtype({"names": constraint_names, "formats": [np.int8]})
        opt_grp["constraint_type"] = dt

        dset = h5_get_dataset(
            opt_grp,
            "constraint_spec",
            maxshape=(len(constraint_names),),
            dtype=opt_grp["constraint_spec_type"].dtype,
        )
        dset.resize((len(constraint_names),))
        a = np.zeros(len(constraint_names), dtype=opt_grp["constraint_spec_type"].dtype)
        for idx, parm in enumerate(constraint_names):
            a[idx]["constraint"] = constraint_mapping[parm]
        dset[:] = a

    dt = h5py.enum_dtype(param_mapping, basetype=np.uint16)
    opt_grp["parameter_enum"] = dt

    dt = np.dtype([("parameter", opt_grp["parameter_enum"]), ("value", np.float32)])
    opt_grp["problem_parameters_type"] = dt

    dset = h5_get_dataset(
        opt_grp,
        "problem_parameters",
        maxshape=(len(param_mapping),),
        dtype=opt_grp["problem_parameters_type"].dtype,
    )
    dset.resize((len(param_mapping),))
    a = np.zeros(len(param_mapping), dtype=opt_grp["problem_parameters_type"].dtype)
    idx = 0
    for idx, (parm, val) in enumerate(problem_parameters.items()):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["value"] = val
    dset[:] = a

    dt = np.dtype(
        [
            ("parameter", opt_grp["parameter_enum"]),
            ("is_integer", bool),
            ("lower", np.float32),
            ("upper", np.float32),
        ]
    )
    opt_grp["parameter_spec_type"] = dt

    is_integer = np.asarray(spec.is_integer_variable, dtype=bool)
    upper = np.asarray(spec.upper, dtype=np.float32)
    lower = np.asarray(spec.lower, dtype=np.float32)

    dset = h5_get_dataset(
        opt_grp,
        "parameter_spec",
        maxshape=(len(param_names),),
        dtype=opt_grp["parameter_spec_type"].dtype,
    )
    dset.resize((len(param_names),))
    a = np.zeros(len(param_names), dtype=opt_grp["parameter_spec_type"].dtype)
    for idx, (parm, is_int, hi, lo) in enumerate(
        zip(param_names, is_integer, upper, lower)
    ):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["is_integer"] = is_int
        a[idx]["lower"] = lo
        a[idx]["upper"] = hi
    dset[:] = a

    dt = np.dtype({"names": param_names, "formats": [np.float32] * len(param_names)})
    opt_grp["parameter_space_type"] = dt


def h5_load_raw(input_file, opt_id):
    # N is number of trials
    # M is number of hyperparameters
    f = h5py.File(input_file, "r")
    opt_grp = h5_get_group(f, opt_id)
    solver_epsilon = opt_grp["solver_epsilon"][()]
    relative_noise_magnitude = opt_grp["relative_noise_magnitude"][()]

    feature_names = None
    feature_types = None
    if "feature_enum" in opt_grp:
        feature_enum_dict = h5py.check_enum_dtype(opt_grp["feature_enum"].dtype)
        feature_idx_dict = {parm: idx for parm, idx in feature_enum_dict.items()}
        feature_name_dict = {idx: parm for parm, idx in feature_idx_dict.items()}
        feature_names = [
            feature_name_dict[spec[0]] for spec in iter(opt_grp["feature_spec"])
        ]
        feature_dtype = opt_grp["feature_type"].dtype
        feature_types = [feature_dtype.fields[x] for x in feature_dtype.fields]

    constraint_names = None
    if "constraint_enum" in opt_grp:
        constraint_enum_dict = h5py.check_enum_dtype(opt_grp["constraint_enum"].dtype)
        constraint_idx_dict = {parm: idx for parm, idx in constraint_enum_dict.items()}
        constraint_name_dict = {idx: parm for parm, idx in constraint_idx_dict.items()}
        constraint_names = [
            constraint_name_dict[spec[0]] for spec in iter(opt_grp["constraint_spec"])
        ]

    parameter_enum_dict = h5py.check_enum_dtype(opt_grp["parameter_enum"].dtype)
    parameters_idx_dict = {parm: idx for parm, idx in parameter_enum_dict.items()}
    parameters_name_dict = {idx: parm for parm, idx in parameters_idx_dict.items()}

    problem_parameters = {
        parameters_name_dict[idx]: val for idx, val in opt_grp["problem_parameters"]
    }
    parameter_specs = [
        (parameters_name_dict[spec[0]], tuple(spec)[1:])
        for spec in iter(opt_grp["parameter_spec"])
    ]

    problem_ids = None
    if "problem_ids" in opt_grp:
        problem_ids = set(opt_grp["problem_ids"])

    raw_results = {}
    for problem_id in problem_ids if problem_ids is not None else [0]:
        if str(problem_id) in opt_grp:
            raw_results[problem_id] = {
                "objectives": opt_grp[str(problem_id)]["objectives"][:],
                "parameters": opt_grp[str(problem_id)]["parameters"][:],
            }
            if "features" in opt_grp[str(problem_id)]:
                raw_results[problem_id]["features"] = opt_grp[str(problem_id)][
                    "features"
                ][:]
            if "constraints" in opt_grp[str(problem_id)]:
                raw_results[problem_id]["constraints"] = opt_grp[str(problem_id)][
                    "constraints"
                ][:]

    f.close()

    param_names = []
    is_integer = []
    lower = []
    upper = []
    for parm, spec in parameter_specs:
        param_names.append(parm)
        is_int, lo, hi = spec
        is_integer.append(is_int)
        lower.append(lo)
        upper.append(hi)

    raw_spec = (is_integer, lower, upper)
    info = {
        "features": feature_names,
        "feature_types": feature_types,
        "constraints": constraint_names,
        "params": param_names,
        "solver_epsilon": solver_epsilon,
        "relative_noise_magnitude": relative_noise_magnitude,
        "problem_parameters": problem_parameters,
        "problem_ids": problem_ids,
    }

    return raw_spec, raw_results, info


def h5_load_all(file_path, opt_id):
    """
    Loads an HDF5 file containing
    (spec, results, info) where
      results: np.array of shape [N, M+1] where
        N is number of trials
        M is number of hyperparameters
        results[:, 0] is result/loss
        results[:, 1:] is [param1, param2, ...]
      spec: (is_integer, lower, upper)
        where each element is list of length M
      info: dict with keys
        params, solver_epsilon, relative_noise_magnitude, problem
    Assumes the structure is located in group /{opt_id}
    Returns
    (dlib.function_spec, [dlib.function_eval], dict, prev_best)
      where prev_best: np.array[result, param1, param2, ...]
    """
    raw_spec, raw_problem_results, info = h5_load_raw(file_path, opt_id)
    is_integer, lo_bounds, hi_bounds = raw_spec
    feature_names = info["features"]
    constraint_names = info.get("constraints", None)
    spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_integer)
    evals = {problem_id: [] for problem_id in raw_problem_results}
    n_features = 0
    feature_evals = None
    if feature_names is not None:
        n_features = len(feature_names)
        feature_evals = {problem_id: [] for problem_id in raw_problem_results}
    n_constraints = 0
    constraint_evals = None
    if constraint_names is not None:
        n_constraints = len(constraint_names)
        constraint_evals = {problem_id: [] for problem_id in raw_problem_results}
    prev_best_dict = {}
    for problem_id in raw_problem_results:
        raw_results = raw_problem_results[problem_id]
        ys = raw_results["objectives"]["y"]
        xs = raw_results["parameters"]
        fs = None
        cs = None
        if n_features > 0:
            fs = raw_results["features"]
        if n_constraints > 0:
            cs = raw_results["constraints"]
        prev_best_index = np.argmax(ys, axis=0)
        prev_best_dict[problem_id] = (ys[prev_best_index], xs[prev_best_index])
        for i in range(ys.shape[0]):
            x = list(xs[i])
            result = dlib.function_evaluation(x=x, y=ys[i])
            evals[problem_id].append(result)
            if fs is not None:
                feature_evals[problem_id].append(fs[i])
            if cs is not None:
                constraint_evals[problem_id].append(cs[i])

    return raw_spec, spec, evals, feature_evals, constraint_evals, info, prev_best_dict


def init_from_h5(file_path, param_names, opt_id, logger):
    # Load progress and settings from file, then compare each
    # restored setting with settings specified by args (if any)
    (
        old_raw_spec,
        old_spec,
        old_evals,
        old_feature_evals,
        old_constraint_evals,
        info,
        prev_best,
    ) = h5_load_all(file_path, opt_id)
    saved_params = info["params"]
    for problem_id in old_evals:
        n_old_evals = len(old_evals[problem_id])
        logger.info(
            f"Restored {n_old_evals} trials for problem {problem_id}, prev best: "
            f"{prev_best[problem_id][0]}@"
            f"{list(zip(saved_params, prev_best[problem_id][1]))}"
        )
    if (param_names is not None) and param_names != saved_params:
        # Switching params being optimized over would throw off Dlib.
        # Must use restore params from specified
        logger.warning(
            f"Saved params {saved_params} differ from currently specified "
            f"{param_names}. Using saved."
        )
    params = saved_params
    raw_spec = old_raw_spec
    is_int, lo_bounds, hi_bounds = raw_spec
    if len(params) != len(is_int):
        raise ValueError(f"Params {params} and spec {raw_spec} are of different length")

    feature_types = info["feature_types"]
    constraint_names = info.get("constraints", None)
    eps = info["solver_epsilon"]
    noise_mag = info["relative_noise_magnitude"]
    problem_parameters = info["problem_parameters"]
    problem_ids = info["problem_ids"] if "problem_ids" in info else None

    return (
        old_evals,
        old_feature_evals,
        old_constraint_evals,
        params,
        feature_types,
        constraint_names,
        is_int,
        lo_bounds,
        hi_bounds,
        eps,
        noise_mag,
        problem_parameters,
        problem_ids,
    )


def save_to_h5(
    opt_id: str,
    problem_ids: Set[int],
    has_problem_ids: bool,
    feature_dtypes: List[
        Union[Tuple[str, Tuple[Type[int], int]], Tuple[str, Tuple[Type[float], int]]]
    ],
    constraint_names: Optional[List[str]],
    param_names: List[str],
    spec: function_spec,
    evals: Dict[int, List[function_evaluation]],
    feature_evals: Dict[int, List[ndarray]],
    constraint_evals: Optional[Dict[int, List[ndarray]]],
    solver_epsilon: float,
    relative_noise_magnitude: float,
    problem_parameters: Dict[str, float],
    metadata: Optional[Dict[str, Any]],
    fpath: str,
    logger: logging.Logger,
) -> None:
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    if opt_id not in f.keys():
        h5_init_types(
            f,
            opt_id,
            feature_dtypes,
            constraint_names,
            param_names,
            problem_parameters,
            spec,
        )
        opt_grp = h5_get_group(f, opt_id)
        if metadata is not None:
            opt_grp["metadata"] = metadata
        opt_grp["solver_epsilon"] = solver_epsilon
        opt_grp["relative_noise_magnitude"] = relative_noise_magnitude
        if has_problem_ids:
            opt_grp["problem_ids"] = np.asarray(list(problem_ids), dtype=np.int32)

    opt_grp = h5_get_group(f, opt_id)
    for problem_id in problem_ids:
        prob_evals = evals[problem_id]
        if not (len(prob_evals) > 0):
            continue

        prob_features = None
        if feature_evals is not None:
            prob_features = feature_evals[problem_id]
        prob_constraints = None
        if constraint_evals is not None:
            prob_constraints = constraint_evals[problem_id]
        opt_prob = h5_get_group(opt_grp, f"{problem_id}")

        logger.info(
            f"Saving {len(prob_evals)} trials for problem id {problem_id} to {fpath}."
        )

        dset = h5_get_dataset(
            opt_prob, "objectives", maxshape=(None,), dtype=opt_grp["objective_type"]
        )
        data = np.array(
            [tuple([eeval.y]) for eeval in prob_evals], dtype=opt_grp["objective_type"]
        )
        h5_concat_dataset(dset, data)

        dset = h5_get_dataset(
            opt_prob,
            "parameters",
            maxshape=(None,),
            dtype=opt_grp["parameter_space_type"],
        )
        data = np.array(
            [tuple(eeval.x) for eeval in prob_evals],
            dtype=opt_grp["parameter_space_type"],
        )
        h5_concat_dataset(dset, data)

        if prob_features is not None:
            dset = h5_get_dataset(
                opt_prob, "features", maxshape=(None,), dtype=opt_grp["feature_type"]
            )
            data = np.concatenate(prob_features, axis=None)
            h5_concat_dataset(dset, data)
        if prob_constraints is not None:
            dset = h5_get_dataset(
                opt_prob,
                "constraints",
                maxshape=(None,),
                dtype=opt_grp["constraint_type"],
            )
            data = np.concatenate(prob_constraints, axis=None)
            h5_concat_dataset(dset, data)

    f.close()


def eval_obj_fun_sp(
    obj_fun: Callable,
    pp: Dict[str, float],
    space_params: List[str],
    is_int: List[bool],
    problem_id: int,
    i: int,
    space_vals: Dict[int, List[float]],
) -> Dict[int, float]:
    """
    Objective function evaluation (single problem).
    """

    this_space_vals = space_vals[problem_id]
    for j, key in enumerate(space_params):
        pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]

    result = obj_fun(pp, pid=i)
    return {problem_id: result}


def eval_obj_fun_mp(obj_fun, pp, space_params, is_int, problem_ids, i, space_vals):
    """
    Objective function evaluation (multiple problems).
    """

    mpp = {}
    for problem_id in problem_ids:
        this_pp = copy.deepcopy(pp)
        this_space_vals = space_vals[problem_id]
        for j, key in enumerate(space_params):
            this_pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]
        mpp[problem_id] = this_pp

    result_dict = obj_fun(mpp, pid=i)
    return result_dict


def gfsinit(
    gfsopt_params: Dict[str, Union[str, Dict[str, float], Dict[str, List[float]], int]],
    worker: Optional[int] = None,
    verbose: bool = False,
) -> DistGFSOptimizer:
    objfun = None
    objfun_module = gfsopt_params.get("obj_fun_module", "__main__")
    objfun_name = gfsopt_params.get("obj_fun_name", None)
    if distwq.is_worker:
        if objfun_name is not None:
            if objfun_module not in sys.modules:
                importlib.import_module(objfun_module)

            objfun = eval(objfun_name, sys.modules[objfun_module].__dict__)
        else:
            objfun_init_module = gfsopt_params.get("obj_fun_init_module", "__main__")
            objfun_init_name = gfsopt_params.get("obj_fun_init_name", None)
            objfun_init_args = gfsopt_params.get("obj_fun_init_args", None)
            if objfun_init_name is None:
                raise RuntimeError("distgfs.gfsinit: objfun is not provided")
            if objfun_init_module not in sys.modules:
                importlib.import_module(objfun_init_module)
            objfun_init = eval(
                objfun_init_name, sys.modules[objfun_init_module].__dict__
            )
            objfun = objfun_init(**objfun_init_args, worker=worker)
    else:
        ctrl_init_fun_module = gfsopt_params.get(
            "controller_init_fun_module", "__main__"
        )
        ctrl_init_fun_name = gfsopt_params.get("controller_init_fun_name", None)
        ctrl_init_fun_args = gfsopt_params.get("controller_init_fun_args", {})
        if ctrl_init_fun_name is not None:
            if ctrl_init_fun_module not in sys.modules:
                importlib.import_module(ctrl_init_fun_module)
                ctrl_init_fun = eval(
                    ctrl_init_fun_name, sys.modules[ctrl_init_fun_module].__dict__
                )
            ctrl_init_fun(**ctrl_init_fun_args)
        reducefun_module = gfsopt_params.get("reduce_fun_module", "__main__")
        reducefun_name = gfsopt_params.get("reduce_fun_name", None)
        reducefun_args = gfsopt_params.get("reduce_fun_args", {})
        if reducefun_module not in sys.modules:
            importlib.import_module(reducefun_module)
        if reducefun_name is not None:
            reducefun = eval(reducefun_name, sys.modules[reducefun_module].__dict__)
            gfsopt_params["reduce_fun"] = reducefun
            gfsopt_params["reduce_fun_args"] = reducefun_args
    gfsopt_params["obj_fun"] = objfun
    gfsopt = DistGFSOptimizer(**gfsopt_params, verbose=verbose)
    gfsopt_dict[gfsopt.opt_id] = gfsopt
    return gfsopt


def gfsctrl(
    controller: distwq.MPIController,
    gfsopt_params: Dict[str, Union[str, Dict[str, float], Dict[str, List[float]], int]],
    verbose: bool = False,
) -> None:
    """Controller for distributed GFS optimization."""
    logger = logging.getLogger(gfsopt_params["opt_id"])
    if verbose:
        logger.setLevel(logging.INFO)

    n_max_tasks = gfsopt_params.get("n_max_tasks", None)
    if (n_max_tasks is None) or (n_max_tasks < 1):
        gfsopt_params["n_max_tasks"] = distwq.n_workers

    gfsopt = gfsinit(gfsopt_params)
    logger.info(f"Optimizing for {gfsopt.n_iter} iterations...")
    iter_count = 0
    task_ids = []
    n_tasks = 0
    while iter_count < gfsopt.n_iter:
        controller.process()

        if (iter_count > 0) and gfsopt.save and (iter_count % gfsopt.save_iter == 0):
            gfsopt.save_evals()

        if len(task_ids) > 0:
            rets = controller.probe_all_next_results()
            for ret in rets:
                task_id, res = ret
                gfsopt.update_result_value(task_id, res)
                task_ids.remove(task_id)
                iter_count += 1

        if (n_tasks < gfsopt.n_iter) and (len(task_ids) == 0):
            while len(task_ids) < gfsopt.n_max_tasks:
                vals_dict = {}
                eval_req_dict = {}
                for problem_id in gfsopt.problem_ids:
                    eval_req = gfsopt.optimizer_dict[problem_id].get_next_x()
                    eval_req_dict[problem_id] = eval_req
                    vals = list(eval_req.x)
                    vals_dict[problem_id] = vals
                task_id = controller.submit_call(
                    "eval_fun",
                    module_name="distgfs",
                    args=(
                        gfsopt.opt_id,
                        iter_count,
                        vals_dict,
                    ),
                )
                task_ids.append(task_id)
                n_tasks += 1
                for problem_id in gfsopt.problem_ids:
                    gfsopt.evals[problem_id][task_id] = eval_req_dict[problem_id]

    if gfsopt.save:
        gfsopt.save_evals()
    controller.info()


def gfswork(
    worker: distwq.MPIWorker,
    gfsopt_params: Dict[
        str,
        Union[
            str,
            Dict[str, float],
            Dict[str, List[float]],
            int,
            List[
                Union[
                    Tuple[str, Tuple[Type[int], int]],
                    Tuple[str, Tuple[Type[float], int]],
                ]
            ],
        ],
    ],
    verbose: bool = False,
) -> None:
    """Worker for distributed GFS optimization."""
    gfsinit(gfsopt_params, worker=worker, verbose=verbose)


def eval_fun(opt_id: str, *args) -> Dict[int, float]:
    return gfsopt_dict[opt_id].eval_fun(*args)


def run(
    gfsopt_params: Dict[str, Union[str, Dict[str, float], Dict[str, List[float]], int]],
    collective_mode: str = "gather",
    spawn_workers: bool = False,
    sequential_spawn: bool = False,
    spawn_startup_wait: Optional[int] = None,
    max_workers: int = -1,
    nprocs_per_worker: int = 1,
    spawn_executable: Optional[str] = None,
    spawn_args: List[Any] = [],
    verbose: bool = False,
) -> Tuple[List[Tuple[str, float]], float]:
    if distwq.is_controller:
        distwq.run(
            fun_name="gfsctrl",
            module_name="distgfs",
            verbose=verbose,
            args=(
                gfsopt_params,
                verbose,
            ),
            max_workers=max_workers,
            worker_grouping_method="spawn" if spawn_workers else "split",
            broker_is_worker=True,
            sequential_spawn=sequential_spawn,
            spawn_startup_wait=spawn_startup_wait,
            spawn_executable=spawn_executable,
            spawn_args=spawn_args,
            nprocs_per_worker=nprocs_per_worker,
            collective_mode=collective_mode,
        )
        opt_id = gfsopt_params["opt_id"]
        gfsopt = gfsopt_dict[opt_id]
        gfsopt.print_best()
        return gfsopt.get_best()
    else:
        if "file_path" in gfsopt_params:
            del gfsopt_params["file_path"]
        if "save" in gfsopt_params:
            del gfsopt_params["save"]

        distwq.run(
            fun_name="gfswork",
            module_name="distgfs",
            broker_is_worker=not spawn_workers,
            broker_fun_name=gfsopt_params.get("broker_fun_name", None),
            broker_module_name=gfsopt_params.get("broker_module_name", None),
            verbose=verbose,
            args=(
                gfsopt_params,
                verbose,
            ),
            max_workers=max_workers,
            worker_grouping_method="spawn" if spawn_workers else "split",
            sequential_spawn=sequential_spawn,
            spawn_startup_wait=spawn_startup_wait,
            spawn_executable=spawn_executable,
            spawn_args=spawn_args,
            nprocs_per_worker=nprocs_per_worker,
            collective_mode=collective_mode,
        )
        return None
