import copy
import importlib
import logging
import os
import sys
import warnings
from collections.abc import Mapping
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

ProblemId = Union[int, str]

import distwq
import dlib
import numpy as np
from distwq.config import Config, LoggingConfig, ResourceConfig, WorkerGroupConfig
from distwq.future import (
    FIRST_COMPLETED,
    ControllerExitedError,
    DistributedEnvironment,
    DistributedFuture,
    TaskController,
    wait,
)
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

PARAMETER_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def validate_parameter_dtype(value: Any) -> np.dtype:
    """Convert a requested parameter storage precision to a NumPy dtype.

    :param value: ``"float32"``, ``"float64"``, the matching NumPy type or
        dtype, or None, which selects float64.
    :return: ``np.dtype(np.float32)`` or ``np.dtype(np.float64)``.
    :raises ValueError: if the value names any other type, including
        non-native byte order.
    """
    allowed = "'float32' or 'float64' (or the matching NumPy dtypes)"
    if value is None:
        return np.dtype(np.float64)
    try:
        dt = np.dtype(value)
    except TypeError as e:
        raise ValueError(
            f"parameter_dtype {value!r} is not a valid dtype; expected {allowed}"
        ) from e
    if dt not in PARAMETER_DTYPES or not dt.isnative:
        raise ValueError(
            f"parameter_dtype {value!r} is not supported; expected {allowed}"
        )
    return dt


def compare_problem_parameters(
    given: Dict[str, float],
    stored: Dict[str, float],
    param_names: List[str],
    stored_dtype: np.dtype,
) -> List[str]:
    """Describe how the given fixed problem parameters differ from stored ones.

    A value counts as equal when it matches the stored value after rounding
    to the stored precision. Two NaN values are equal. Names in
    'param_names' are ignored because the search sets them on every
    evaluation.

    :param given: Fixed parameters supplied by the caller.
    :param stored: Fixed parameters read from the checkpoint.
    :param param_names: Names of the parameters being optimized over.
    :param stored_dtype: Precision of the stored values.
    :return: One human-readable entry per difference; empty if none.
    """
    ignored = set(param_names)
    given_names = set(given) - ignored
    stored_names = set(stored) - ignored
    differences = []
    missing = sorted(stored_names - given_names)
    if missing:
        differences.append(f"not given: {missing}")
    extra = sorted(given_names - stored_names)
    if extra:
        differences.append(f"not stored: {extra}")
    for name in sorted(given_names & stored_names):
        try:
            rounded = np.asarray(given[name], dtype=stored_dtype)
            same = bool(np.array_equal(rounded, stored[name], equal_nan=True))
        except (TypeError, ValueError):
            same = False
        if not same:
            differences.append(
                f"{name}: given {given[name]!r}, stored {stored[name]!r}"
            )
    return differences


def h5_has_opt_id(file_path: str, opt_id: str) -> bool:
    """Return True if the HDF5 file at 'file_path' holds a group named 'opt_id'."""
    with h5py.File(file_path, "r") as f:
        return opt_id in f


def validate_inputs(
    problem_parameters: Dict[str, float],
    space: Dict[str, List[float]],
    save: bool,
    file_path: Optional[str],
    opt_id: Optional[str] = None,
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
        elif opt_id is not None and not h5_has_opt_id(file_path, opt_id):
            if problem_parameters is None or space is None:
                raise ValueError(
                    f"File {file_path} has no optimization group '{opt_id}'; "
                    "to start a new one, specify problem parameters "
                    "`problem_parameters` and a hyperparameter space `space`."
                )


def _group_result_dict(
    task_id: int,
    values: List[Any],
    reduce_fun: Optional[Callable],
    reduce_fun_args: Optional[Dict[str, Any]],
) -> Any:
    """Combine the per-process results of a worker group into one result.

    Processes that returned None are left out.  If one result remains it
    is used as it is, unless reduce_fun is given, in which case reduce_fun
    receives the list of remaining results.  More than one remaining
    result requires reduce_fun.
    """
    if not values:
        raise ValueError(f"task {task_id}: the worker group returned no results")
    present = [value for value in values if value is not None]
    if not present:
        raise ValueError(
            f"task {task_id}: every process of the worker group returned None "
            f"({len(values)} results), so there is no result to use"
        )
    if reduce_fun is not None:
        return reduce_fun(present, **(reduce_fun_args or {}))
    if len(present) > 1:
        raise ValueError(
            f"task {task_id} returned {len(present)} sub-worker results but no "
            "reduce_fun is specified. Provide reduce_fun to merge them."
        )
    return present[0]


def _task_result_dict(
    task_id: int,
    res: Any,
    reduce_fun: Optional[Callable] = None,
    reduce_fun_args: Optional[Dict[str, Any]] = None,
) -> Mapping:
    """Return the {problem_id: value} mapping held in the result of a task.

    The result is what the task's future returned.  A worker group gives a
    distwq.CollectiveTaskResult, whose values are combined by
    _group_result_dict(); a single process gives the objective's mapping
    itself.  reduce_fun is applied only to the results of a worker group.

    :raises ValueError: if a worker group gives no usable result, or more
        than one without reduce_fun.
    :raises TypeError: if the result is not a mapping.
    """
    source = "the objective function"
    match res:
        case distwq.CollectiveTaskResult():
            rres = _group_result_dict(
                task_id, list(res.values), reduce_fun, reduce_fun_args
            )
            if reduce_fun is not None:
                source = "reduce_fun"
        case _:
            rres = res
    if not isinstance(rres, Mapping):
        raise TypeError(
            f"task {task_id}: expected a dict of problem ids from {source}, "
            f"got {type(rres).__name__}"
        )
    return rres


class DistGFSOptimizer:
    def __init__(
        self,
        opt_id: str,
        obj_fun: Callable,
        reduce_fun: Optional[Callable] = None,
        reduce_fun_args: Dict[str, Any] = dict(),
        problem_ids: Optional[List[ProblemId]] = None,
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
        initial_evals: Optional[Dict[ProblemId, List[dlib.function_evaluation]]] = None,
        initial_feature_evals: Optional[
            Dict[ProblemId, List[Tuple[int, ndarray]]]
        ] = None,
        parameter_dtype: Union[str, np.dtype, type, None] = "float64",
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
            When resuming from 'file_path', these values are used as given and
            compared with the stored ones; a warning is issued if they differ.
            If omitted on resume, the stored values are used.
        :param dict space: Hyperparameters to optimize over.
            Entries should be of the form:
            ``parameter: (Low_Bound, High_Bound)`` e.g:
            ``{'alpha': (0.65, 0.85), 'gamma': (1, 8)}``. If both bounds for a
            parameter are Ints, then only integers within the (inclusive) range
            will be sampled and tested.
        :param func reduce_fun: function to combine the results of one
            evaluation computed by a worker group. distgfs.run() always
            uses worker groups, of ``nprocs_per_worker`` processes, so
            with run() it is called for every result. It receives the list of
            the group's results, leaving out processes that returned None,
            followed by the keyword arguments in 'reduce_fun_args', and must
            return a single result of the form ``{problem_id: value}``. It is
            not called for results from a single process. Without it, a
            group must produce exactly one result that is not None.
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
        :param dict initial_evals: (optional) Per-problem_id list of
            ``dlib.function_evaluation(x=..., y=...)`` objects for points
            already evaluated by the caller before this optimizer was
            constructed (dlib has no API to request evaluation of a
            caller-specified x through the normal search loop, so seed points
            must already carry their result). ``x`` must be ordered the same
            as 'space' (or, if 'file_path' is also given and is resumed, the
            same as the checkpoint's saved parameter order). Merged with, not
            replacing, any evaluations resumed from 'file_path'.
        :param dict initial_feature_evals: (optional) Per-problem_id list of
            ``(placeholder_task_id, feature_ndarray)`` tuples giving the
            feature vectors for the corresponding entries in 'initial_evals'
            (index i in both lists must describe the same seed point). Only
            valid together with 'feature_dtypes'.
        :param str parameter_dtype: (optional) Precision used to store
            evaluated parameters, search bounds and fixed problem parameters
            in a new checkpoint: ``"float64"`` (default) or ``"float32"``.
            The precision of a checkpoint is fixed when its 'opt_id' group is
            created; a resumed checkpoint keeps its own precision, with a
            warning if it differs from this value.
        """

        self.opt_id = opt_id
        self.verbose = verbose

        self.logger = logging.getLogger(opt_id)
        if self.verbose:
            self.logger.setLevel(logging.INFO)

        self.parameter_dtype = validate_parameter_dtype(parameter_dtype)
        validate_inputs(problem_parameters, space, save, file_path, opt_id)
        if save and save_iter < 1:
            raise ValueError(
                f"save_iter must be at least 1 when saving, not {save_iter}"
            )

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
        if (
            (file_path is not None)
            and os.path.isfile(file_path)
            and h5_has_opt_id(file_path, opt_id)
        ):
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
                stored_problem_parameters,
                problem_ids,
                file_parameter_dtype,
            ) = init_from_h5(file_path, param_names, opt_id, self.logger)
            if file_parameter_dtype != self.parameter_dtype:
                msg = (
                    f"Checkpoint {file_path} stores parameters for opt_id "
                    f"'{opt_id}' as {file_parameter_dtype}, not the requested "
                    f"{self.parameter_dtype}. New evaluations will be stored as "
                    f"{file_parameter_dtype}; use a new file_path or opt_id to "
                    f"store them as {self.parameter_dtype}."
                )
                self.logger.warning(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                self.parameter_dtype = file_parameter_dtype
            if problem_parameters is None:
                problem_parameters = stored_problem_parameters
            else:
                differences = compare_problem_parameters(
                    problem_parameters,
                    stored_problem_parameters,
                    param_names,
                    file_parameter_dtype,
                )
                if differences:
                    msg = (
                        f"problem_parameters for opt_id '{opt_id}' differ from "
                        f"those stored in {file_path}: {'; '.join(differences)}. "
                        "Using the given values; the stored values describe "
                        "the earlier evaluations."
                    )
                    self.logger.warning(msg)
                    warnings.warn(msg, UserWarning, stacklevel=2)
        problem_parameters = dict(problem_parameters)

        self.feature_dtypes = feature_dtypes
        self.feature_names = (
            None if feature_dtypes is None else [dtype[0] for dtype in feature_dtypes]
        )

        self.constraint_names = constraint_names

        spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)

        has_problem_ids = problem_ids is not None
        if not has_problem_ids:
            problem_ids = set([0])
        elif len(problem_ids) == 0:
            raise ValueError(
                "problem_ids must name at least one problem; "
                "omit it to optimize a single problem"
            )

        for label, seed_dict in (
            ("initial_evals", initial_evals),
            ("initial_feature_evals", initial_feature_evals),
        ):
            if seed_dict is None:
                continue
            unknown = set(seed_dict.keys()) - set(problem_ids)
            if unknown:
                raise ValueError(
                    f"{label} contains problem_id(s) {unknown} not present "
                    f"in problem_ids {problem_ids}"
                )

        n_resumed_evals = {pid: len(evals) for pid, evals in old_evals.items()}

        if initial_evals:
            for problem_id, evals in initial_evals.items():
                for ev in evals:
                    ev_x = list(ev.x)
                    if len(ev_x) != len(param_names):
                        raise ValueError(
                            f"initial_evals for problem {problem_id}: eval x "
                            f"has {len(ev_x)} dims, expected "
                            f"{len(param_names)} ({param_names})"
                        )
                old_evals[problem_id] = old_evals.get(problem_id, []) + list(evals)

        n_saved_evals = {pid: 0 for pid in problem_ids}
        n_saved_features = {pid: 0 for pid in problem_ids}
        n_saved_constraints = {pid: 0 for pid in problem_ids}
        optimizer_dict = {}
        for problem_id in problem_ids:
            if problem_id in old_evals and old_evals[problem_id]:
                optimizer = dlib.global_function_search(
                    [spec],
                    initial_function_evals=[old_evals[problem_id]],
                    relative_noise_magnitude=noise_mag,
                )
                n_saved_evals[problem_id] = n_resumed_evals.get(problem_id, 0)
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
            if initial_feature_evals:
                for problem_id, feats in initial_feature_evals.items():
                    n_seeded_evals = len(old_evals.get(problem_id, [])) - (
                        n_resumed_evals.get(problem_id, 0)
                    )
                    if len(feats) != n_seeded_evals:
                        raise ValueError(
                            f"initial_feature_evals for problem {problem_id} "
                            f"has {len(feats)} entries, expected "
                            f"{n_seeded_evals} to match initial_evals"
                        )
                    self.feature_evals[problem_id] = (
                        list(feats) + self.feature_evals[problem_id]
                    )
        elif initial_feature_evals:
            raise ValueError(
                "initial_feature_evals given but no feature_dtypes configured"
            )
        self.constraint_evals = None
        if self.constraint_names is not None:
            self.constraint_evals = {problem_id: [] for problem_id in problem_ids}

        self.has_problem_ids = has_problem_ids
        self.problem_ids = problem_ids
        self.per_problem_tasks = has_problem_ids and kwargs.get(
            "per_problem_tasks", False
        )

    def save_evals(self) -> None:
        """Store results of finished evals to file; print best eval"""
        finished_feature_evals = None
        finished_constraint_evals = None
        finished_evals = {
            problem_id: self.optimizer_dict[problem_id].get_function_evaluations()[1][
                0
            ][self.n_saved_evals[problem_id] :]
            for problem_id in self.problem_ids
        }
        if self.feature_dtypes is not None:
            finished_feature_evals = {
                problem_id: list(
                    [
                        x[1]
                        for x in self.feature_evals[problem_id][
                            self.n_saved_features[problem_id] :
                        ]
                    ]
                )
                for problem_id in self.problem_ids
            }
            for problem_id in self.problem_ids:
                self.n_saved_features[problem_id] += len(
                    finished_feature_evals[problem_id]
                )
        if self.constraint_names is not None:
            finished_constraint_evals = {
                problem_id: list(
                    [
                        x[1]
                        for x in self.constraint_evals[problem_id][
                            self.n_saved_constraints[problem_id] :
                        ]
                    ]
                )
                for problem_id in self.problem_ids
            }
            for problem_id in self.problem_ids:
                self.n_saved_constraints[problem_id] += len(
                    finished_constraint_evals[problem_id]
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
            parameter_dtype=self.parameter_dtype,
        )

        for problem_id in self.problem_ids:
            self.n_saved_evals[problem_id] += len(finished_evals[problem_id])

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
        self,
        task_id: int,
        res: Any,
        expected_ids: Optional[Tuple[ProblemId, ...]] = None,
    ) -> None:
        """Pass the result of task 'task_id' to the optimizer of each problem.

        :param res: The result returned by the task's future: the
            objective's mapping, or a CollectiveTaskResult from a worker
            group.  See _task_result_dict().
        :param expected_ids: (optional) The problem ids the task was
            submitted for.  If given, the result must hold exactly these
            ids; otherwise ValueError is raised before any value is set.
        """
        rres = _task_result_dict(task_id, res, self.reduce_fun, self.reduce_fun_args)
        if expected_ids is not None:
            missing = set(expected_ids) - set(rres)
            unexpected = set(rres) - set(expected_ids)
            if missing or unexpected:
                problems = []
                if missing:
                    problems.append(f"missing ids {sorted(map(str, missing))}")
                if unexpected:
                    problems.append(f"unexpected ids {sorted(map(str, unexpected))}")
                raise ValueError(
                    f"update_result_value: task {task_id} was submitted for "
                    f"problem ids {sorted(map(str, expected_ids))}, but its "
                    f"result has {' and '.join(problems)}"
                )
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
    parameter_dtype: Union[str, np.dtype, type] = np.float64,
) -> None:
    """Create the named types and settings datasets of group '/{opt_id}'.

    'parameter_dtype' sets the precision of evaluated parameters, search
    bounds and fixed problem parameters for the lifetime of the group.
    """
    parameter_dtype = validate_parameter_dtype(parameter_dtype)
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
    # Store objectives as float64; dlib's global_function_search works in double
    # precision throughout.
    dt = np.dtype({"names": objective_names, "formats": [np.float64]})
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

    dt = np.dtype(
        [("parameter", opt_grp["parameter_enum"]), ("value", parameter_dtype)]
    )
    opt_grp["problem_parameters_type"] = dt

    dset = h5_get_dataset(
        opt_grp,
        "problem_parameters",
        maxshape=(len(param_mapping),),
        dtype=opt_grp["problem_parameters_type"].dtype,
    )
    dset.resize((len(problem_parameters),))
    a = np.zeros(
        len(problem_parameters), dtype=opt_grp["problem_parameters_type"].dtype
    )
    idx = 0
    for idx, (parm, val) in enumerate(problem_parameters.items()):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["value"] = val
    dset[:] = a

    dt = np.dtype(
        [
            ("parameter", opt_grp["parameter_enum"]),
            ("is_integer", bool),
            ("lower", parameter_dtype),
            ("upper", parameter_dtype),
        ]
    )
    opt_grp["parameter_spec_type"] = dt

    is_integer = np.asarray(spec.is_integer_variable, dtype=bool)
    upper = np.asarray(spec.upper, dtype=parameter_dtype)
    lower = np.asarray(spec.lower, dtype=parameter_dtype)

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

    dt = np.dtype(
        {"names": param_names, "formats": [parameter_dtype] * len(param_names)}
    )
    opt_grp["parameter_space_type"] = dt


def h5_parameter_dtype(opt_grp: Group) -> np.dtype:
    """Return the precision used for parameter values in group 'opt_grp'.

    The precision is read from the fields of the committed type
    ``parameter_space_type``, or from the ``lower`` bound field of
    ``parameter_spec_type`` if the space type has no fields. The result is
    in native byte order.

    :raises ValueError: if the parameter fields do not share one type.
    """
    space_dt = opt_grp["parameter_space_type"].dtype
    field_dts = [space_dt.fields[name][0] for name in space_dt.names or ()]
    if not field_dts:
        field_dts = [opt_grp["parameter_spec_type"].dtype.fields["lower"][0]]
    found = {np.dtype(dt.type) for dt in field_dts}
    if len(found) != 1:
        raise ValueError(
            f"Group {opt_grp.name} stores parameters with mixed types "
            f"{sorted(str(dt) for dt in found)}"
        )
    return found.pop()


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

    # The first row for each name holds its value. Some files end with
    # zero-filled rows that repeat the name with index 0, so later rows must
    # not overwrite earlier ones.
    problem_parameters = {}
    for idx, val in opt_grp["problem_parameters"]:
        problem_parameters.setdefault(parameters_name_dict[idx], float(val))
    parameter_dtype = h5_parameter_dtype(opt_grp)
    parameter_specs = [
        (parameters_name_dict[spec[0]], tuple(spec)[1:])
        for spec in iter(opt_grp["parameter_spec"])
    ]

    problem_ids = None
    if "problem_ids" in opt_grp:
        raw_ids = opt_grp["problem_ids"][:]
        problem_ids = set()
        for pid in raw_ids:
            if isinstance(pid, (int, np.integer)):
                # Backward compat: old files stored problem_ids as int32.
                problem_ids.add(int(pid))
            else:
                pid_str = pid.decode("utf-8") if isinstance(pid, bytes) else str(pid)
                try:
                    # "0", "1" etc. round-trip back to int for backward compat.
                    problem_ids.add(int(pid_str))
                except ValueError:
                    problem_ids.add(pid_str)

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
        "parameter_dtype": parameter_dtype,
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
        # checkpoints written by older versions objective values as float32.
        # Upcast explicitly so that dlib receives doubles and Lipschitz-bound
        # estimates are not degraded.
        if ys.dtype == np.float32:
            warnings.warn(
                f"Checkpoint for opt_id '{opt_id}' stores objective values as "
                "float32 (written by an older version of distgfs).  Values are "
                "being upcast to float64; precision is limited for these "
                "restored evaluations.",
                UserWarning,
                stacklevel=3,
            )
            ys = ys.astype(np.float64)
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
    parameter_dtype = info["parameter_dtype"]

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
        parameter_dtype,
    )


def save_to_h5(
    opt_id: str,
    problem_ids: Set[ProblemId],
    has_problem_ids: bool,
    feature_dtypes: List[
        Union[Tuple[str, Tuple[Type[int], int]], Tuple[str, Tuple[Type[float], int]]]
    ],
    constraint_names: Optional[List[str]],
    param_names: List[str],
    spec: function_spec,
    evals: Dict[ProblemId, List[function_evaluation]],
    feature_evals: Dict[ProblemId, List[ndarray]],
    constraint_evals: Optional[Dict[ProblemId, List[ndarray]]],
    solver_epsilon: float,
    relative_noise_magnitude: float,
    problem_parameters: Dict[str, float],
    metadata: Optional[Dict[str, Any]],
    fpath: str,
    logger: logging.Logger,
    parameter_dtype: Union[str, np.dtype, type] = np.float64,
) -> None:
    """
    Save progress and settings to an HDF5 file 'fpath'.

    'parameter_dtype' sets the parameter precision only when the group
    '/{opt_id}' is created; later calls append in the group's own precision.
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
            parameter_dtype=parameter_dtype,
        )
        opt_grp = h5_get_group(f, opt_id)
        if metadata is not None:
            opt_grp["metadata"] = metadata
        opt_grp["solver_epsilon"] = solver_epsilon
        opt_grp["relative_noise_magnitude"] = relative_noise_magnitude
        if has_problem_ids:
            dt = h5py.string_dtype(encoding="utf-8")
            opt_grp.create_dataset(
                "problem_ids",
                data=np.array([str(pid) for pid in problem_ids], dtype=object),
                dtype=dt,
            )

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
    problem_id: ProblemId,
    i: int,
    space_vals: Dict[ProblemId, List[float]],
) -> Dict[ProblemId, float]:
    """
    Objective function evaluation (single problem).
    """

    this_space_vals = space_vals[problem_id]
    pp = dict(pp)
    for j, key in enumerate(space_params):
        pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]

    result = obj_fun(pp, pid=i)
    return {problem_id: result}


def eval_obj_fun_mp(obj_fun, pp, space_params, is_int, problem_ids, i, space_vals):
    """
    Objective function evaluation (multiple problems).
    """

    mpp = {}
    for problem_id in space_vals:
        this_pp = copy.deepcopy(pp)
        this_space_vals = space_vals[problem_id]
        for j, key in enumerate(space_params):
            this_pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]
        mpp[problem_id] = this_pp

    result_dict = obj_fun(mpp, pid=i)
    return result_dict


# gfsopt_params keys that gave the module of a *_fun_name before the names
# became paths, and the name key that now carries the module.
REMOVED_MODULE_KEYS = {
    "obj_fun_module": "obj_fun_name",
    "obj_fun_init_module": "obj_fun_init_name",
    "controller_init_fun_module": "controller_init_fun_name",
    "reduce_fun_module": "reduce_fun_name",
    "broker_module_name": "broker_fun_name",
}


def check_removed_module_keys(gfsopt_params: Dict[str, Any]) -> None:
    """Raise TypeError if gfsopt_params uses a removed *_module key.

    Functions are now named by a single path such as "mymodule.obj_fun",
    so a separate module key would be ignored.
    """
    found = sorted(key for key in REMOVED_MODULE_KEYS if key in gfsopt_params)
    if found:
        hints = ", ".join(
            f"{key} -> {REMOVED_MODULE_KEYS[key]}='{gfsopt_params[key]}.<name>'"
            for key in found
        )
        raise TypeError(
            f"distgfs: gfsopt_params keys {found} are no longer supported; give "
            f"the module as part of the function name instead ({hints})"
        )


def resolve_object_path(path: str) -> Any:
    """Return the object named by 'path'.

    A path has the form "module.name", for example "mymodule.obj_fun" or
    "mypackage.sub.obj_fun".  The longest leading part of the path that
    names an importable module is taken as the module, and the rest is
    looked up in it as a chain of attributes, so that "mymodule.Model.run"
    also works.  A path with no importable module part, such as
    "obj_fun", names an object in __main__.

    :raises TypeError: if 'path' is not a string.
    :raises ImportError: if 'path' is malformed or names nothing.
    """
    if not isinstance(path, str):
        raise TypeError(f"distgfs: expected a path string, got {path!r}")
    parts = path.split(".")
    if not all(part.isidentifier() for part in parts):
        raise ImportError(f"distgfs: '{path}' is not a valid module.name path")
    for i in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # Try a shorter prefix only if this module itself is missing,
            # not if it exists but fails to import one of its dependencies.
            if e.name is not None and (
                module_name == e.name or module_name.startswith(e.name + ".")
            ):
                continue
            raise
        return _attribute_path(module, parts[i:], path)
    return _attribute_path(sys.modules["__main__"], parts, path, in_main=True)


def _attribute_path(
    module: Any, names: List[str], path: str, in_main: bool = False
) -> Any:
    """Follow the attributes 'names' from 'module'.

    'path' and 'in_main' (True when no module part of the path could be
    imported) only shape the error message.
    """
    obj = module
    for name in names:
        try:
            obj = getattr(obj, name)
        except AttributeError as e:
            where = f"module '{module.__name__}' has no attribute '{'.'.join(names)}'"
            if in_main and len(names) > 1:
                where = f"no module '{names[0]}' could be imported, and {where}"
            raise ImportError(f"distgfs: cannot find '{path}': {where}") from e
    return obj


def resolve_callable(path: str, key: str) -> Callable:
    """Return the function named by 'path', given as gfsopt_params[key]."""
    obj = resolve_object_path(path)
    if not callable(obj):
        raise TypeError(
            f"distgfs: {key} '{path}' names a {type(obj).__name__}, not a function"
        )
    return obj


def _init_controller_side(gfsopt_params: Dict[str, Any]) -> None:
    """Do the setup that belongs to the controller.

    Calls the controller init function, if one is named, and resolves a
    named reduce_fun into gfsopt_params.
    """
    ctrl_init_fun_name = gfsopt_params.get("controller_init_fun_name", None)
    if ctrl_init_fun_name is not None:
        ctrl_init_fun = resolve_callable(ctrl_init_fun_name, "controller_init_fun_name")
        ctrl_init_fun(**gfsopt_params.get("controller_init_fun_args", {}))
    reducefun_name = gfsopt_params.get("reduce_fun_name", None)
    if reducefun_name is not None:
        gfsopt_params["reduce_fun"] = resolve_callable(
            reducefun_name, "reduce_fun_name"
        )
        gfsopt_params["reduce_fun_args"] = gfsopt_params.get("reduce_fun_args", {})


def _init_worker_side(
    gfsopt_params: Dict[str, Any], worker: Optional[Any] = None
) -> Callable:
    """Return the objective function, found by name or made by the named
    init function, which receives 'worker'."""
    objfun_name = gfsopt_params.get("obj_fun_name", None)
    if objfun_name is not None:
        return resolve_callable(objfun_name, "obj_fun_name")
    objfun_init_name = gfsopt_params.get("obj_fun_init_name", None)
    if objfun_init_name is None:
        raise RuntimeError("distgfs.gfsinit: objfun is not provided")
    objfun_init = resolve_callable(objfun_init_name, "obj_fun_init_name")
    return objfun_init(**(gfsopt_params.get("obj_fun_init_args") or {}), worker=worker)


def gfsinit(
    gfsopt_params: Dict[str, Union[str, Dict[str, float], Dict[str, List[float]], int]],
    worker: Optional[int] = None,
    verbose: bool = False,
) -> DistGFSOptimizer:
    """Create the optimizer for this rank and register it in gfsopt_dict.

    The controller runs the controller init function and resolves
    reduce_fun_name; a worker resolves the objective function.  A single
    process is both controller and worker, so it does both, controller
    part first.
    """
    check_removed_module_keys(gfsopt_params)
    if distwq.is_controller:
        _init_controller_side(gfsopt_params)
    objfun = None
    if distwq.is_worker:
        objfun = _init_worker_side(gfsopt_params, worker)
    gfsopt_params["obj_fun"] = objfun
    gfsopt = DistGFSOptimizer(**gfsopt_params, verbose=verbose)
    gfsopt_dict[gfsopt.opt_id] = gfsopt
    return gfsopt


def _ordered_problem_ids(problem_ids: Iterable[ProblemId]) -> List[ProblemId]:
    """Return the distinct problem ids in a reproducible order.

    Ids that can be compared with each other are sorted, so that the order
    does not depend on how a set happens to iterate.  Ids that cannot be
    compared, such as ints mixed with strings, keep their given order.
    """
    ids = list(dict.fromkeys(problem_ids))
    try:
        return sorted(ids)
    except TypeError:
        return ids


def _next_unit(n_submitted: List[int], n_iter: int, start: int) -> Optional[int]:
    """Return the first unit at or after 'start', wrapping around, that has
    fewer than 'n_iter' submitted tasks, or None if every unit is full."""
    n_units = len(n_submitted)
    for offset in range(n_units):
        unit = (start + offset) % n_units
        if n_submitted[unit] < n_iter:
            return unit
    return None


def _submit_task(
    controller: TaskController,
    gfsopt: "DistGFSOptimizer",
    problem_ids: Tuple[ProblemId, ...],
    index: int,
) -> DistributedFuture:
    """Submit one evaluation of the problems in 'problem_ids'.

    A new point is requested from the optimizer of each problem, and the
    request is recorded under the task id of the new future so that
    update_result_value() can pass the result back.  'index' reaches the
    objective function as 'pid'.  Returns the future.
    """
    eval_reqs = {
        problem_id: gfsopt.optimizer_dict[problem_id].get_next_x()
        for problem_id in problem_ids
    }
    vals_dict = {problem_id: list(req.x) for problem_id, req in eval_reqs.items()}
    future = controller.submit(eval_fun, gfsopt.opt_id, index, vals_dict)
    for problem_id, req in eval_reqs.items():
        gfsopt.evals[problem_id][future.task_id] = req
    return future


def _finished(in_flight: Dict[DistributedFuture, int]) -> List[DistributedFuture]:
    """Wait until at least one future in 'in_flight' is done; return the
    done ones in task id order."""
    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
    return sorted(done, key=lambda future: future.task_id)


def _gfsctrl_loop(controller: TaskController, gfsopt: "DistGFSOptimizer") -> None:
    """Submit evaluations and collect their results until the search is done.

    The work is divided into units; a unit is the set of problem ids that
    one task evaluates.  With per_problem_tasks every problem is a unit of
    its own, and a new task is submitted as soon as a slot is free, taking
    the units in turn.  Otherwise one unit holds every problem, and tasks
    are submitted in batches: a new batch starts once every task of the
    previous one has returned.

    Each unit receives exactly n_iter tasks.  At most n_max_tasks tasks,
    and at least one, are in flight at a time.  The task's submission
    index within its unit is passed to the objective function as 'pid'.
    With saving enabled, a checkpoint is written after every save_iter
    results; gfsctrl() writes the last one.

    If a task raises, the other results that arrived with it are still
    passed to the optimizers, and then the task's exception is raised.
    The failed point is not passed to its optimizer.
    """
    problem_ids = _ordered_problem_ids(gfsopt.problem_ids)
    rolling = gfsopt.per_problem_tasks
    if rolling:
        units = [(problem_id,) for problem_id in problem_ids]
    else:
        units = [tuple(problem_ids)]
    n_iter = max(0, gfsopt.n_iter)
    limit = max(1, gfsopt.n_max_tasks)
    total = n_iter * len(units)

    n_submitted = [0] * len(units)
    in_flight: Dict[DistributedFuture, int] = {}  # future -> unit index
    next_unit = 0
    n_done = 0
    last_saved = 0

    while n_done < total:
        if rolling or not in_flight:
            while len(in_flight) < limit:
                unit = _next_unit(n_submitted, n_iter, next_unit)
                if unit is None:
                    break
                future = _submit_task(
                    controller, gfsopt, units[unit], n_submitted[unit]
                )
                in_flight[future] = unit
                n_submitted[unit] += 1
                next_unit = (unit + 1) % len(units)
        if not in_flight:
            raise RuntimeError(
                f"distgfs: no tasks in flight, but only {n_done} of {total} "
                "results have been received"
            )

        error = None
        for future in _finished(in_flight):
            unit = in_flight.pop(future)
            try:
                res = future.result()
            except Exception as e:
                if error is None:
                    error = e
                continue
            gfsopt.update_result_value(future.task_id, res, expected_ids=units[unit])
            n_done += 1
        if error is not None:
            raise error

        if gfsopt.save and n_done < total and n_done - last_saved >= gfsopt.save_iter:
            gfsopt.save_evals()
            last_saved = n_done


def gfsctrl(
    controller: TaskController,
    gfsopt_params: Dict[str, Union[str, Dict[str, float], Dict[str, List[float]], int]],
    verbose: bool = False,
    n_workers: int = 0,
) -> "DistGFSOptimizer":
    """Controller for distributed GFS optimization.

    'n_workers' is the number of workers, or worker groups, of the
    environment; it sets the default n_max_tasks.  Returns the optimizer.

    If the search raises, for example because the objective function
    raised, a checkpoint with every result received so far is written
    (when saving is enabled) and the exception is raised again.
    """
    logger = logging.getLogger(gfsopt_params["opt_id"])
    if verbose:
        logger.setLevel(logging.INFO)

    gfsopt = gfsinit(gfsopt_params)
    # The default is computed from the optimizer, whose problem ids may
    # come from a resumed checkpoint rather than from gfsopt_params.
    if (gfsopt.n_max_tasks is None) or (gfsopt.n_max_tasks < 1):
        if gfsopt.per_problem_tasks:
            gfsopt.n_max_tasks = max(n_workers, len(gfsopt.problem_ids))
        else:
            gfsopt.n_max_tasks = n_workers
    logger.info(f"Optimizing for {gfsopt.n_iter} iterations...")
    try:
        _gfsctrl_loop(controller, gfsopt)
    except BaseException:
        if gfsopt.save:
            try:
                gfsopt.save_evals()
            except Exception:
                logger.exception("distgfs: unable to save a checkpoint after an error")
        raise
    if gfsopt.save:
        gfsopt.save_evals()
    controller.info()
    return gfsopt


def gfswork(
    worker: Any,
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
    """Worker init function: set up this process to evaluate the objective.

    distwq calls it, as worker_init, on every process that computes tasks,
    with the process's server as 'worker'.
    """
    gfsinit(gfsopt_params, worker=worker, verbose=verbose)


def gfsbroker(broker: Any, gfsopt_params: Dict[str, Any], verbose: bool = False) -> Any:
    """Broker init function that distgfs gives to distwq.

    distwq needs a function defined at module level, which a path such as
    "mymodule.Model.setup" does not name.  When gfsopt_params names a
    broker function in 'broker_fun_name', distgfs passes this function
    instead; it calls the named function with the arguments distwq gives a
    broker init function.
    """
    broker_fun = resolve_callable(gfsopt_params["broker_fun_name"], "broker_fun_name")
    return broker_fun(broker, gfsopt_params, verbose)


def eval_fun(opt_id: str, *args) -> Dict[ProblemId, float]:
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
) -> Optional[Tuple[List[Tuple[str, float]], float]]:
    """Run a distributed GFS optimization; every rank must call it.

    The controller rank runs the search, and the other ranks evaluate the
    objective function in worker groups of 'nprocs_per_worker' processes,
    split from the MPI world or, with 'spawn_workers', spawned by one
    broker rank each.  'max_workers' limits the number of ranks used as
    workers; ranks left out stay idle.  Without MPI, or on one rank, the
    controller evaluates the objective itself.

    Returns the best result found, as DistGFSOptimizer.get_best() gives it,
    on every rank, or None on idle ranks.  If the search raises on the
    controller, the exception is raised there again, after a checkpoint is
    written, and the other ranks return None.
    """
    check_removed_module_keys(gfsopt_params)
    broker_fun_name = gfsopt_params.get("broker_fun_name", None)
    if broker_fun_name is not None:
        if not spawn_workers:
            raise ValueError(
                "distgfs: broker_fun_name needs spawn_workers=True; split worker "
                "groups have no brokers that only relay calls"
            )
        # Checked on every rank, before any MPI call.
        resolve_callable(broker_fun_name, "broker_fun_name")
    # Only the controller saves results.
    worker_params = {
        key: value
        for key, value in gfsopt_params.items()
        if key not in ("file_path", "save")
    }
    worker_config = WorkerGroupConfig(
        grouping_method="spawn" if spawn_workers else "split",
        nprocs_per_worker=nprocs_per_worker,
        broker_is_worker=not spawn_workers,
        collective_mode=collective_mode,
        sequential_spawn=sequential_spawn,
        spawn_startup_wait=spawn_startup_wait,
        spawn_executable=spawn_executable,
        spawn_args=spawn_args,
    )
    config = Config(
        resources=ResourceConfig(max_workers=max_workers),
        logging=LoggingConfig(verbose=verbose),
    )
    best = None
    with DistributedEnvironment(
        config,
        worker_config,
        worker_init=gfswork,
        broker_init=gfsbroker if broker_fun_name is not None else None,
        init_args=(worker_params, verbose),
    ) as env:
        for controller in env.controller_session():
            gfsopt = gfsctrl(
                controller, gfsopt_params, verbose, n_workers=env.worker_count
            )
            gfsopt.print_best()
            best = gfsopt.get_best()
        try:
            best = env.broadcast(best)
        except ControllerExitedError:
            logging.getLogger(gfsopt_params["opt_id"]).warning(
                "distgfs: the controller stopped with an error; see its output"
            )
            best = None
    return best
