import os, sys, importlib, logging, pprint
from functools import partial
import numpy as np  
import dlib
import distwq

logger = logging.getLogger(__name__)
try:
    import h5py
except ImportError as e:
    logger.warning('distgfs: unable to import h5py: %s' % str(e))

gfsopt_dict = {}


class DistGFSOptimizer():
    def __init__(
        self,
        opt_id,
        obj_fun,
        reduce_fun=None,
        problem_parameters=None,
        space=None,
        solver_epsilon=None,
        relative_noise_magnitude=None,
        n_iter=100,
        nprocs_per_worker=1,
        save_iter=10,
        file_path=None,
        save=False,
        **kwargs
    ):
        """
        `Creates an optimizer based on the Global Function Search
        <http://dlib.net/optimization.html#global_function_search>`_
        (GFS) optimizer in dlib. Supports distributed optimization
        runs via mpi4py. Based on GFSOPtimizer by https://github.com/tsoernes

        :param dict problem_parameters: Problem parameters.
            All hyperparameters and their values for the objective
            function, including those not being optimized over. E.g: ``{'beta': 0.44}``.
            Can be an empty dict.
            Can include hyperparameters being optimized over, but does not need to.
            If a hyperparameter is specified in both 'problem_parameters' and 'space', its value
            in 'problem_parameters' will be overridden.
        :param dict space: Hyperparameters to optimize over.
            Entries should be of the form:
            ``parameter: (Low_Bound, High_Bound)`` e.g:
            ``{'alpha': (0.65, 0.85), 'gamma': (1, 8)}``. If both bounds for a
            parameter are Ints, then only integers within the (inclusive) range
            will be sampled and tested.
        :param func obj_fun: function to maximize.
            Must take as argument every parameter specified in
            both 'problem_parameters' and 'space', in addition to 'pid',
            and return the result as float.
            'pid' specifies simulation run number.
            If you want to minimize instead,
            simply negate the result in the objective function before returning it.
        :param int n_iter: (optional) Number of times to sample and test params.
        :param int save_iter: (optional) How often to save progress.
        :param str file_path: (optional) File name for restoring and/or saving results and settings.
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
        """

        self.opt_id = opt_id
        
        # Verify inputs
        if file_path is None:
            if problem_parameters is None or space is None:
                raise ValueError(
                    "You must specify at least file name `file_path` or problem "
                    "parameters `problem_parameters` along with a hyperparameter space `space`."
                )
            if save:
                raise ValueError(
                    "If you want to save you must specify a file name `file_path`."
                )
        else:
            if not os.path.isfile(file_path):
                if problem_parameters is None or space is None:
                    raise FileNotFoundError(file_path)
        eps = solver_epsilon
        noise_mag = relative_noise_magnitude

        param_names, is_int, lo_bounds, hi_bounds = [], [], [], []
        if space is not None:
            for parm, conf in space.items():
                param_names.append(parm)
                lo, hi = conf
                is_int.append(type(lo) == int and type(hi) == int)
                lo_bounds.append(lo)
                hi_bounds.append(hi)
        old_evals = []
        if file_path is not None:
            if os.path.isfile(file_path):
                old_evals, param_names, is_int, lo_bounds, hi_bounds, eps, noise_mag, problem_parameters = \
                  init_from_h5(file_path, param_names, opt_id)
        eps = 0.0005 if eps is None else eps
        noise_mag = 0.001 if noise_mag is None else noise_mag
        spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)

        if len(old_evals) > 0:
            optimizer = dlib.global_function_search(
                [spec],
                initial_function_evals=[old_evals],
                relative_noise_magnitude=noise_mag
            )
        else:
            optimizer = dlib.global_function_search(
                [spec]
            )
            optimizer.set_relative_noise_magnitude(noise_mag)
        optimizer.set_solver_epsilon(eps)

        self.problem_parameters, self.param_names, self.optimizer, self.spec = problem_parameters, param_names, optimizer, spec
        self.eps, self.noise_mag, self.is_int = eps, noise_mag, is_int
        self.file_path, self.save = file_path, save

        self.n_iter = n_iter
        self.save_iter = save_iter

        self.eval_fun = partial(eval_obj_fun, obj_fun, self.problem_parameters, self.param_names, self.is_int)
        self.reduce_fun = reduce_fun
        
        self.evals = [[] for _ in range(n_iter)]


    def save_evals(self):
        """Store results of finished evals to file; print best eval"""
        finished_evals = self.optimizer.get_function_evaluations()[1][0]
        save_to_h5(self.opt_id, self.param_names, self.spec, finished_evals, 
                   self.eps, self.noise_mag, self.problem_parameters, self.file_path)

    def get_best(self):
        best_eval = self.optimizer.get_best_function_eval()
        prms = list(zip(self.param_names, list(best_eval[0])))
        res = best_eval[1]
        return prms, res
        
    def print_best(self):
        prms, res = self.get_best()
        logger.info(f"Best eval so far: {res}@{prms}")

def h5_get_group (h, groupname):
    if groupname in h.keys():
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g

def h5_get_dataset (g, dsetname, **kwargs):
    if dsetname in g.keys():
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset

def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize+len(data),)
    dset.resize(newshape)
    dset[dsize:] = data
    return dset

def h5_init_types(f, opt_id, param_names, problem_parameters, spec):
    
    opt_grp = h5_get_group(f, opt_id)

    # create an HDF5 enumerated type for the parameter label
    param_mapping = { name: idx for (idx, name) in
                      enumerate(sorted(problem_parameters.keys())) }

    dt = h5py.enum_dtype(param_mapping, basetype=np.uint16)
    opt_grp['parameter_enum'] = dt

    dt = np.dtype([("parameter", opt_grp['parameter_enum']),
                   ("value", np.float32)])
    opt_grp['problem_parameters_type'] = dt

    dset = h5_get_dataset(opt_grp, 'problem_parameters', maxshape=(len(param_mapping),),
                          dtype=opt_grp['problem_parameters_type'].dtype)
    dset.resize((len(param_mapping),))
    a = np.zeros(len(param_mapping), dtype=opt_grp['problem_parameters_type'].dtype)
    idx = 0
    for idx, (parm, val) in enumerate(problem_parameters.items()):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["value"] = val
    dset[:] = a
    
    dt = np.dtype([("parameter", opt_grp['parameter_enum']),
                   ("is_integer", np.bool),
                   ("lower", np.float32),
                   ("upper", np.float32)])
    opt_grp['parameter_spec_type'] = dt

    is_integer = np.asarray(spec.is_integer_variable, dtype=np.bool)
    upper = np.asarray(spec.upper, dtype=np.float32)
    lower = np.asarray(spec.lower, dtype=np.float32)
    
    dset = h5_get_dataset(opt_grp, 'parameter_spec', maxshape=(len(param_names),),
                          dtype=opt_grp['parameter_spec_type'].dtype)
    dset.resize((len(param_names),))
    a = np.zeros(len(param_names), dtype=opt_grp['parameter_spec_type'].dtype)
    idx = 0
    for parm, is_int, hi, lo in zip(param_names, is_integer, upper, lower):
        a[idx]["parameter"] = param_mapping[parm]
        a[idx]["is_integer"] = is_int
        a[idx]["lower"] = lo
        a[idx]["upper"] = hi
    dset[:] = a
    
def h5_load_raw(input_file, opt_id):
    ## N is number of trials
    ## M is number of hyperparameters
    f = h5py.File(input_file, 'r')
    opt_grp = h5_get_group(f, opt_id)
    solver_epsilon = opt_grp['solver_epsilon'][()]
    relative_noise_magnitude = opt_grp['relative_noise_magnitude'][()]

    parameter_enum_dict = h5py.check_enum_dtype(opt_grp['parameter_enum'].dtype)
    parameters_idx_dict = { parm: idx for parm, idx in parameter_enum_dict.items() }
    parameters_name_dict = { idx: parm for parm, idx in parameters_idx_dict.items() }
    
    problem_parameters = { parameters_name_dict[idx]: val
                           for idx, val in opt_grp['problem_parameters'] }
    parameter_spec_dict = { parameters_name_dict[spec[0]]: tuple(spec)[1:] 
                            for spec in iter(opt_grp['parameter_spec']) }
    
    M = len(parameter_spec_dict)
    raw_results = opt_grp['results'][:].reshape((-1,M+1)) # np.array of shape [N, M+1]
    f.close()
    
    param_names = []
    is_integer = []
    lower = []
    upper = []
    for parm, spec in parameter_spec_dict.items():
        param_names.append(parm)
        is_int, lo, hi = spec
        is_integer.append(is_int)
        lower.append(lo)
        upper.append(hi)
        

    raw_spec = (is_integer, lower, upper)
    info = { 'params': param_names,
             'solver_epsilon': solver_epsilon,
             'relative_noise_magnitude': relative_noise_magnitude,
             'problem_parameters': problem_parameters }
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
    raw_spec, raw_results, info = h5_load_raw(file_path, opt_id)
    is_integer, lo_bounds, hi_bounds = raw_spec
    spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_integer)
    evals = []
    prev_best = raw_results[np.argmax(raw_results, axis=0)[0]]
    for raw_result in raw_results:
        x = list(raw_result[1:])
        result = dlib.function_evaluation(x=x, y=raw_result[0])
        evals.append(result)
    return raw_spec, spec, evals, info, prev_best
    
def init_from_h5(file_path, param_names, opt_id):        
    # Load progress and settings from file, then compare each
    # restored setting with settings specified by args (if any)
    old_raw_spec, old_spec, old_evals, info, prev_best = h5_load_all(file_path, opt_id)
    saved_params = info['params']
    logger.info(
        f"Restored {len(old_evals)} trials, prev best: "
        f"{prev_best[0]}@{list(zip(saved_params, prev_best[1:]))}"
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
        raise ValueError(
            f"Params {params} and spec {raw_spec} are of different length"
            )
    eps = info['solver_epsilon']
    noise_mag = info['relative_noise_magnitude']
    problem_parameters = info['problem_parameters']

    return old_evals, params, is_int, lo_bounds, hi_bounds, eps, noise_mag, problem_parameters

def save_to_h5(opt_id, param_names, spec, evals, solver_epsilon, relative_noise_magnitude, problem_parameters, fpath):
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    if opt_id not in f.keys():
        h5_init_types(f, opt_id, param_names, problem_parameters, spec)
        opt_grp = h5_get_group(f, opt_id)
        opt_grp['solver_epsilon'] = solver_epsilon
        opt_grp['relative_noise_magnitude'] = relative_noise_magnitude
        dset = h5_get_dataset(opt_grp, 'results', dtype=np.float32, 
                              maxshape=(None,), compression=6)

    opt_grp = h5_get_group(f, opt_id)
    M = len(param_names)
    old_size = int(opt_grp['results'].shape[0] / (M+1))
    raw_results = np.zeros((len(evals)-old_size, len(evals[0].x) + 1))
    for i, eeval in enumerate(evals[old_size:]):
        raw_results[i][0] = eeval.y
        raw_results[i][1:] = list(eeval.x)
    logger.info(f"Saving {raw_results.shape[0]} trials to {fpath}.")
    
    h5_concat_dataset(opt_grp['results'], raw_results.ravel())
    
    f.close()
        
def eval_obj_fun(obj_fun, pp, space_params, is_int, i, space_vals):
    """
    """
    for j, key in enumerate(space_params):
        pp[key] = int(space_vals[j]) if is_int[j] else space_vals[j]

    result = obj_fun(**pp, pid=i)
    return result

def gfsinit(gfsopt_params):
    objfun_module = gfsopt_params.get('obj_fun_module', '__main__')
    objfun_name = gfsopt_params.get('obj_fun_name', None)
    if objfun_module not in sys.modules:
        importlib.import_module(objfun_module)
    objfun = eval(objfun_name, sys.modules[objfun_module].__dict__)
    gfsopt_params['obj_fun'] = objfun
    reducefun_module = gfsopt_params.get('reduce_fun_module', '__main__')
    reducefun_name = gfsopt_params.get('reduce_fun_name', None)
    if reducefun_module not in sys.modules:
        importlib.import_module(reducefun_module)
    if reducefun_name is not None:
        reducefun = eval(reducefun_name, sys.modules[reducefun_module].__dict__)
        gfsopt_params['reduce_fun'] = reducefun        
    gfsopt = DistGFSOptimizer(**gfsopt_params)
    gfsopt_dict[gfsopt.opt_id] = gfsopt
    return gfsopt
    
def gfsctrl(controller, gfsopt_params):
    """Controller for distributed GFS optimization."""
    gfsopt = gfsinit(gfsopt_params)
    logger.info("Optimizing for %d iterations..." % gfsopt.n_iter)
    n_steps = distwq.n_workers if distwq.workers_available else 1
    for i in range(0, gfsopt.n_iter):
        if (i > 0) and gfsopt.save and (i % gfsopt.save_iter == 0):
            gfsopt.save_evals()
        step_ids = []
        for j in range(0, n_steps):
            eval_req = gfsopt.optimizer.get_next_x()
            gfsopt.evals[i].append(eval_req)
            vals = list(eval_req.x)
            logger.info("optimization iteration %d: evaluating parameter coordinates %s" % (i, vals))
            step_ids.append(controller.submit_call("eval_fun", module_name="distgfs",
                                                   args=(gfsopt.opt_id, i, vals,)))
        for j, step_id in enumerate(step_ids):
            res = controller.get_result(step_id)
            
            if gfsopt.reduce_fun is None:
                rres = res
            else:
                rres = gfsopt.reduce_fun(res)
            print(rres)
            gfsopt.evals[i][j].set(rres)
    if gfsopt.save:
        gfsopt.save_evals()
    controller.info()

def gfswork(worker, gfsopt_params):
    """Worker for distributed GFS optimization."""
    gfsinit(gfsopt_params)

def eval_fun(opt_id, *args):
    return gfsopt_dict[opt_id].eval_fun(*args)

def run(gfsopt_params, spawn_workers=False, nprocs_per_worker=1, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)
    
    if distwq.is_controller:
        distwq.run(fun_name="gfsctrl", module_name="distgfs",
                   verbose=True, args=(gfsopt_params,),
                   spawn_workers=spawn_workers,
                   nprocs_per_worker=nprocs_per_worker)
        opt_id = gfsopt_params['opt_id']
        gfsopt = gfsopt_dict[opt_id]
        gfsopt.print_best()
        return gfsopt.get_best()
    else:
        distwq.run(fun_name="gfswork", module_name="distgfs",
                   verbose=True, args=(gfsopt_params,),
                   spawn_workers=spawn_workers,
                   nprocs_per_worker=nprocs_per_worker)


