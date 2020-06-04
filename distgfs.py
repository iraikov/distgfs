import os, sys, importlib, logging, pprint, copy
from functools import partial
import numpy as np  
import dlib
import distwq

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
        verbose=False,
        reduce_fun=None,
        problem_ids=None,
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

        :param set problem_ids (optional): Set of problem ids.
            For solving sets of related problems with the same set of parameters.
            If this parameter is not None, it is expected that the objective function 
            will return a dictionary of the form { problem_id: value }
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
        self.verbose = verbose

        self.logger = logging.getLogger(opt_id)
        if self.verbose:
            self.logger.setLevel(logging.INFO)

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
        old_evals = {}
        if file_path is not None:
            if os.path.isfile(file_path):
                old_evals, param_names, is_int, lo_bounds, hi_bounds, eps, noise_mag, problem_parameters, problem_ids = \
                  init_from_h5(file_path, param_names, opt_id, self.logger)
        eps = 0.0005 if eps is None else eps
        noise_mag = 0.001 if noise_mag is None else noise_mag
        spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)

        has_problem_ids = (problem_ids is not None)
        if not has_problem_ids:
            problem_ids = set([0])
        
        optimizer_dict = {}
        for problem_id in problem_ids:
            if problem_id in old_evals:
                optimizer = dlib.global_function_search(
                    [spec],
                    initial_function_evals=[old_evals[problem_id]],
                    relative_noise_magnitude=noise_mag
                )
            else:
                optimizer = dlib.global_function_search(
                    [spec]
                )
                optimizer.set_relative_noise_magnitude(noise_mag)
            optimizer.set_solver_epsilon(eps)
            optimizer_dict[problem_id] = optimizer
        
        self.optimizer_dict = optimizer_dict
        
        self.problem_parameters, self.param_names, self.spec = problem_parameters, param_names, spec
        self.eps, self.noise_mag, self.is_int = eps, noise_mag, is_int
        self.file_path, self.save = file_path, save

        self.n_iter = n_iter
            
        self.save_iter = save_iter

        if has_problem_ids:
            self.eval_fun = partial(eval_obj_fun_mp, obj_fun, self.problem_parameters, self.param_names, self.is_int, problem_ids)
        else:
            self.eval_fun = partial(eval_obj_fun_sp, obj_fun, self.problem_parameters, self.param_names, self.is_int, 0)
            
        self.reduce_fun = reduce_fun
        
        self.evals = { problem_id: {} for problem_id in problem_ids }

        self.has_problem_ids = has_problem_ids
        self.problem_ids = problem_ids

    def save_evals(self):
        """Store results of finished evals to file; print best eval"""
        finished_evals = { problem_id: self.optimizer_dict[problem_id].get_function_evaluations()[1][0]
                           for problem_id in self.problem_ids }
        save_to_h5(self.opt_id, self.problem_ids, self.has_problem_ids,
                   self.param_names, self.spec, finished_evals, 
                   self.eps, self.noise_mag, self.problem_parameters, 
                   self.file_path, self.logger)

    def get_best(self):
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
        
    def print_best(self):
        best_results = self.get_best()
        if self.has_problem_ids:
            for problem_id in self.problem_ids:
                res, prms = best_results[problem_id]
                self.logger.info(f"Best eval so far for id {problem_id}: {res}@{prms}")
        else:
            res, prms = best_results
            self.logger.info(f"Best eval so far for: {res}@{prms}")
            

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

    param_keys = set(param_names)
    param_keys.update(problem_parameters.keys())
    # create an HDF5 enumerated type for the parameter label
    param_mapping = { name: idx for (idx, name) in
                      enumerate(param_keys) }

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
    for idx, (parm, is_int, hi, lo) in enumerate(zip(param_names, is_integer, upper, lower)):
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

    problem_ids = None
    if 'problem_ids' in opt_grp:
        problem_ids = set(opt_grp['problem_ids'])
    
    M = len(parameter_spec_dict)
    raw_results = {}
    if problem_ids is not None:
        for problem_id in problem_ids:
            raw_results[problem_id] = opt_grp['%d' % problem_id]['results'][:].reshape((-1,M+1)) # np.array of shape [N, M+1]
    else:
        raw_results[0] = opt_grp['%d' % 0]['results'][:].reshape((-1,M+1)) # np.array of shape [N, M+1]
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
             'problem_parameters': problem_parameters,
             'problem_ids': problem_ids }
    
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
    spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_integer)
    evals = { problem_id: [] for problem_id in raw_problem_results }
    prev_best_dict = {}
    for problem_id in raw_problem_results:
        raw_results = raw_problem_results[problem_id]
        prev_best_dict[problem_id] = raw_results[np.argmax(raw_results, axis=0)[0]]
        for raw_result in raw_results:
            x = list(raw_result[1:])
            result = dlib.function_evaluation(x=x, y=raw_result[0])
            evals[problem_id].append(result)
    return raw_spec, spec, evals, info, prev_best_dict
    
def init_from_h5(file_path, param_names, opt_id, logger):        
    # Load progress and settings from file, then compare each
    # restored setting with settings specified by args (if any)
    old_raw_spec, old_spec, old_evals, info, prev_best = h5_load_all(file_path, opt_id)
    saved_params = info['params']
    for problem_id in old_evals:
        n_old_evals = len(old_evals[problem_id])
        logger.info(
            f"Restored {n_old_evals} trials for problem {problem_id}, prev best: "
            f"{prev_best[problem_id][0]}@{list(zip(saved_params, prev_best[problem_id][1:]))}"
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
    problem_ids = info['problem_ids'] if 'problem_ids' in info else None

    return old_evals, params, is_int, lo_bounds, hi_bounds, eps, noise_mag, problem_parameters, problem_ids

def save_to_h5(opt_id, problem_ids, has_problem_ids, param_names, spec, evals, solver_epsilon, relative_noise_magnitude, problem_parameters, fpath, logger):
    """
    Save progress and settings to an HDF5 file 'fpath'.
    """

    f = h5py.File(fpath, "a")
    if opt_id not in f.keys():
        h5_init_types(f, opt_id, param_names, problem_parameters, spec)
        opt_grp = h5_get_group(f, opt_id)
        opt_grp['solver_epsilon'] = solver_epsilon
        opt_grp['relative_noise_magnitude'] = relative_noise_magnitude
        if has_problem_ids:
            opt_grp['problem_ids'] = np.asarray(list(problem_ids), dtype=np.int32)

    opt_grp = h5_get_group(f, opt_id)
    M = len(param_names)
    for problem_id in problem_ids:
        prob_evals = evals[problem_id]
        opt_prob = h5_get_group(opt_grp, '%d' % problem_id)
        dset = h5_get_dataset(opt_prob, 'results', maxshape=(None,),
                              dtype=np.float32) 
        old_size = int(dset.shape[0] / (M+1))
        raw_results = np.zeros((len(prob_evals)-old_size, len(prob_evals[0].x) + 1))
        for i, eeval in enumerate(prob_evals[old_size:]):
            raw_results[i][0] = eeval.y
            raw_results[i][1:] = list(eeval.x)
        logger.info(f"Saving {raw_results.shape[0]} trials for problem id %d to {fpath}." % problem_id)
        h5_concat_dataset(opt_prob['results'], raw_results.ravel())
    
    f.close()

    
def eval_obj_fun_sp(obj_fun, pp, space_params, is_int, problem_id, i, space_vals):
    """
    Objective function evaluation (single problem).
    """
    
    this_space_vals = space_vals[problem_id]
    for j, key in enumerate(space_params):
        pp[key] = int(this_space_vals[j]) if is_int[j] else this_space_vals[j]

    
    result = obj_fun(**pp, pid=i)
    return { problem_id: result }


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


def gfsinit(gfsopt_params, worker=None, verbose=False):
    objfun = None
    objfun_module = gfsopt_params.get('obj_fun_module', '__main__')
    objfun_name = gfsopt_params.get('obj_fun_name', None)
    if distwq.is_worker:
        if objfun_name is not None:
            if objfun_module not in sys.modules:
                importlib.import_module(objfun_module)
                
            objfun = eval(objfun_name, sys.modules[objfun_module].__dict__)
        else:
            objfun_init_module = gfsopt_params.get('obj_fun_init_module', '__main__')
            objfun_init_name = gfsopt_params.get('obj_fun_init_name', None)
            objfun_init_args = gfsopt_params.get('obj_fun_init_args', None)
            if objfun_init_name is None:
                raise RuntimeError("distgfs.gfsinit: objfun is not provided")
            if objfun_init_module not in sys.modules:
                importlib.import_module(objfun_init_module)
            objfun_init = eval(objfun_init_name, sys.modules[objfun_init_module].__dict__)
            objfun = objfun_init(**objfun_init_args, worker=worker)
            
    gfsopt_params['obj_fun'] = objfun
    reducefun_module = gfsopt_params.get('reduce_fun_module', '__main__')
    reducefun_name = gfsopt_params.get('reduce_fun_name', None)
    if reducefun_module not in sys.modules:
        importlib.import_module(reducefun_module)
    if reducefun_name is not None:
        reducefun = eval(reducefun_name, sys.modules[reducefun_module].__dict__)
        gfsopt_params['reduce_fun'] = reducefun        
    gfsopt = DistGFSOptimizer(**gfsopt_params, verbose=verbose)
    gfsopt_dict[gfsopt.opt_id] = gfsopt
    return gfsopt


def gfsctrl(controller, gfsopt_params, verbose=False):
    """Controller for distributed GFS optimization."""
    logger = logging.getLogger(gfsopt_params['opt_id'])
    if verbose:
        logger.setLevel(logging.INFO)
    gfsopt = gfsinit(gfsopt_params)
    logger.info("Optimizing for %d iterations..." % gfsopt.n_iter)
    iter_count = 0
    task_ids = []
    n_tasks = 0
    while iter_count < gfsopt.n_iter:
        controller.recv()
        
        if (iter_count > 0) and gfsopt.save and (iter_count % gfsopt.save_iter == 0):
            gfsopt.save_evals()

        if len(task_ids) > 0:
            task_id, res = controller.get_next_result()

            if gfsopt.reduce_fun is None:
                rres = res
            else:
                rres = gfsopt.reduce_fun(res)

            for problem_id in rres:
                eval_req = gfsopt.evals[problem_id][task_id]
                vals = list(eval_req.x)
                eval_req.set(rres[problem_id])
                logger.info("problem id %d: optimization iteration %d: parameter coordinates %s: %s" % (problem_id, iter_count, str(vals), str(rres[problem_id])))
                
            task_ids.remove(task_id)
            iter_count += 1
            
        while ((len(controller.ready_workers) > 0) or (not controller.workers_available)) and (n_tasks < gfsopt.n_iter):
            vals_dict = {}
            eval_req_dict = {}
            for problem_id in gfsopt.problem_ids:
                eval_req = gfsopt.optimizer_dict[problem_id].get_next_x()
                eval_req_dict[problem_id] = eval_req
                vals = list(eval_req.x)
                vals_dict[problem_id] = vals
            task_id = controller.submit_call("eval_fun", module_name="distgfs",
                                             args=(gfsopt.opt_id, iter_count, vals_dict,))
            task_ids.append(task_id)
            n_tasks += 1
            for problem_id in gfsopt.problem_ids:
                gfsopt.evals[problem_id][task_id] = eval_req_dict[problem_id]
                
    if gfsopt.save:
        gfsopt.save_evals()
    controller.info()

def gfswork(worker, gfsopt_params, verbose=False):
    """Worker for distributed GFS optimization."""
    gfsinit(gfsopt_params, worker=worker, verbose=verbose)

def eval_fun(opt_id, *args):
    return gfsopt_dict[opt_id].eval_fun(*args)

def run(gfsopt_params, spawn_workers=False, nprocs_per_worker=1, verbose=False):
    if distwq.is_controller:
        distwq.run(fun_name="gfsctrl", module_name="distgfs",
                   verbose=True, args=(gfsopt_params, verbose,),
                   spawn_workers=spawn_workers,
                   nprocs_per_worker=nprocs_per_worker)
        opt_id = gfsopt_params['opt_id']
        gfsopt = gfsopt_dict[opt_id]
        gfsopt.print_best()
        return gfsopt.get_best()
    else:
        if 'file_path' in gfsopt_params:
            del(gfsopt_params['file_path'])
        if 'save' in gfsopt_params:
            del(gfsopt_params['save'])
        distwq.run(fun_name="gfswork", module_name="distgfs",
                   verbose=True, args=(gfsopt_params, verbose, ),
                   spawn_workers=spawn_workers,
                   nprocs_per_worker=nprocs_per_worker)
        return None
        
