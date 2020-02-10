import sys, importlib, logging, pprint
from functools import partial
import numpy as np  
import dlib
import distwq

logger = logging.getLogger(__name__)

gfsopt_dict = {}


class GFSOptimizer():
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
        save_iter=30,
        fname=None,
        save=False,
        **kwargs
    ):
        """
        `Creates an optimizer based on the Global Function Search
        <http://dlib.net/optimization.html#global_function_search>`_
        (GFS) optimizer in dlib. Supports distributed optimization
        runs via mpi4py.

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
        :param str fname: File name for restoring and/or saving results,
            progress and settings.
        :param bool save: (optional) Save settings and progress periodically,
            on user quit (CTRL-C), and on completion.
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
        if fname is None:
            if problem_parameters is None or space is None:
                raise ValueError(
                    "You must specify at least file name `fname` or problem "
                    "parameters `problem_parameters` along with a hyperparameter space `space`."
                )
            if save:
                raise ValueError(
                    "If you want to save you must specify a file name `fname`."
                )
        else:
            if not os.path.isfile(fname):
                if problem_parameters is None or space is None:
                    raise FileNotFoundError(fname)
        eps = solver_epsilon
        noise_mag = relative_noise_magnitude

        params, is_int, lo_bounds, hi_bounds = [], [], [], []
        if space is not None:
            for parm, conf in space.items():
                params.append(parm)
                lo, hi = conf
                is_int.append(type(lo) == int and type(hi) == int)
                lo_bounds.append(lo)
                hi_bounds.append(hi)
        old_evals = []
        if fname is not None:
            try:
                # Load progress and settings from file, then compare each
                # restored setting with settings specified by args (if any)
                old_raw_spec, old_spec, old_evals, info, prev_best = _load(fname)
                saved_params = info['params']
                logger.info(
                    f"Restored {len(old_evals)} trials, prev best: "
                    f"{prev_best[0]}@{list(zip(saved_params, prev_best[1:]))}"
                )
                if params and params != saved_params:
                    # Switching params being optimized over would throw off Dlib.
                    # Must use restore params from specified
                    logger.info(
                        f"Saved params {saved_params} differ from currently specified "
                        f"{params}. Using saved."
                    )
                params = saved_params
                if is_int:
                    raw_spec = _cmp_and_choose(
                        'bounds', old_raw_spec, (is_int, lo_bounds, hi_bounds)
                    )
                else:
                    raw_spec = old_raw_spec
                is_int, lo_bounds, hi_bounds = raw_spec
                if len(params) != len(is_int):
                    raise ValueError(
                        f"Params {params} and spec {raw_spec} are of different length"
                    )
                eps = _cmp_and_choose('solver_epsilon', info['solver_epsilon'], eps)
                noise_mag = _cmp_and_choose(
                    'relative_noise_magnitude', info['relative_noise_magnitude'],
                    noise_mag
                )
                _, problem_parameters = _compare_pps(info['problem_parameters'], problem_parameters)
            except FileNotFoundError:
                # Create a new file
                pass
        eps = 0.0005 if eps is None else eps
        noise_mag = 0.001 if noise_mag is None else noise_mag
        spec = dlib.function_spec(bound1=lo_bounds, bound2=hi_bounds, is_integer=is_int)
        if old_evals:
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

        self.problem_parameters, self.params, self.optimizer, self.spec = problem_parameters, params, optimizer, spec
        self.eps, self.noise_mag, self.is_int = eps, noise_mag, is_int
        self.fname, self.save = fname, save

        self.n_iter = n_iter
        self.save_iter = save_iter

        self.eval_fun = partial(eval_obj_fun, obj_fun, self.problem_parameters, self.params, self.is_int)
        self.reduce_fun = reduce_fun
        
        self.evals = [[] for _ in range(n_iter)]

    def print_best(self):
        best_eval = self.optimizer.get_best_function_eval()
        prms = list(zip(self.params, list(best_eval[0])))
        res = best_eval[1]
        print(f"Best eval so far: {res}@{prms}")


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
    gfsopt = GFSOptimizer(**gfsopt_params)
    gfsopt_dict[gfsopt.opt_id] = gfsopt
    return gfsopt
    
def gfsctrl(controller, gfsopt_params):
    """Controller for distributed GFS optimization."""
    gfsopt = gfsinit(gfsopt_params)
    logger.info("Optimizing for %d iterations..." % gfsopt.n_iter)
    n_steps = distwq.n_workers if distwq.workers_available else 1
    for i in range(0, gfsopt.n_iter):
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
            gfsopt.evals[i][j].set(rres)
    controller.info()

def gfswork(worker, gfsopt_params):
    """Worker for distributed GFS optimization."""
    gfsinit(gfsopt_params)

def eval_fun(opt_id, *args):
    return gfsopt_dict[opt_id].eval_fun(*args)

def run(gfsopt_params, nprocs_per_worker=1, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)
    
    if distwq.is_controller:
        distwq.run(fun_name="gfsctrl", module_name="distgfs",
                   verbose=True, args=(gfsopt_params,), nprocs_per_worker=nprocs_per_worker)
        opt_id = gfsopt_params['opt_id']
        gfsopt = gfsopt_dict[opt_id]
        gfsopt.print_best()
    else:
        distwq.run(fun_name="gfswork", module_name="distgfs",
                   verbose=True, args=(gfsopt_params,), nprocs_per_worker=nprocs_per_worker)


