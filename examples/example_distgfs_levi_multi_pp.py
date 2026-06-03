import logging
import math

import distgfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    results = {}
    for problem_id, params in pp.items():
        x, y = params["x"], params["y"]
        scale = 0.5 if problem_id == 0 else 0.4
        results[problem_id] = -levi(scale * x, y)
    logger.info(f"Iter: {pid}\t pp:{pp}, result:{results}")
    return results


if __name__ == "__main__":
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}

    distgfs_params = {
        "opt_id": "distgfs_levi_multi_pp",
        "problem_ids": {0, 1},
        "per_problem_tasks": True,
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "example_distgfs_levi_multi_pp",
        "problem_parameters": problem_parameters,
        "space": space,
        "n_iter": 20,
        "file_path": "distgfs.levi.multi.pp.h5",
        "save": True,
    }

    distgfs.run(distgfs_params, verbose=True)
