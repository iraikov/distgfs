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
    x_0, y_0 = pp[0]["x"], pp[0]["y"]
    x_1, y_1 = pp[1]["x"], pp[1]["y"]
    res = {0: -levi(0.5 * x_0, y_0), 1: -levi(0.4 * x_1, y_1)}
    logger.info(f"Iter: {pid}\t x_0:{x_0}, x_1:{x_1}, y_0:{y_0}, y:{y_1}, result:{res}")
    return res


if __name__ == "__main__":
    # For this example, we pretend that we want to keep 'y' fixed at 1.0
    # while optimizing 'x' in the range -4.5 to 4.5
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 1.0}

    # Create an optimizer parameter set
    distgfs_params = {
        "opt_id": "distgfs_levi_multi",
        "problem_ids": set([0, 1]),
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "example_distgfs_levi_multi",
        "problem_parameters": problem_parameters,
        "space": space,
        "n_iter": 10,
        "file_path": "distgfs.levi.multi.h5",
        "save": True,
    }

    distgfs.run(distgfs_params, verbose=True)
