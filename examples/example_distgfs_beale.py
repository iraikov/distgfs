import logging

import distgfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def beale(x, y):
    """
     Beale's function
    (see https://en.wikipedia.org/wiki/Test_functions_for_optimization).
     Has a global _minimum_ of 0 at x=3, y=0.5.
    """
    a = (1.5 - x + x * y) ** 2
    b = (2.25 - x + x * y * y) ** 2
    c = (2.625 - x + x * y * y * y) ** 2
    return a + b + c


def obj_fun(pp, pid):
    """Objective function to be _maximized_ by GFS."""
    res = beale(**pp)
    logger.info(f"Iter: {pid}\t x:{pp['x']}, y:{pp['y']}, result:{res}")
    # Since Dlib maximizes, but we want to find the minimum,
    # we negate the result before passing it to the Dlib optimizer.
    return -res


if __name__ == "__main__":
    # For this example, we pretend that we want to keep 'y' fixed at 0.5
    # while optimizing 'x' in the range -4.5 to 4.5
    space = {"x": [-4.5, 4.5]}
    problem_parameters = {"y": 0.5}

    # Create an optimizer
    distgfs_params = {
        "opt_id": "distgfs_beale",
        "obj_fun_name": "obj_fun",
        "obj_fun_module": "example_distgfs_beale",
        "problem_parameters": problem_parameters,
        "space": space,
        "seed": 42,
        "n_iter": 10,
    }

    distgfs.run(distgfs_params, spawn_workers=False, verbose=True)
