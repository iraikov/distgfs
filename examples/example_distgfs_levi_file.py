import math, logging, distgfs, pickle
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

feature_dtypes = [('pid1', (np.int32, 2)), ('pid2', (np.float32, 4)) ]

def levi(x, y):
    """
    Levi's function (see https://en.wikipedia.org/wiki/Test_functions_for_optimization).
    Has a global _minimum_ of 0 at x=1, y=1.
    """
    a = math.sin(3. * math.pi * x)**2
    b = (x - 1)**2 * (1 + math.sin(3. * math.pi * y)**2)
    c = (y - 1)**2 * (1 + math.sin(2. * math.pi * y)**2)
    return a + b + c


def obj_fun(pp, pid):
    """ Objective function to be _maximized_ by GFS. """
    res = levi(**pp)
    logger.info(f"Iter: {pid}\t x:{pp['x']}, y:{pp['y']}, result:{res}")
    # Since Dlib maximizes, but we want to find the minimum,
    # we negate the result before passing it to the Dlib optimizer.
    return -res, np.array([[pid, pid], [float(pid), float(pid)]], dtype=np.dtype(feature_dtypes))


if __name__ == '__main__':

    # For this example, we pretend that we want to keep 'y' fixed at 1.0
    # while optimizing 'x' in the range -4.5 to 4.5
    space = {'x': [-4.5, 4.5]}
    problem_parameters = {'y': 1.}
    
    # Create an optimizer parameter set
    distgfs_params = {'opt_id': 'distgfs_levi',
                      'obj_fun_name': 'obj_fun',
                      'obj_fun_module': 'example_distgfs_levi_file',
                      'problem_parameters': problem_parameters,
                      'space': space,
                      'n_iter': 10,
                      'file_path': 'distgfs.levi.h5',
                      'save': True,
                      'feature_dtypes': feature_dtypes,
                      }

    distgfs.run(distgfs_params, verbose=True)


