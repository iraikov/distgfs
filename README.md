# distgfs

Distributed computing framework for the
[Global Function Search](http://dlib.net/optimization.html#global_function_search) 
(GFS) hyperparameter optimizer from the [Dlib](http://dlib.net) library.
Based on [gfsopt](https://github.com/tsoernes/gfsopt).

Provides the following features:
* Parallel optimization: Run distributed hyperparameter searches via [mpi4py](https://github.com/mpi4py/mpi4py).
* Save and restore progress: Save/restore settings, parameters and optimization progress to/from HDF5 file. 
* Average over multiple runs: Run a stochastic objective function using the same
parameters multiple times and report the average to Dlib's Global Function
Search. Useful in highly stochastic domains to avoid biasing the search towards
lucky runs.

For theoretical background of GFS, see ['A Global Optimization Algorithm Worth Using'](http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html) and [Malherbe & Vayatis 2017: Global optimization of Lipschitz functions](https://arxiv.org/abs/1703.02628)

# Example usage
A basic example where we maximize Levi's function with as many parallel processes as there are logical cores, and save progress to file.

```python
import math, distgfs

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
    x = pp['x']
    y = pp['y']

    res = levi(0.4*x, y)
    print(f"Iter: {pid}\t x:{x}, y:{y}, result:{res}")
    # Since Dlib maximizes, but we want to find the minimum,
    # we negate the result before passing it to the Dlib optimizer.
    return -res

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
                 }

distgfs.run(distgfs_params, verbose=True)
```

For additional examples, see [examples](https://github.com/iraikov/distgfs/tree/master/examples).
