# Changelog

## 2.0.0

distgfs now runs on distwq 2.0. Read "Breaking changes" before upgrading:
function names and the `pid` passed to the objective have changed.

### Breaking changes

- distwq 2.0 or later and Python 3.11 or later are now required.
- Functions are named by one path that includes the module, for example
  `"obj_fun_name": "mymodule.obj_fun"`. This applies to `obj_fun_name`,
  `obj_fun_init_name`, `controller_init_fun_name`, `reduce_fun_name` and
  `broker_fun_name`. A name without a module is looked up in `__main__`,
  as before.
- The keys `obj_fun_module`, `obj_fun_init_module`,
  `controller_init_fun_module`, `reduce_fun_module` and
  `broker_module_name` are removed. Passing one raises `TypeError` that
  names the path to use instead.
- A name that cannot be found raises `ImportError` (before, `NameError` or
  `ModuleNotFoundError`). A name that is not a function raises `TypeError`.
- Each task passes its own `pid` to the objective: 0 to `n_iter`-1,
  counted per problem with `per_problem_tasks`. Before, every task in a
  batch got the same value. Check objectives that derive seeds or file
  names from `pid`.
- `broker_fun_name` requires `spawn_workers=True`; otherwise it raises
  `ValueError`.

### New

- If the objective raises, distgfs writes a checkpoint with every result
  received so far (when `save` is set), then `run()` raises the same
  exception. Before, the whole MPI job aborted and the results since the
  last save were lost.
- `run()` returns the best result on every rank, not only on the
  controller. Ranks left idle by `max_workers` return `None`.

### Changed

- A run makes exactly `n_iter` evaluations. Before, with `n_max_tasks`
  greater than 1, the last batch could run past `n_iter`.
- With `per_problem_tasks`, problems take turns across refills. Before,
  each refill started again at the first problem.
- Problem ids are processed in sorted order. Ids that cannot be compared
  keep their given order.
- A checkpoint is written after every `save_iter` new results and at the
  end. Before, it was written only when the result count was an exact
  multiple of `save_iter`.
- In a worker group's results, `None` values are dropped before a result
  is chosen or `reduce_fun` is called, as with distwq 1.3. If every
  process returns `None`, `ValueError` is raised.
- distwq's run statistics are logged at INFO on the `distwq.future`
  logger instead of printed.
- `run()` no longer changes the caller's `gfsopt_params` on worker ranks.

### Fixed

- The controller waits for results in a blocking MPI probe instead of
  polling, and no longer rewrites the checkpoint on every pass.
- A run without MPI, without `per_problem_tasks` and with `n_max_tasks`
  unset no longer hangs.
- In a single process, `controller_init_fun_name` and `reduce_fun_name`
  are now used. Before, they were ignored.
- `broker_fun_name` can name a function outside distgfs. Before, no user
  function could be found. It is called as
  `fun(broker, gfsopt_params, verbose)`.
- `obj_fun_init_name` without `obj_fun_init_args` no longer raises
  `TypeError`.
- On a resume that takes `problem_ids` from the checkpoint, the default
  `n_max_tasks` with `per_problem_tasks` counts those problems.

### New checks

- `ValueError` if `save` is set and `save_iter` is less than 1, if
  `problem_ids` is empty, or if a result does not hold exactly the problem
  ids of its task.
- `TypeError` if a task result, or the value returned by `reduce_fun`, is
  not a dict keyed by problem id.

## 1.6.0

This is the last release for distwq 1.x. A later release will move distgfs
to the interface of distwq 2.0.

### Changed

- distwq is now limited to versions below 2.0.0 (`distwq>=1.1.0,<2.0.0`).
  distwq 2.0 changes the interface that distgfs uses, so this release does
  not work with it.
- Python 3.10 or later is now required (before, 3.9). CI tests Python 3.10
  and 3.12.

### Fixed

- The 1.4.0 checkpoint that the resume tests read,
  `tests/data/distgfs_v1_4_0.h5`, is now in the repository. Before, those
  tests failed on a fresh clone.
- CI now runs the test suite, on Python 3.10 and 3.12. Before, it ran tox
  without a tox configuration, which installed distgfs and ran no tests.
- `pyproject.toml` has a `test` dependency group (pytest, pytest-timeout),
  used by CI.

## 1.5.0

### New

- New option `parameter_dtype` (a `gfsopt_params` key and a
  `DistGFSOptimizer` keyword). It sets the precision used to store evaluated
  parameters, search bounds and fixed problem parameters in a checkpoint.
  Allowed values are `"float64"` and `"float32"`; any other value raises
  `ValueError`.

### Changed

- **New checkpoints store parameters as float64 by default.** Before 1.5.0
  they were always float32, so a point read back from a checkpoint could
  differ slightly from the point that was evaluated.
- A checkpoint keeps the precision it was created with. If you resume a
  float32 checkpoint (including files written by 1.4.0 and earlier) while
  asking for float64, the run continues in float32 and emits a `UserWarning`.
  Use a new `file_path` or `opt_id` to get float64 storage.
- **`problem_parameters` on resume:** values passed by the caller are now
  used as given. They are compared with the stored values, and a
  `UserWarning` is emitted if a name is missing or a value differs beyond the
  stored precision. Stored values are used only if the caller passes no
  `problem_parameters`. Before this release, serial runs used the stored
  values, which could be rounded to float32. MPI workers already used the
  caller's values.
- The caller's `problem_parameters` dict is no longer changed during
  evaluation. The checkpoint's `problem_parameters` dataset now holds only
  the fixed parameters, not the last evaluated values of the optimised
  parameters.

### Fixed

- Loading `problem_parameters` from older checkpoints could set a fixed
  parameter to 0.0. Those files contain zero-filled trailing rows, and these
  rows overwrote the real value. New checkpoints no longer write such rows,
  and the loader ignores them in old files.
- A new `opt_id` can now be started in an existing checkpoint file that holds
  other groups. Before, this failed while trying to resume the missing group.
