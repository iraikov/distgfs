# Changelog

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
