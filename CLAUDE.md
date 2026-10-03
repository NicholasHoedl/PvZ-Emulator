# emulator/: Nick's fork of PvZ-Emulator

The x64 C++ emulator with pybind11 bindings, loaded by the Python agent. The pvz-rl root CLAUDE.md still applies.

## Build

- **x64 only**, built with CMake and Visual Studio Build Tools against 64-bit Python 3.11+. Never use the i686 harness toolchain here.
- Packaging: `pyproject.toml` with scikit-build-core drives `CMakeLists.txt`; pybind11 comes from pip (`pybind11>=3.1`), not a vendored copy. The CMake static library is `pvzemu_core`; the Python module target and file are `pvzemu`.
- Build (from the pvz-rl root, after `pip install scikit-build-core pybind11 ninja` into `.venv` once): `.venv\Scripts\python.exe -m pip install -e emulator --no-build-isolation`. The editable install copies the `.pyd` into site-packages, so rerun it after any C++ change.
- Test (from the pvz-rl root): `.venv\Scripts\python.exe -m pytest emulator/tests`.

## Changes

- **Every behavior change (a new feature or a fix) ships with a pytest** that fails without it.
- **Upstream mechanics change only for a fidelity fix** backed by a Phase 4 scenario test showing the real game differs. The corrected value comes from the measurement in pvz-rl `docs/parity.md`, never from memory. Record each fix in pvz-rl `docs/decisions.md`.
- **Exception: the upstream PRs Nick picks in step 0.7.** Record each one in pvz-rl `docs/decisions.md`, including whether it changes game mechanics, so Phase 4 can cover it. The 0.9 smoke tests check those PRs and the 0.8 build fixes; from then on, every change keeps the whole suite passing.
- New features (cob fire `op = -3`, seeding, `World.clone()`, curriculum hooks, the schema adapter) are additions. When a feature is not used, behavior stays identical to upstream.
- Reuse the emulator's own conversions and constants (row to pixel, tick timings, plant and zombie stats). Find them in the source and cite file and line; never invent them.
- `schema/pvz_state.h` is a generated, byte-identical copy of pvz-rl's `schema/pvz_state.h` (written there by `python -m schema.gen --write`). Never edit it here. A header change is committed in this fork first, then in pvz-rl.
- Commits carry the roadmap step ID. Merged upstream PRs keep their merge commit, named after the PR number. Push each fork commit before pvz-rl updates its submodule pointer to it.

## Legal

- Keep the upstream license.
- Never commit images or other files extracted from the game's `main.pak`. The fork's own `.gitignore` must exclude the folder where step 1.9 puts them.
