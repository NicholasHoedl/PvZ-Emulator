"""Build smoke tests: the pybind11 module imports and a pool world runs."""
import os


def test_import():
    import pvzemu  # noqa: F401


def test_world_constructs():
    import pvzemu

    pvzemu.World(pvzemu.SceneType.pool)


def test_update_returns_bool():
    import pvzemu

    w = pvzemu.World(pvzemu.SceneType.pool)
    assert isinstance(w.update(), bool)


def test_fresh_world_not_game_over():
    import pvzemu

    w = pvzemu.World(pvzemu.SceneType.pool)
    assert w.scene.is_game_over is False


def test_module_is_built_pyd():
    import pvzemu

    path = pvzemu.__file__
    assert "lib" + os.sep + "pybind11" not in path
    assert path.endswith(".pyd")
