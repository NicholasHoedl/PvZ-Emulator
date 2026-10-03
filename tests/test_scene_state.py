"""Step 1.6: scene.tick counts simulated updates; World.scene can't be rebound from Python.

Emulator facts used here (cited as emulator <path>:<line>):
- world.cpp:23-29: world::update() returns at once when scene.is_game_over, before it advances
  zombie_dancing_clock and tick; world.cpp:37-39: the update in which zombie.update() reports
  a zombie entering the house sets is_game_over and returns true;
  system/zombie/zombie_system.cpp:757-758 and 827-828: that happens when a zombie walks past
  the house threshold.
- world.cpp:71-104: update(action) applies the action, ignoring an out-of-range row or col
  (world.cpp:81), then ends with return update().
- world.h:155-181: every World.reset overload calls a scene.reset overload;
  object/scene.h:182-196 and object/scene.cpp:310-311: every scene.reset overload runs
  scene::reset(), which sets tick = 0 (scene.cpp:275).
- world.h:83-84: world's copy constructor copies the scene with scene(w.scene);
  object/scene.cpp:5-23 is that copy constructor's initialiser list.
- object/scene.cpp:83-87: to_json writes "rows", then "tick"; scene.cpp:131-132 and 265-266:
  it writes "total_flags" and "stop_spawn"; world.cpp:224-230: World.to_json is scene.to_json
  written with a rapidjson Writer (no spaces).
- object/scene.h:92 and object/scene.cpp:289: the first wave comes 600 updates in, so a board
  with no plants is eventually overrun.
"""
import pvzemu

SHOVEL = -1  # emulator world.cpp:86
SEED = 7

# Test choices, not game facts: a short tick count, an out-of-range cell for the illegal
# action, a cap on the game-over loop (the 0.9 smoke test lost by tick ~2,000 to 3,000), how
# long to keep updating once the game is over, and writable values the 1.5 tests also use.
TICKS = 50
OFF_BOARD = (SHOVEL, 99, 99)
EMPTY_CELL_SHOVEL = (SHOVEL, 0, 0)
GAME_OVER_CAP = 20_000
AFTER_GAME_OVER = 50
SUN = 400
FLAGS = 1020


def _worlds():
    return [pvzemu.World(pvzemu.SceneType.pool), pvzemu.World(pvzemu.SceneType.pool, SEED)]


def _ticked_world(n=TICKS):
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    for _ in range(n):
        w.update()
    assert w.scene.tick == n
    return w


def test_tick_starts_at_zero():
    for w in _worlds():
        assert w.scene.tick == 0
    assert pvzemu.Scene(pvzemu.SceneType.pool).tick == 0
    assert pvzemu.Scene(pvzemu.SceneType.pool, SEED).tick == 0


def test_tick_counts_updates():
    for w in _worlds():
        for n in range(1, TICKS + 1):
            w.update()
            assert w.scene.tick == n
        for n in range(TICKS + 1, 2 * TICKS + 1):
            w.update(EMPTY_CELL_SHOVEL if n % 2 else OFF_BOARD)
            assert w.scene.tick == n


def test_tick_unchanged_by_queries():
    w = _ticked_world()
    actions = [EMPTY_CELL_SHOVEL, OFF_BOARD]
    masks = pvzemu.IntVector()
    w.get_available_actions(actions, masks)
    assert w.scene.tick == TICKS
    w.to_json()
    assert w.scene.tick == TICKS
    c = w.clone()
    assert w.scene.tick == TICKS
    assert c.scene.tick == TICKS


def test_tick_frozen_after_game_over():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    assert len(w.scene.plants) == 0
    calls = 0
    while not w.scene.is_game_over:
        assert calls < GAME_OVER_CAP, "no game over within the cap"
        w.update()
        calls += 1
    print("game over after %d updates" % calls)
    end = w.scene.tick
    assert end == calls

    for _ in range(AFTER_GAME_OVER):
        assert w.update() is True
    w.update(EMPTY_CELL_SHOVEL)
    assert w.scene.tick == end


def test_every_reset_zeroes_tick():
    resets = [
        lambda w: w.reset(),
        lambda w: w.reset(seed=SEED),
        lambda w: w.reset(pvzemu.SceneType.pool),
        lambda w: w.reset(pvzemu.SceneType.pool, SEED),
    ]
    for reset in resets:
        w = _ticked_world()
        reset(w)
        assert w.scene.tick == 0

    # The Scene overloads directly, as World.reset calls them.
    for reset in resets:
        w = _ticked_world()
        reset(w.scene)
        assert w.scene.tick == 0


def test_clone_carries_tick():
    w = _ticked_world()
    c = w.clone()
    assert c.scene.tick == TICKS
    for n in range(TICKS + 1, 2 * TICKS + 1):
        w.update()
        c.update()
        assert w.scene.tick == n
        assert c.scene.tick == n
    c.update()
    assert c.scene.tick == 2 * TICKS + 1
    assert w.scene.tick == 2 * TICKS


def test_tick_in_to_json():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    assert '"tick":0' in w.to_json()
    for _ in range(TICKS):
        w.update()
    assert '"tick":%d' % TICKS in w.to_json()
    assert '"tick":0' not in w.to_json()


def test_tick_is_read_only():
    w = _ticked_world()
    try:
        w.scene.tick = 5
    except AttributeError:
        pass
    else:
        raise AssertionError("scene.tick was writable")
    assert w.scene.tick == TICKS


def test_world_scene_cannot_be_rebound():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    w2 = pvzemu.World(pvzemu.SceneType.pool, SEED + 1)
    for _ in range(TICKS):
        w2.update()
    before = w.to_json()
    try:
        w.scene = w2.scene
    except AttributeError:
        pass
    else:
        raise AssertionError("World.scene was rebindable")
    assert w.to_json() == before

    # Writes through the scene still reach the world.
    w.scene.stop_spawn = True
    w.scene.sun.sun = SUN
    w.scene.spawn.total_flags = FLAGS
    assert w.scene.stop_spawn is True
    assert w.scene.sun.sun == SUN
    assert w.scene.spawn.total_flags == FLAGS
    j = w.to_json()
    assert '"stop_spawn":true' in j
    assert '"total_flags":%d' % FLAGS in j
