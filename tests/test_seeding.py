"""Step 1.3: one seed drives every random draw, so a seeded world replays exactly.

Emulator facts used here (cited as emulator <path>:<line>):
- object/scene.h:43: scene.rng (std::mt19937) is the only random engine; system/rng.h draws
  every random number from it. Unseeded, it is seeded from std::random_device
  (object/scene.h:142, object/scene.cpp:272).
- world.h:53-68: world's members are built in order scene, sun, spawn, ...; the sun system's
  constructor draws natural_sun_countdown (system/sun.h:21-23, sun.cpp:10-13) and then the
  spawn system's constructor (system/spawn.cpp:33-41, 458-469) builds the spawn flags and the
  20 x 50 spawn list, all from that rng, so the seed must be in place first.
- world.cpp:85-89 and 469-474: (-3, row, x) fires the next armed cannon and does nothing when
  none is armed; world.cpp:90-112: plant and shovel ops that are not legal do nothing.
- object/scene.h:175: pool has 6 rows; world.cpp:90: 9 columns.
"""
import random
import time

import pvzemu

COB_FIRE = -3  # step 1.2 action op
SHOVEL = -1  # emulator world.cpp:96
SHOVEL_PUMPKIN = -2  # emulator world.cpp:91
POOL_ROWS = 6  # emulator object/scene.h:175
COLS = 9  # emulator world.cpp:90
TICKS = 10_000

# Test choices, not game facts: the same ten cards as the 0.9 smoke test, four cannons on
# land rows at col 0, a fixed click pixel inside the lawn, and a defence placed at tick 0 so
# the scripted run lasts several waves instead of ending early with game over.
CARDS = [
    pvzemu.PlantType.pea_shooter,
    pvzemu.PlantType.sunflower,
    pvzemu.PlantType.cherry_bomb,
    pvzemu.PlantType.wallnut,
    pvzemu.PlantType.potato_mine,
    pvzemu.PlantType.snow_pea,
    pvzemu.PlantType.chomper,
    pvzemu.PlantType.repeater,
    pvzemu.PlantType.lily_pad,
    pvzemu.PlantType.pumpkin,
]
COB_CELLS = [(0, 0), (1, 0), (4, 0), (5, 0)]
CLICK_X = 600
_P = pvzemu.PlantType
LAND_DEFENCE = [None, None, _P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_DEFENCE = [_P.cattail] * 2 + [_P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_ROWS = (2, 3)  # emulator object/scene.h:171


def _new_world(seed):
    w = pvzemu.World(pvzemu.SceneType.pool, seed)
    assert w.scene.rows == POOL_ROWS
    assert w.select_plants(CARDS, pvzemu.PlantType.none) is True
    return w


def _script_action(tick):
    """One fixed action (or None for a plain update) per tick."""
    if tick > 0 and tick % 250 == 0:
        return (COB_FIRE, (tick // 250) % POOL_ROWS, CLICK_X)
    if tick > 0 and tick % 1000 == 500:
        k = tick // 1000
        return (SHOVEL if k % 2 else SHOVEL_PUMPKIN, k % POOL_ROWS, 2 + k % 7)
    if tick % 37 == 0:
        k = tick // 37
        return (int(CARDS[k % len(CARDS)]), k % POOL_ROWS, 1 + (k * 5) % 8)
    return None


def _place_defence(w):
    for row in range(POOL_ROWS):
        water = row in WATER_ROWS
        for col, plant in enumerate(WATER_DEFENCE if water else LAND_DEFENCE):
            if plant is None:
                continue
            if water:
                w.plant_factory.create(_P.lily_pad, row, col)
            w.plant_factory.create(plant, row, col)
            w.plant_factory.create(_P.pumpkin, row, col)
    for row, col in COB_CELLS:
        w.plant_factory.create(_P.cob_cannon, row, col)


def _run_script(w, ticks=TICKS):
    _place_defence(w)
    for tick in range(ticks):
        action = _script_action(tick)
        if action is None:
            w.update()
        else:
            w.update(action)


def _candidate_actions():
    actions = []
    for card in CARDS:
        for row in range(POOL_ROWS):
            for col in range(COLS):
                actions.append((int(card), row, col))
    for op in (SHOVEL, SHOVEL_PUMPKIN):
        for row in range(POOL_ROWS):
            for col in range(COLS):
                actions.append((op, row, col))
    return actions


def _run_smoke_loop(w, ticks=TICKS):
    # The 0.9 smoke loop: random legal actions picked by random.Random(9).
    rng = random.Random(9)
    actions = _candidate_actions()
    masks = pvzemu.IntVector()
    for _ in range(ticks):
        w.get_available_actions(actions, masks)
        legal = [i for i, m in enumerate(masks) if m]
        choice = rng.choice(legal)
        if choice == len(actions):
            w.update()
        else:
            w.update(actions[choice])


def test_same_seed_same_script_identical_state():
    a = _new_world(7)
    b = _new_world(7)
    assert a.to_json() == b.to_json()

    start = time.perf_counter()
    _run_script(a)
    elapsed = time.perf_counter() - start
    _run_script(b)

    print("scripted %d ticks: %.3f s; wave %d, zombies %d, plants %d, game over %s" % (
        TICKS, elapsed, a.scene.spawn.wave, len(a.scene.zombies), len(a.scene.plants),
        a.scene.is_game_over))
    # The run must get several waves in, or the comparison proves little.
    assert a.scene.spawn.wave >= 3
    assert a.to_json() == b.to_json()


def test_same_seed_smoke_loop_identical_state():
    a = _new_world(7)
    b = _new_world(7)
    _run_smoke_loop(a)
    _run_smoke_loop(b)
    assert a.scene.spawn.wave > 0
    assert a.to_json() == b.to_json()


def test_different_seeds_diverge():
    a = _new_world(7)
    b = _new_world(8)
    assert a.scene.spawn.spawn_list != b.scene.spawn.spawn_list

    _run_script(a)
    _run_script(b)
    assert a.to_json() != b.to_json()


def test_reset_with_seed_reproduces_run():
    w = _new_world(7)
    first_spawn_list = w.scene.spawn.spawn_list
    _run_script(w)
    first = w.to_json()

    w.reset(7)
    assert w.select_plants(CARDS, pvzemu.PlantType.none) is True
    assert w.scene.spawn.spawn_list == first_spawn_list
    _run_script(w)
    assert w.to_json() == first

    w.reset(pvzemu.SceneType.pool, 7)
    assert w.scene.rows == POOL_ROWS
    assert w.select_plants(CARDS, pvzemu.PlantType.none) is True
    assert w.scene.spawn.spawn_list == first_spawn_list
    _run_script(w)
    assert w.to_json() == first


def test_reset_with_seed_matches_fresh_seeded_world():
    fresh = _new_world(7)
    w = _new_world(8)
    _run_script(w, 1000)
    w.reset(7)
    assert w.select_plants(CARDS, pvzemu.PlantType.none) is True
    assert w.to_json() == fresh.to_json()
    assert w.scene.spawn.spawn_list == fresh.scene.spawn.spawn_list
    assert w.scene.zombie_dancing_clock == fresh.scene.zombie_dancing_clock


def test_seeded_scene_constructor_and_reset():
    s = pvzemu.Scene(pvzemu.SceneType.pool, 7)
    t = pvzemu.Scene(pvzemu.SceneType.pool, 7)
    assert s.zombie_dancing_clock == t.zombie_dancing_clock
    # world(t, seed) seeds its scene the same way before the spawn system draws.
    assert pvzemu.World(pvzemu.SceneType.pool, 7).scene.zombie_dancing_clock == \
        s.zombie_dancing_clock

    u = pvzemu.Scene(pvzemu.SceneType.day, 8)
    u.reset(7)
    assert u.zombie_dancing_clock == s.zombie_dancing_clock
    u.reset(pvzemu.SceneType.pool, 8)
    assert u.type == pvzemu.SceneType.pool
    u.reset(pvzemu.SceneType.pool, 7)
    assert u.zombie_dancing_clock == s.zombie_dancing_clock


def test_unseeded_world_still_runs():
    w = pvzemu.World(pvzemu.SceneType.pool)
    assert w.scene.rows == POOL_ROWS
    for _ in range(600):
        assert isinstance(w.update(), bool)
    assert w.scene.spawn.wave == 1
    w.reset()
    w.reset(pvzemu.SceneType.pool)
    assert w.scene.spawn.wave == 0


def test_to_json_is_stable_without_updates():
    w = _new_world(7)
    assert w.to_json() == w.to_json()
    _run_script(w, 1000)
    assert w.to_json() == w.to_json()
