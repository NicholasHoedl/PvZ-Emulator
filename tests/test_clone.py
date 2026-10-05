"""Step 1.4: World.clone() returns a faithful, independent copy of a world.

Emulator facts used here (cited as emulator <path>:<line>):
- world.h:88-103: world's copy constructor copies the scene, then builds every system on the
  copy; the sun system's constructor redraws natural_sun_countdown (system/sun.h:21-23) and the
  spawn system's constructor calls reset() (system/spawn.cpp:33-41), which rebuilds the spawn
  flags and the spawn list (spawn.cpp:458-469), all from scene.rng.
- object/scene.cpp:5-47: scene's copy constructor rebuilds plant_map from the plant list;
  system/plant/plant_factory.cpp:442-444: a cannon occupies plant_map[row][col] and [col + 1];
  plant_factory.cpp:257-259: a plain plant can't go where plant_map[row][col].content is set.
- world.cpp:96-104: op -1 shovels plant_map[row][col].content; world.cpp:178-189: a card's mask
  is 1 when it is off cooldown and plant_factory.can_plant agrees.
- system/plant/cob_cannon.cpp:34: launch() sets countdown.launch = 206, so the cob projectile
  appears about 200 updates after the fire.
- object/scene.h:175: pool has 6 rows; world.cpp:90: 9 columns.
"""
import copy

import pvzemu

COB_FIRE = -3  # step 1.2 action op
SHOVEL = -1  # emulator world.cpp:96
SHOVEL_PUMPKIN = -2  # emulator world.cpp:91
POOL_ROWS = 6  # emulator object/scene.h:175
COLS = 9  # emulator world.cpp:90

# Test choices, not game facts: the 1.3 seeding test's ten cards and defence, two cannons on
# land rows at col 0, a click pixel inside the lawn, and caps on tick loops.
_P = pvzemu.PlantType
PS = pvzemu.PlantStatus
CARDS = [
    _P.pea_shooter,
    _P.sunflower,
    _P.cherry_bomb,
    _P.wallnut,
    _P.potato_mine,
    _P.snow_pea,
    _P.chomper,
    _P.repeater,
    _P.lily_pad,
    _P.pumpkin,
]
COB_CELLS = [(1, 0), (4, 0)]
CLICK_X = 600
LAND_DEFENCE = [None, None, _P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_DEFENCE = [_P.cattail] * 2 + [_P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_ROWS = (2, 3)  # emulator object/scene.h:171
ARM_CAP = 5000  # same cap as the 1.2 cob test
FLIGHT_CAP = 2000


def _new_world(seed=7):
    w = pvzemu.World(pvzemu.SceneType.pool, seed)
    assert w.select_plants(CARDS, _P.none) is True
    return w


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


def _script_action(tick, variant=0):
    """One fixed action (or None for a plain update) per tick; variant shifts rows and cols."""
    if tick > 0 and tick % 250 == 0:
        return (COB_FIRE, (tick // 250 + variant) % POOL_ROWS, CLICK_X)
    if tick > 0 and tick % 1000 == 500:
        k = tick // 1000 + variant
        return (SHOVEL if k % 2 else SHOVEL_PUMPKIN, k % POOL_ROWS, 2 + k % 7)
    if tick % 37 == 0:
        k = tick // 37 + variant
        return (int(CARDS[k % len(CARDS)]), k % POOL_ROWS, 1 + (k * 5) % 8)
    return None


def _step(w, tick, variant=0):
    action = _script_action(tick, variant)
    if action is None:
        w.update()
    else:
        w.update(action)


def _run(worlds, start, stop, variant=0):
    for tick in range(start, stop):
        for w in worlds:
            _step(w, tick, variant)


def _scripted_world(ticks=1000):
    w = _new_world()
    _place_defence(w)
    _run([w], 0, ticks)
    return w


def _cobs(w):
    return [p for p in w.scene.plants if p.type == _P.cob_cannon]


def _cob_statuses(w):
    return [(p.row, p.col, p.status) for p in _cobs(w)]


def _cob_projectiles(w):
    return [q for q in w.scene.projectiles if q.type == pvzemu.ProjectileType.cob_cannon]


def _tick_until(w, cond, cap):
    for n in range(1, cap + 1):
        w.update()
        if cond():
            return n
    return None


def _armed_cob_world():
    w = _new_world()
    for row, col in COB_CELLS:
        w.plant_factory.create(_P.cob_cannon, row, col)
    n = _tick_until(
        w, lambda: all(p.status == PS.cob_cannon_armed_idle for p in _cobs(w)), ARM_CAP)
    assert n is not None, "cannons did not arm within the cap"
    return w


def _mask(w, actions):
    m = pvzemu.IntVector()
    w.get_available_actions(actions, m)
    return list(m)[:-1]


def test_clone_lockstep():
    w = _scripted_world(1000)
    assert w.scene.spawn.wave >= 1
    c = w.clone()
    assert isinstance(c, pvzemu.World)
    assert c.to_json() == w.to_json()

    for start in range(1000, 6000, 1000):
        _run([w, c], start, start + 1000)
        assert c.to_json() == w.to_json(), "diverged by tick %d" % (start + 1000)

    # The run must get several waves in with zombies on the board, or it proves little.
    print("wave %d, zombies %d, plants %d" % (
        w.scene.spawn.wave, len(w.scene.zombies), len(w.scene.plants)))
    assert w.scene.spawn.wave >= 3
    assert not w.scene.is_game_over


def test_clone_immediately_equal():
    for ticks in (0, 1000):
        w = _scripted_world(ticks)
        c = w.clone()
        assert c.to_json() == w.to_json()
        assert c.scene.spawn.spawn_list == w.scene.spawn.spawn_list
        assert c.scene.spawn.wave == w.scene.spawn.wave
        assert c.scene.spawn.countdown.next_wave == w.scene.spawn.countdown.next_wave
        assert c.scene.sun.sun == w.scene.sun.sun
        assert c.scene.sun.natural_sun_countdown == w.scene.sun.natural_sun_countdown
        assert c.scene.zombie_dancing_clock == w.scene.zombie_dancing_clock
        assert [k.cold_down for k in c.scene.cards] == [k.cold_down for k in w.scene.cards]
        # scene.rng is not in to_json: one update each must still agree.
        w.update()
        c.update()
        assert c.to_json() == w.to_json()


def _snapshot(w):
    return w.to_json(), len(w.scene.plants), _cob_statuses(w)


def _disturb(w, row, col):
    assert w.plant_factory.can_plant(row, col, _P.pea_shooter, _P.none)
    w.plant_factory.create(_P.pea_shooter, row, col)
    w.update((COB_FIRE, 2, CLICK_X))
    assert any(p.status == PS.cob_cannon_launch for p in _cobs(w))
    for _ in range(300):
        w.update()


def test_clone_independent_both_ways():
    w = _armed_cob_world()
    c = w.clone()
    before = _snapshot(w)
    _disturb(c, 0, 5)
    assert _snapshot(w) == before
    assert len(c.scene.plants) == len(w.scene.plants) + 1

    w = _armed_cob_world()
    c = w.clone()
    before = _snapshot(c)
    _disturb(w, 5, 5)
    assert _snapshot(c) == before
    assert len(w.scene.plants) == len(c.scene.plants) + 1


def test_clone_diverges_on_different_scripts():
    w = _scripted_world(1000)
    c = w.clone()
    _run([w], 1000, 3000, variant=0)
    _run([c], 1000, 3000, variant=1)
    assert c.to_json() != w.to_json()


def test_clone_keeps_cob_second_cell():
    w = _new_world()
    w.plant_factory.create(_P.cob_cannon, 0, 3)
    c = w.clone()
    free = (int(_P.pea_shooter), 0, 6)
    second = (int(_P.pea_shooter), 0, 4)
    for x in (w, c):
        assert x.plant_factory.can_plant(0, 4, _P.pea_shooter, _P.none) is False
        assert x.plant_factory.can_plant(0, 6, _P.pea_shooter, _P.none) is True
        assert _mask(x, [second, free]) == [0, 1]
        assert x.scene.plant_map[0][4].content is not None
        assert x.scene.plant_map[0][4].content.type == _P.cob_cannon

    # Shovelling the second cell removes the cannon in the clone as in the original.
    c.update((SHOVEL, 0, 4))
    assert [p.is_dead for p in _cobs(c)] in ([], [True])
    assert c.plant_factory.can_plant(0, 4, _P.pea_shooter, _P.none) is True
    assert len(_cobs(w)) == 1 and not _cobs(w)[0].is_dead
    assert w.plant_factory.can_plant(0, 4, _P.pea_shooter, _P.none) is False


def test_clone_of_clone_lockstep():
    w = _scripted_world(1000)
    c = w.clone()
    _run([w, c], 1000, 1500)
    cc = c.clone()
    assert cc.to_json() == w.to_json()
    _run([w, c, cc], 1500, 3500)
    assert c.to_json() == w.to_json()
    assert cc.to_json() == w.to_json()


def _flight_ticks(w, cap):
    """Per-update count of cob projectiles, until one has appeared and all have gone."""
    counts = []
    seen = False
    for _ in range(cap):
        w.update()
        n = len(_cob_projectiles(w))
        counts.append(n)
        seen = seen or n > 0
        if seen and n == 0:
            return counts
    return None


def test_clone_mid_flight():
    # Clone 50 updates after the fire (cannon still in its launch animation) and again
    # while the cob projectile is in the air.
    for in_air in (False, True):
        w = _armed_cob_world()
        w.update((COB_FIRE, 2, CLICK_X))
        for _ in range(50):
            w.update()
        if in_air:
            assert _tick_until(w, lambda: len(_cob_projectiles(w)) > 0, FLIGHT_CAP) is not None
        c = w.clone()
        a = _flight_ticks(w, FLIGHT_CAP)
        b = _flight_ticks(c, FLIGHT_CAP)
        assert a is not None, "cob projectile did not appear and disappear within the cap"
        assert b == a
        assert c.to_json() == w.to_json()


def test_copy_and_deepcopy_are_clones():
    w = _scripted_world(1000)
    for c in (copy.copy(w), copy.deepcopy(w)):
        assert c is not w
        assert c.to_json() == w.to_json()
        c.update((COB_FIRE, 1, CLICK_X))
        w2 = w.clone()
        w2.update((COB_FIRE, 1, CLICK_X))
        assert c.to_json() == w2.to_json()
    before = w.to_json()
    w.clone().update()
    assert w.to_json() == before
