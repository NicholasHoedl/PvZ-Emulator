"""Step 1.5: curriculum hooks: World.build, writable sun and flag counter, PlantFactory.get_cost.

Emulator facts used here (cited as emulator <path>:<line>):
- world.h:157-160: check_list is a vector of (plant_type, row, col); world.cpp:484-510:
  check_build checks a pumpkin entry against plant_map[row][col].pumpkin, a lily pad or flower
  pot against .base with that type, and any other plant against .content with that type.
- system/plant/plant_factory.cpp:275-279: create(type, row, col, imitater_target) places a plant
  with no sun cost or cooldown; plant_factory.cpp:436-447 sets the plant_map cells, a cannon
  taking [row][col] and [row][col + 1].
- world.cpp:96-104: op -1 shovels plant_map[row][col].content; plant_factory.cpp:594-596: destroy
  clears that cell.
- object/scene.h:62: spawn.total_flags; scene.h:91 and object/scene.cpp:288 (reset): it starts
  at 1000; scene.cpp:131-132: to_json writes it under the key "total_flags".
- system/spawn.cpp:458-469: spawn::reset sets wave 0 and rebuilds the spawn flags and list
  without touching total_flags.
- object/scene.h:98: sun.sun; scene.h:103 and scene.cpp:294 (reset): it starts at 9990.
- plant_factory.cpp:157: can_plant refuses when get_cost(type) > scene.sun.sun;
  plant_factory.cpp:531: plant spends get_cost(type); world.cpp:178-189: a card's mask is 1 when
  it is off cooldown and can_plant agrees.
- system/sun.cpp:11-12: the natural sun countdown is at least 425, so one update on a fresh pool
  world adds no sun.
- system/spawn.cpp:166: gen_spawn_list picks wave types from t < giga_gargantuar, and
  object/zombie.h:38: giga_gargantuar = 0x20 is the last zombie type, so no spawn list holds one.
  spawn.cpp:108-110: gargantuar is gated only by its per-wave count, never by the flag counter.
- object/scene.h:171: pool water rows are 2 and 3; scene.h:175: pool has 6 rows.
"""
import pvzemu

SHOVEL = -1  # emulator world.cpp:96
POOL_ROWS = 6  # emulator object/scene.h:175
START_FLAGS = 1000  # emulator object/scene.h:91
START_SUN = 9990  # emulator object/scene.h:103

# Test choices, not game facts: the 1.4 clone test's ten cards, defence and cannon cells, a
# flag counter well past the start, a free land cell, a seed count for the spawn-list check.
_P = pvzemu.PlantType
_Z = pvzemu.ZombieType
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
LAND_DEFENCE = [None, None, _P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_DEFENCE = [_P.cattail] * 2 + [_P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_ROWS = (2, 3)  # emulator object/scene.h:171
FLAGS = 1020
FREE_LAND = (0, 5)
SEEDS = range(20)
SEED = 7


def _new_world(seed=SEED):
    w = pvzemu.World(pvzemu.SceneType.pool, seed)
    assert w.select_plants(CARDS, _P.none) is True
    return w


def _defence_list():
    """The 1.4 defence as a check_list: lily pad before its plant, pumpkin after it."""
    lst = []
    for row in range(POOL_ROWS):
        water = row in WATER_ROWS
        for col, plant in enumerate(WATER_DEFENCE if water else LAND_DEFENCE):
            if plant is None:
                continue
            if water:
                lst.append((_P.lily_pad, row, col))
            lst.append((plant, row, col))
            lst.append((_P.pumpkin, row, col))
    for row, col in COB_CELLS:
        lst.append((_P.cob_cannon, row, col))
    return lst


def _mask(w, actions):
    m = pvzemu.IntVector()
    w.get_available_actions(actions, m)
    return list(m)[:-1]


def _cobs(w):
    return [p for p in w.scene.plants if p.type == _P.cob_cannon]


def _spawn_types(w):
    return [z for wave in w.scene.spawn.spawn_list for z in wave]


def test_build_matches_check_build():
    w = _new_world()
    lst = _defence_list()
    assert w.build(lst) is True
    assert w.check_build(lst) is True
    assert len(w.scene.plants) == len(lst)
    assert len(_cobs(w)) == len(COB_CELLS)

    pm = w.scene.plant_map
    for t, r, c in lst:
        cell = pm[r][c]
        if t == _P.pumpkin:
            assert cell.pumpkin is not None and cell.pumpkin.type == _P.pumpkin
        elif t == _P.lily_pad:
            assert cell.base is not None and cell.base.type == _P.lily_pad
        else:
            assert cell.content is not None and cell.content.type == t
    for r, c in COB_CELLS:
        assert pm[r][c + 1].content is not None
        assert pm[r][c + 1].content.type == _P.cob_cannon
        assert w.plant_factory.can_plant(r, c + 1, _P.pea_shooter, _P.none) is False

    # Losing one plant of the build makes check_build fail.
    r, c = 0, 3
    assert pm[r][c].content.type == _P.gatling_pea
    w.update((SHOVEL, r, c))
    assert w.scene.plant_map[r][c].content is None
    assert w.check_build(lst) is False


def test_build_empty_list():
    w = _new_world()
    assert w.build([]) is True
    assert len(w.scene.plants) == 0


def test_flag_counter_writable():
    w = _new_world()
    assert w.scene.spawn.total_flags == START_FLAGS
    w.scene.spawn.total_flags = FLAGS
    assert w.scene.spawn.total_flags == FLAGS
    assert '"total_flags":%d' % FLAGS in w.to_json()

    for _ in range(700):
        w.update()
    assert w.scene.spawn.wave >= 1
    w.spawn.reset()
    assert w.scene.spawn.total_flags == FLAGS
    assert w.scene.spawn.wave == 0

    w.reset(SEED)
    assert w.scene.spawn.total_flags == START_FLAGS


def test_sun_writable_and_get_cost():
    cost = _new_world().plant_factory.get_cost(_P.pea_shooter)
    assert cost > 0
    r, c = FREE_LAND
    action = (int(_P.pea_shooter), r, c)

    w = _new_world()
    assert w.scene.sun.natural_sun_countdown > 1
    w.scene.sun.sun = cost
    assert w.scene.sun.sun == cost
    assert _mask(w, [action]) == [1]
    w.update(action)
    assert w.scene.plant_map[r][c].content is not None
    assert w.scene.plant_map[r][c].content.type == _P.pea_shooter
    assert w.scene.sun.sun == 0

    w = _new_world()
    w.scene.sun.sun = cost - 1
    assert _mask(w, [action]) == [0]
    n = len(w.scene.plants)
    w.update(action)
    assert len(w.scene.plants) == n
    assert w.scene.sun.sun == cost - 1

    w.reset(SEED)
    assert w.scene.sun.sun == START_SUN


def test_get_cost_counts_upgrades():
    # plant_factory.cpp:119-128: an upgrade plant costs 50 more per copy already on the board.
    w = _new_world()
    base = w.plant_factory.get_cost(_P.gatling_pea)
    w.plant_factory.create(_P.gatling_pea, 0, 5)
    assert w.plant_factory.get_cost(_P.gatling_pea) == base + 50


def test_no_giga_in_spawn_lists():
    seeds_with_garg = []
    for seed in SEEDS:
        w = pvzemu.World(pvzemu.SceneType.pool, seed)
        types = _spawn_types(w)
        assert _Z.giga_gargantuar not in types
        if _Z.gargantuar in types:
            seeds_with_garg.append(seed)

        w.scene.spawn.total_flags = FLAGS
        w.spawn.reset()
        assert w.scene.spawn.total_flags == FLAGS
        assert _Z.giga_gargantuar not in _spawn_types(w)

    print("seeds with a gargantuar at flag 0: %d of %d" % (len(seeds_with_garg), len(SEEDS)))
    assert seeds_with_garg


def test_clone_carries_hooks():
    w = _new_world()
    lst = _defence_list()
    w.scene.spawn.total_flags = FLAGS
    w.scene.sun.sun = 400
    assert w.build(lst) is True
    c = w.clone()
    assert c.scene.spawn.total_flags == FLAGS
    assert c.scene.sun.sun == 400
    assert c.check_build(lst) is True
    assert c.to_json() == w.to_json()
