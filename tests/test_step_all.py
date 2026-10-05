"""Step 1.7: the vector-env additions: World.apply_action, pvzemu.OP_NOOP, the batched
World.step_all and World.masks_all writing into NumPy arrays, and the writable endgame countdown.

Emulator facts used here (cited as emulator <path>:<line>):
- world.cpp:24-26: update() returns true at once, before tick advances (world.cpp:29), when
  the game is over; world.cpp:37-39: it sets is_game_over and returns true when
  zombie.update() says so.
- world.cpp:62-65: when endgame.update() is true, update() resets the spawn data, increments
  total_flags and returns true (a round end); system/endgame.h:12-14: endgame.update() is true
  on the update that counts countdown.endgame down from 1 to 0.
- system/spawn.cpp:458-465: spawn::reset sets wave 0 and countdown.endgame 0.
- object/scene.h:92: next_wave starts at 600; system/spawn.cpp:404-421: the first wave spawns,
  and wave becomes 1, when it reaches 0.
- world.cpp:319-322 (update_all): update(action), then frames - 1 more update() calls,
  stopping once one returns true.
- world.cpp:198-202: get_available_actions writes len(actions) + 1 entries, the last always 1,
  the rest from world.cpp:135-191: op -3 is legal iff the row is in range and a cob is armed
  (x is ignored), op -1 iff the cell holds a coffee bean, content or base plant that is alive
  and not smashed, op >= 0 iff a card of that type is in hand, off cooldown and can_plant
  agrees.
- world.cpp:85 / 91 / 96: ops -3 (cob fire, (row, x)), -2 (pumpkin shovel), -1 (shovel).
- world.cpp:90: plant and shovel ops need 0 <= row < rows and 0 <= col < 9.
- object/scene.h:175: pool has 6 rows; object/scene.h:171: its water rows are 2 and 3.
- object/plant.h:12-13 and :61: plant codes run from pea_shooter 0 to imitater 0x30 (48).
- system/plant/plant_factory.cpp:457-462: plant() refuses a card on cooldown;
  plant_factory.cpp:531 spends the cost; :543 sets the card's cooldown.
- system/plant/plant_factory.cpp:210-215: a coffee bean needs a sleeping content plant;
  system/plant/plant_base.cpp:219-223: a mushroom created in a pool scene goes to sleep.
- system/sun.cpp:23-24: in a pool scene, natural sun adds 25 to sun.sun when its countdown
  reaches 0.
- system/plant/cob_cannon.cpp:43-48: a new cob starts unarmed; cob_cannon.cpp:28-41: launch()
  turns an armed cob into cob_cannon_launch.
- object/scene.cpp:86-87: to_json includes tick; scene.cpp:159-163: it writes hugewave_fade and
  the endgame countdown under their own keys (upstream wrote hugewave_fade under "endgame";
  fixed in step 1.7).
"""
import functools

import numpy as np
import pytest

import pvzemu

COB_FIRE = -3  # emulator world.cpp:85
PUMPKIN_SHOVEL = -2  # emulator world.cpp:91
SHOVEL = -1  # emulator world.cpp:96
POOL_ROWS = 6  # emulator object/scene.h:175
WATER_ROWS = (2, 3)  # emulator object/scene.h:171
COLS = 9  # emulator world.cpp:90
N_TYPES = 49  # emulator object/plant.h:13 and :61: pea_shooter 0 .. imitater 48

_P = pvzemu.PlantType
_PS = pvzemu.PlantStatus

# Test choices, not game facts: the 1.4/1.5 cards, defence and cannon cells; a second loadout
# with an imitater and a coffee bean; click pixel, seeds, caps, sentinel and frame counts.
SEED = 7
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
CARDS_B = [_P.coffee_bean, _P.fumeshroom, _P.imitater, _P.kernelpult, _P.pumpkin, _P.squash]
IMITATER_B = _P.doomshroom
COB_CELLS = [(1, 0), (4, 0)]
LAND_DEFENCE = [None, None, _P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_DEFENCE = [_P.cattail] * 2 + [_P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
FREE = (0, 0)  # a land cell the defence leaves empty
FREE2 = (5, 1)
TAKEN = (0, 3)  # a gatling pea of the defence
SLEEPER_CELL = (5, 0)  # loadout-B worlds get a sleeping fume-shroom here
CLICK_X = 400
ARM_CAP = 5000
GAME_OVER_CAP = 20_000
WAVE_ONE_TICKS = 700  # past next_wave 600 (scene.h:92), so wave 1 has spawned
MIDGAME_TICKS = 900
SENTINEL = 0xCD
FRAMES = (1, 3, 10)
N_THREADED = 64
THREADED_CALLS = 50
THREADED_FRAMES = 20

PLANT = int(_P.pea_shooter)


def _noop():
    return (pvzemu.OP_NOOP, 0, 0)


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


def _design_candidates():
    """The design's shared candidate list: 49 types x 54 cells, 54 shovels, 6 cob rows."""
    plants = [(t, r, c) for t in range(N_TYPES) for r in range(POOL_ROWS) for c in range(COLS)]
    shovels = [(SHOVEL, r, c) for r in range(POOL_ROWS) for c in range(COLS)]
    cobs = [(COB_FIRE, r, CLICK_X) for r in range(POOL_ROWS)]
    return plants + shovels + cobs


CANDS = _design_candidates()
assert len(CANDS) == 2706
COB_SLICE = slice(N_TYPES * 54 + 54, None)
SMALL_CANDS = [
    (PLANT, *FREE),
    (PLANT, *TAKEN),
    (PLANT, POOL_ROWS, 0),
    (int(_P.wallnut), *FREE2),
    (int(_P.lily_pad), 2, 0),
    (int(_P.coffee_bean), *SLEEPER_CELL),
    (int(_P.imitater), *FREE),
    (SHOVEL, *TAKEN),
    (SHOVEL, *FREE),
    (PUMPKIN_SHOVEL, *TAKEN),
    (COB_FIRE, 2, CLICK_X),
    (COB_FIRE, POOL_ROWS, CLICK_X),
]


def _new_world(seed=SEED, cards=CARDS, imitater=_P.none, build=True):
    w = pvzemu.World(pvzemu.SceneType.pool, seed)
    assert w.select_plants(cards, imitater) is True
    if build:
        assert w.build(_defence_list()) is True
    return w


def _cobs(w):
    return [p for p in w.scene.plants if p.type == _P.cob_cannon]


@functools.lru_cache(maxsize=None)
def _armed_base():
    """The defence with both cobs armed and no zombies; tests clone it, never edit it."""
    w = _new_world()
    w.scene.stop_spawn = True
    for _ in range(ARM_CAP):
        if all(p.status == _PS.cob_cannon_armed_idle for p in _cobs(w)):
            return w
        w.update()
    raise AssertionError("cobs did not arm within the cap")


@functools.lru_cache(maxsize=None)
def _loadout_b_base():
    """Loadout B (imitater, coffee bean) over the defence, with a sleeping fume-shroom."""
    w = _new_world(cards=CARDS_B, imitater=IMITATER_B)
    w.scene.stop_spawn = True
    assert w.plant_factory.create(_P.fumeshroom, *SLEEPER_CELL).is_sleeping
    return w


@functools.lru_cache(maxsize=None)
def _midgame_base():
    w = _new_world()
    for _ in range(MIDGAME_TICKS):
        w.update()
    assert len(w.scene.zombies) > 0
    return w


@functools.lru_cache(maxsize=None)
def _game_over_base():
    w = _new_world(build=False)
    for _ in range(GAME_OVER_CAP):
        if w.scene.is_game_over:
            return w
        w.update()
    raise AssertionError("no game over within the cap")


def _armed():
    return _armed_base().clone()


def _mask_row(w, cands):
    m = pvzemu.IntVector()
    w.get_available_actions(cands, m)
    lst = list(m)
    assert len(lst) == len(cands) + 1 and lst[-1] == 1
    return np.array(lst[:-1], dtype=np.uint8)


def _reference(w, action, frames):
    """update(action), then up to frames - 1 update() calls, stopping once one returns true
    (world.cpp:319-322)."""
    done = w.update(action)
    n = 1
    while n < frames and not done:
        done = w.update()
        n += 1


def _arrays(n, cands, actions=None, active=None, sun_set=None):
    acts = np.array(actions if actions is not None else [_noop()] * n, np.int32).reshape(n, 3)
    act = np.ones(n, np.uint8) if active is None else np.array(active, np.uint8)
    sun = np.full(n, -1, np.int32) if sun_set is None else np.array(sun_set, np.int32)
    masks = np.full((n, len(cands)), SENTINEL, np.uint8)
    applied = np.full(n, SENTINEL, np.uint8)
    return acts, act, sun, masks, applied


def _step(worlds, actions, frames, cands=SMALL_CANDS, active=None, sun_set=None):
    acts, act, sun, masks, applied = _arrays(len(worlds), cands, actions, active, sun_set)
    pvzemu.World.step_all(worlds, acts, act, sun, frames, cands, masks, applied)
    return masks, applied


def _snap(w):
    return w.to_json(), w.scene.tick


# ---------------------------------------------------------------- apply_action

def test_op_noop_constant():
    assert pvzemu.OP_NOOP == -4
    assert pvzemu.OP_NOOP not in (COB_FIRE, PUMPKIN_SHOVEL, SHOVEL)


def test_apply_action_legal():
    w = _armed()
    tick = w.scene.tick

    n, sun = len(w.scene.plants), w.scene.sun.sun
    cost = w.plant_factory.get_cost(_P.pea_shooter)
    assert w.apply_action((PLANT, *FREE)) is True
    assert len(w.scene.plants) == n + 1
    assert w.scene.sun.sun == sun - cost
    assert w.scene.plant_map[FREE[0]][FREE[1]].content.type == _P.pea_shooter

    assert w.apply_action((SHOVEL, *TAKEN)) is True
    assert w.scene.plant_map[TAKEN[0]][TAKEN[1]].content is None

    assert w.apply_action((PUMPKIN_SHOVEL, *TAKEN)) is True
    assert w.scene.plant_map[TAKEN[0]][TAKEN[1]].pumpkin is None

    statuses = [p.status for p in _cobs(w)]
    assert statuses.count(_PS.cob_cannon_armed_idle) == 2
    assert w.apply_action((COB_FIRE, 2, CLICK_X)) is True
    statuses = [p.status for p in _cobs(w)]
    assert statuses.count(_PS.cob_cannon_launch) == 1

    before = w.to_json()
    assert w.apply_action(_noop()) is True
    assert w.apply_action((pvzemu.OP_NOOP, POOL_ROWS, -5)) is True
    assert w.to_json() == before
    assert w.scene.tick == tick


def test_apply_action_illegal():
    w = _armed()
    assert w.apply_action((PLANT, *FREE)) is True
    tick, before = w.scene.tick, w.to_json()
    illegal = [
        (PLANT, *FREE2),  # the pea shooter card is now on cooldown
        (int(_P.sunflower), *TAKEN),  # occupied cell
        (int(_P.kernelpult), *FREE2),  # not in hand
        (SHOVEL, *FREE2),  # empty cell
        (PUMPKIN_SHOVEL, *FREE2),  # no pumpkin
        (int(_P.sunflower), POOL_ROWS, 0),
        (int(_P.sunflower), -1, 0),
        (int(_P.sunflower), 0, COLS),
        (int(_P.sunflower), 0, -1),
        (SHOVEL, POOL_ROWS, 3),
        (SHOVEL, 0, COLS),
        (COB_FIRE, POOL_ROWS, CLICK_X),
        (COB_FIRE, -1, CLICK_X),
        (-5, *FREE2),  # not an op
    ]
    for a in illegal:
        assert w.apply_action(a) is False, a
        assert w.to_json() == before, a
    assert w.scene.tick == tick

    # No armed cob: a fresh defence's cobs start unarmed (cob_cannon.cpp:43-48).
    fresh = _new_world()
    before = fresh.to_json()
    assert fresh.apply_action((COB_FIRE, 2, CLICK_X)) is False
    assert fresh.to_json() == before and fresh.scene.tick == 0


def _kind_actions():
    return [
        (PLANT, *FREE),
        (int(_P.sunflower), *TAKEN),
        (SHOVEL, *TAKEN),
        (SHOVEL, *FREE),
        (PUMPKIN_SHOVEL, *TAKEN),
        (COB_FIRE, 2, CLICK_X),
        (COB_FIRE, POOL_ROWS, CLICK_X),
        (PLANT, POOL_ROWS, 0),
        _noop(),
    ]


def test_update_action_is_apply_then_update():
    for a in _kind_actions():
        x, y = _armed(), _armed()
        rx = x.update(a)
        y.apply_action(a)
        ry = y.update()
        assert rx == ry, a
        assert x.to_json() == y.to_json(), a
        assert x.scene.tick == y.scene.tick == _armed_base().scene.tick + 1


def test_noop_ticks_without_touching_board():
    x, y = _armed(), _armed()
    x.update(_noop())
    y.update()
    assert x.to_json() == y.to_json()

    z = _armed()
    masks, applied = _step([z], [_noop()], 3)
    w = _armed()
    for _ in range(3):
        w.update()
    assert z.to_json() == w.to_json()
    assert applied[0] == 1


# ---------------------------------------------------------------- step_all

def _mixed_worlds():
    """(world, action) pairs: every action kind, legal and illegal, on varied boards."""
    pairs = [(_armed(), a) for a in _kind_actions()]
    pairs.append((_midgame_base().clone(), (int(_P.wallnut), *FREE2)))
    pairs.append((_midgame_base().clone(), _noop()))
    pairs.append((_loadout_b_base().clone(), (int(_P.coffee_bean), *SLEEPER_CELL)))
    pairs.append((_loadout_b_base().clone(), (int(_P.imitater), *FREE)))
    return pairs


@pytest.mark.parametrize("frames", FRAMES)
@pytest.mark.parametrize("cands", [CANDS, SMALL_CANDS], ids=["design", "small"])
def test_step_all_equals_update(frames, cands):
    pairs = _mixed_worlds()
    worlds = [w for w, _ in pairs]
    actions = [a for _, a in pairs]
    refs = [w.clone() for w in worlds]
    probes = [w.clone() for w in worlds]
    masks, applied = _step(worlds, actions, frames, cands)

    for k, (w, a) in enumerate(pairs):
        _reference(refs[k], a, frames)
        assert w.to_json() == refs[k].to_json(), (k, a)
        assert w.scene.tick == refs[k].scene.tick
        assert applied[k] == int(probes[k].apply_action(a)), (k, a)
        assert np.array_equal(masks[k], _mask_row(w, cands)), (k, a)
    # Each kind took effect at least once, and an illegal action reported 0.
    assert applied.tolist()[:len(_kind_actions())] == [1, 0, 1, 0, 1, 1, 0, 0, 1]
    assert applied[-2] == 1 and applied[-1] == 1


def test_masks_absolute_and_masks_all():
    armed, fresh = _armed(), _new_world()
    b = _loadout_b_base().clone()
    worlds = [armed, fresh, b, _midgame_base().clone()]
    snaps = [_snap(w) for w in worlds]
    masks = np.full((len(worlds), len(CANDS)), SENTINEL, np.uint8)
    pvzemu.World.masks_all(worlds, CANDS, masks)
    assert [_snap(w) for w in worlds] == snaps
    for k, w in enumerate(worlds):
        assert np.array_equal(masks[k], _mask_row(w, CANDS)), k

    # Facts the shared rules must give, checked without get_available_actions.
    assert masks[0, COB_SLICE].tolist() == [1] * POOL_ROWS  # two armed cobs
    assert masks[1, COB_SLICE].tolist() == [0] * POOL_ROWS  # cobs still unarmed
    idx = {a: i for i, a in enumerate(CANDS)}
    assert masks[0, idx[(PLANT, *FREE)]] == 1
    assert masks[0, idx[(PLANT, *TAKEN)]] == 0
    assert masks[0, idx[(SHOVEL, *TAKEN)]] == 1
    assert masks[0, idx[(SHOVEL, *FREE)]] == 0
    assert masks[0, idx[(int(_P.kernelpult), *FREE)]] == 0  # not in hand
    assert masks[2, idx[(int(_P.coffee_bean), *SLEEPER_CELL)]] == 1
    assert masks[2, idx[(int(_P.imitater), *FREE)]] == 1
    assert masks[2, idx[(int(_P.doomshroom), *FREE)]] == 0  # only through the imitater

    small = np.full((len(worlds), len(SMALL_CANDS)), SENTINEL, np.uint8)
    pvzemu.World.masks_all(worlds, SMALL_CANDS, small)
    for k, w in enumerate(worlds):
        assert np.array_equal(small[k], _mask_row(w, SMALL_CANDS)), k

    pvzemu.World.masks_all([], CANDS, np.zeros((0, len(CANDS)), np.uint8))


def test_game_over_world_untouched():
    w = _game_over_base().clone()
    assert w.scene.is_game_over
    a = (PLANT, *FREE)
    assert _mask_row(w, [a]).tolist() == [1]  # the plant would be legal
    before = _snap(w)
    sun = w.scene.sun.sun
    masks, applied = _step([w], [a], 10, CANDS, sun_set=[sun + 100])
    assert applied[0] == 0
    assert _snap(w) == before
    assert w.scene.sun.sun == sun
    assert np.array_equal(masks[0], _mask_row(w, CANDS))


def test_inactive_worlds_untouched():
    pairs = _mixed_worlds()
    worlds = [w for w, _ in pairs]
    actions = [a for _, a in pairs]
    active = [k % 2 for k in range(len(worlds))]
    sun_set = [123] * len(worlds)
    refs = [w.clone() for w in worlds]
    snaps = [_snap(w) for w in worlds]
    masks, applied = _step(worlds, actions, 3, CANDS, active=active, sun_set=sun_set)
    for k, w in enumerate(worlds):
        if active[k]:
            _reference(refs[k], actions[k], 3)
            refs[k].scene.sun.sun = 123
            assert w.to_json() == refs[k].to_json(), k
            assert np.array_equal(masks[k], _mask_row(w, CANDS)), k
        else:
            assert _snap(w) == snaps[k], k
            assert applied[k] == SENTINEL, k
            assert set(masks[k].tolist()) == {SENTINEL}, k


def test_sun_set_after_ticks():
    w = _armed()
    while w.scene.sun.natural_sun_countdown > 5:
        w.update()
    countdown = w.scene.sun.natural_sun_countdown
    generated = w.scene.sun.natural_sun_generated
    ref = w.clone()
    keep, zero = w.clone(), w.clone()

    # The pea shooter is bought before the ticks; natural sun lands during them (sun.cpp:23-24),
    # so a sun_set applied any earlier would not survive to the end.
    masks, applied = _step([w, keep, zero], [(PLANT, *FREE)] * 3, countdown + 2, CANDS,
                           sun_set=[1000, -1, 0])
    assert applied.tolist() == [1, 1, 1]
    assert w.scene.sun.natural_sun_generated == generated + 1
    assert w.scene.sun.sun == 1000
    assert zero.scene.sun.sun == 0
    _reference(ref, (PLANT, *FREE), countdown + 2)
    assert keep.scene.sun.sun == ref.scene.sun.sun != 1000
    assert keep.to_json() == ref.to_json()
    ref.scene.sun.sun = 1000
    assert w.to_json() == ref.to_json()

    # The masks see the new sun: nothing is affordable at 0.
    assert np.array_equal(masks[2], _mask_row(zero, CANDS))
    assert masks[2, :N_TYPES * 54].sum() == 0
    assert masks[0, :N_TYPES * 54].sum() > 0


@pytest.mark.parametrize("endgame", [1, 3])
def test_round_end_stops_frame_skip(endgame):
    w = _new_world()
    for _ in range(WAVE_ONE_TICKS):
        w.update()
    assert w.scene.spawn.wave >= 1
    w.scene.spawn.countdown.endgame = endgame
    ref = w.clone()
    tick, flags = w.scene.tick, w.scene.spawn.total_flags

    masks, applied = _step([w], [_noop()], 10, CANDS)
    assert w.scene.tick - tick == endgame < 10
    assert w.scene.spawn.total_flags == flags + 1
    assert w.scene.spawn.wave == 0
    assert w.scene.spawn.countdown.endgame == 0
    _reference(ref, _noop(), 10)
    assert w.to_json() == ref.to_json()
    assert np.array_equal(masks[0], _mask_row(w, CANDS))


def test_threaded_equals_serial():
    worlds = [_new_world(seed) for seed in range(N_THREADED)]
    serial = [w.clone() for w in worlds]
    script = [
        lambda k, t: (COB_FIRE, (k + t) % POOL_ROWS, CLICK_X),
        lambda k, t: (PLANT, *FREE),
        lambda k, t: (SHOVEL, *FREE),
        lambda k, t: _noop(),
        lambda k, t: (int(_P.wallnut), *FREE2),
        lambda k, t: (SHOVEL, (k + t) % POOL_ROWS, (k * t) % COLS),
        lambda k, t: (int(_P.pumpkin), *FREE),
        lambda k, t: (int(_P.lily_pad), 2 + t % 2, (k + t) % COLS),
    ]
    n = len(worlds)
    for t in range(THREADED_CALLS):
        cands = CANDS if t == THREADED_CALLS - 1 else SMALL_CANDS
        actions = [script[(3 * k + t) % len(script)](k, t) for k in range(n)]
        active = [int((k + t) % 9 != 0) for k in range(n)]
        sun_set = [500 if k % 4 == 0 else -1 for k in range(n)]
        masks, applied = _step(worlds, actions, THREADED_FRAMES, cands, active, sun_set)
        for k, s in enumerate(serial):
            if not active[k]:
                assert applied[k] == SENTINEL and set(masks[k].tolist()) == {SENTINEL}
                continue
            if s.scene.is_game_over:
                assert applied[k] == 0
            else:
                assert applied[k] == int(s.clone().apply_action(actions[k])), (t, k)
                _reference(s, actions[k], THREADED_FRAMES)
                if sun_set[k] >= 0:
                    s.scene.sun.sun = sun_set[k]
            assert np.array_equal(masks[k], _mask_row(s, cands)), (t, k)
        if t % 10 == 9:
            for k in range(n):
                assert worlds[k].to_json() == serial[k].to_json(), (t, k)
    assert any(w.scene.spawn.wave >= 1 for w in worlds)


# ---------------------------------------------------------------- validation

def _good(n, cands):
    return _arrays(n, cands)


def _assert_untouched(worlds, snaps, masks, applied):
    assert [_snap(w) for w in worlds] == snaps
    assert set(masks.ravel().tolist()) <= {SENTINEL}
    assert set(applied.tolist()) <= {SENTINEL}


def test_step_all_validation():
    worlds = [_armed(), _armed()]
    snaps = [_snap(w) for w in worlds]
    cands = SMALL_CANDS
    m = len(cands)
    acts, act, sun, masks, applied = _good(2, cands)
    step = pvzemu.World.step_all

    def ro(a):
        a = a.copy()
        a.setflags(write=False)
        return a

    bad = {
        "actions int64": dict(actions=acts.astype(np.int64)),
        "actions shape (2, 2)": dict(actions=np.zeros((2, 2), np.int32)),
        "actions shape (3, 3)": dict(actions=np.zeros((3, 3), np.int32)),
        "actions 1-D": dict(actions=np.zeros(6, np.int32)),
        "actions strided": dict(actions=np.zeros((2, 6), np.int32)[:, ::2]),
        "actions Fortran": dict(actions=np.asfortranarray(np.zeros((2, 3), np.int32))),
        "active bool": dict(active=np.ones(2, bool)),
        "active int32": dict(active=np.ones(2, np.int32)),
        "active shape": dict(active=np.ones(3, np.uint8)),
        "active strided": dict(active=np.ones(4, np.uint8)[::2]),
        "sun_set int64": dict(sun_set=np.full(2, -1, np.int64)),
        "sun_set shape": dict(sun_set=np.full((2, 1), -1, np.int32)),
        "masks int32": dict(masks=np.zeros((2, m), np.int32)),
        "masks bool": dict(masks=np.zeros((2, m), bool)),
        "masks shape (2, m + 1)": dict(masks=np.zeros((2, m + 1), np.uint8)),
        "masks 1-D": dict(masks=np.zeros(2 * m, np.uint8)),
        "masks strided": dict(masks=np.zeros((2, 2 * m), np.uint8)[:, ::2]),
        "masks read-only": dict(masks=ro(masks)),
        "applied bool": dict(applied=np.zeros(2, bool)),
        "applied shape": dict(applied=np.zeros(3, np.uint8)),
        "applied read-only": dict(applied=ro(applied)),
        "frames 0": dict(frames=0),
        "frames -1": dict(frames=-1),
        "None world": dict(worlds=[worlds[0], None]),
        "same world twice": dict(worlds=[worlds[0], worlds[0]]),
    }
    for name, over in bad.items():
        args = dict(worlds=worlds, actions=acts, active=act, sun_set=sun, frames=1,
                    masks=masks, applied=applied)
        args.update(over)
        with pytest.raises(ValueError) as e:
            step(args["worlds"], args["actions"], args["active"], args["sun_set"],
                 args["frames"], cands, args["masks"], args["applied"])
        assert "expected" in str(e.value) or "None" in str(e.value) \
            or "twice" in str(e.value), (name, str(e.value))
        _assert_untouched(worlds, snaps, masks, applied)

    with pytest.raises(TypeError):
        step(worlds, acts.tolist(), act, sun, 1, cands, masks, applied)
    with pytest.raises(TypeError):
        step(worlds, acts, act, sun, 1, cands, masks.tolist(), applied)
    _assert_untouched(worlds, snaps, masks, applied)

    step(worlds, acts, act, sun, 1, cands, masks, applied)
    assert all(w.scene.tick == s[1] + 1 for w, s in zip(worlds, snaps))
    pvzemu.World.step_all([], np.zeros((0, 3), np.int32), np.zeros(0, np.uint8),
                          np.zeros(0, np.int32), 1, cands, np.zeros((0, m), np.uint8),
                          np.zeros(0, np.uint8))


def test_masks_all_validation():
    worlds = [_armed(), _armed()]
    m = len(SMALL_CANDS)
    masks = np.full((2, m), SENTINEL, np.uint8)
    bad = [
        np.zeros((2, m), np.int32),
        np.zeros((2, m), bool),
        np.zeros((3, m), np.uint8),
        np.zeros((2, m - 1), np.uint8),
        np.zeros(2 * m, np.uint8),
        np.zeros((2, 2 * m), np.uint8)[:, ::2],
    ]
    ro = masks.copy()
    ro.setflags(write=False)
    bad.append(ro)
    for out in bad:
        with pytest.raises(ValueError, match="expected"):
            pvzemu.World.masks_all(worlds, SMALL_CANDS, out)
    with pytest.raises(ValueError, match="None"):
        pvzemu.World.masks_all([worlds[0], None], SMALL_CANDS, masks)
    with pytest.raises(TypeError):
        pvzemu.World.masks_all(worlds, SMALL_CANDS, masks.tolist())
    assert set(masks.ravel().tolist()) == {SENTINEL}


# ---------------------------------------------------------------- endgame hook

def test_endgame_writable_rest_unchanged():
    w = _new_world()
    c = w.scene.spawn.countdown
    assert c.endgame == 0
    c.endgame = 7
    assert w.scene.spawn.countdown.endgame == 7
    # to_json shows it under its own key (object/scene.cpp:162-163), beside hugewave_fade.
    j = w.to_json()
    assert '"endgame":7' in j
    assert '"hugewave_fade":%d' % c.hugewave_fade in j
    for name in ("next_wave", "next_wave_initial", "lurking_squad", "hugewave_fade", "pool"):
        value = getattr(c, name)
        assert isinstance(value, int)
        with pytest.raises(AttributeError):
            setattr(c, name, value)
