"""Step 1.6: World.fill_state writes the shared pvz_state_t (schema v1) from a world.

The expected values are built in Python from the existing bindings and the adapter's rules
(pvz-rl schema/pvz_state.h, the emulator: lines on each field), then compared field by field
and byte by byte with the buffer the adapter wrote.

Emulator facts used here (cited as emulator <path>:<line>):
- object/obj_list.h:26-39: the list iterator skips freed slots and slots whose object
  is_freeable(), in ascending slot order; object/plant.h:217-219 and object/zombie.h:255-257:
  a plant or zombie is freeable once is_dead; object/griditem.h:26-28: a grid item once
  is_disappeared. obj_list.h:138-145: get_index(obj) is the object's slot index.
- obj_list.h:158-188: size() returns n_actives, which only shrink_to_fit() recounts;
  world.cpp:16-21 and 31: world::update() shrinks the lists at its start.
  system/plant/plant_factory.cpp:548-549: destroy() sets is_dead;
  system/zombie/zombie_factory.cpp:183-184: so does the zombie factory's destroy().
- object/scene.h:52-54: list capacities zombies 1024, plants 512, grid items 128.
- object/scene.h:85: spawn_flags has 33 entries.
- object/plant.h:96-99: cob statuses unarmed_idle 0x23, charge 0x24, launch 0x25, armed_idle 0x26.
  system/plant/cob_cannon.cpp:9-13: an unarmed cob starts charging, with a once animation, when
  countdown.status is 0; cob_cannon.cpp:16-20: it arms once reanim.n_repeated > 0;
  cob_cannon.cpp:28-33: launch() turns an armed cob into launch; cob_cannon.cpp:46-47: a new cob
  is unarmed with countdown.status 500; system/plant/plant_system.cpp:434-438: a launching cob
  goes back to unarmed with countdown.status 3000 once its animation has played;
  plant_system.cpp:470-472: countdown.status counts down to 0 and stays there.
- object/plant.cpp:252-254: set_reanim zeroes n_repeated and progress (fps >= 0);
  system/reanim.cpp:163-164: progress advances each update; reanim.cpp:175-178: a once
  animation stops at exactly 1 and sets n_repeated.
- world.cpp:85-89: op -3 fires the next armed cob at (row, x).
- object/plant.cpp:38-45: CD_TABLE, 48 entries; system/plant/plant_factory.cpp:466 and :543: a
  planted card's cooldown is CD_TABLE[target], target = imitater_type for an imitater card.
- world.cpp:126: a card counts as usable when its type is not none and cold_down == 0;
  world.cpp:409-414: select_plants sets every slot past the given list to type none.
- object/zombie.h:231-234: is_flying_or_falling(); zombie.h:240-244: has_death_status().
  system/damage.cpp:124, 157 and 207: set_death_state() sets status dying and then destroys
  the zombie, so an ordinary death is never a live entry. The ash path keeps one:
  system/plant/plant.h:99 a cherry bomb starts with countdown.effect 100;
  system/plant/plant_system.cpp:629-630 it activates when that reaches 0;
  system/damage.cpp:390 its centre is attack_box.width / 2 + x; damage.cpp:401-402 it hits
  zombies within one row whose hit box overlaps a 115 px circle, as an ash attack
  (damage.cpp:552-564); object/plant.cpp:176-186 its attack flags include flying_balloon and
  damage.cpp:73-77 that reaches a flying balloon, while damage.cpp:55-56 skips a hit box past
  x 800; object/zombie.cpp:105-115 get_hit_box() is in board pixels; damage.cpp:326-338 an
  ash attack on a flying balloon sets dying_from_instant_kill for 300 updates instead of
  destroying it.
- system/damage.cpp:647-655: a first accessory shot to 0 HP gets type none and keeps max_hp.
- system/zombie/balloon.cpp:22: a new balloon zombie is balloon_flying.
- system/zombie/zombie_factory.cpp:269-273: a ducky tube or snorkel zombie spawns in pool row 2 or 3.
- system/plant/plant_base.cpp:219-223: a mushroom created in a pool scene goes to sleep;
  system/plant/plant.h:157-159: the fume-shroom uses that init.
- system/plant/plant_system.cpp:450-454: a smashed plant is destroyed when countdown.dead runs out.
- system/griditem_factory.h:18: create(type, row, col); system/griditem_factory.cpp:29-30: a
  crater starts at countdown 18000.
- world.cpp:23-29: once is_game_over is set, update() returns before advancing tick.
- object/scene.h:171-175: pool water rows are 2 and 3, pool has 6 rows.
- object/scene.h:91 and object/scene.cpp:288 (reset): spawn.total_flags starts at a nonzero
  value, read here from a fresh world; step 1.8: the adapter writes flag = total_flags minus it.
  system/spawn.cpp:421-425: total_flags + 1 when wave reaches 10; world.cpp:62-65: + 1 at a
  round end, which system/endgame.h:12-14 signals on the update that counts countdown.endgame
  from 1 to 0. system/spawn.cpp:390-396: the next wave comes 200 ticks early once the zombies'
  HP is at or below the threshold, so clearing the zombies speeds the waves up.
"""
import collections
import time

import numpy as np
import pytest

import pvzemu
import schema

D = schema.PVZ_STATE_DTYPE
HDR_DT = D.fields["hdr"][0]
CARD_DT = D.fields["cards"][0].subdtype[0]
REC_DT = {m: D.fields[m][0].subdtype[0] for m, _, _ in schema.ARRAYS}
CAP = dict(schema.CAPS)
HEAD_BYTES = D.fields["plants"][1]  # header and all cards come before plants[]
LIVE_ARRAYS = ("plants", "zombies", "griditems")  # the emulator writes no mowers

COB_FIRE = -3  # step 1.2 action op, emulator world.cpp:85
SHOVEL = -1  # emulator world.cpp:96
POOL_ROWS = 6  # emulator object/scene.h:175
WATER_ROWS = (2, 3)  # emulator object/scene.h:171
N_SPAWN_FLAGS = 33  # emulator object/scene.h:85
N_CD_TABLE = 48  # emulator object/plant.cpp:38
COB_REARM = 3000  # emulator system/plant/plant_system.cpp:437
# the flag counter's start value (emulator object/scene.h:91), read from a fresh world
START_FLAGS = pvzemu.World(pvzemu.SceneType.pool, 0).scene.spawn.total_flags

_P = pvzemu.PlantType
_Z = pvzemu.ZombieType
_PS = pvzemu.PlantStatus
_A1 = pvzemu.ZombieAccessoriesType1
_A2 = pvzemu.ZombieAccessoriesType2
_G = pvzemu.GriditemType

COB_STATES = {  # the adapter rule, schema/pvz_state.h cob_state
    int(_PS.cob_cannon_unarmed_idle): schema.COB_STATE_UNARMED,
    int(_PS.cob_cannon_charge): schema.COB_STATE_CHARGING,
    int(_PS.cob_cannon_armed_idle): schema.COB_STATE_ARMED,
    int(_PS.cob_cannon_launch): schema.COB_STATE_FIRING,
}

# Test choices, not game facts: the 1.4/1.5 cards, defence and cannon cells; extra plants,
# grid items and zombies placed so every field rule has something to bite on; tick counts,
# sampling strides and the timing bound.
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
COB_CELLS = [(1, 0), (4, 0)]
LAND_DEFENCE = [None, None, _P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
WATER_DEFENCE = [_P.cattail] * 2 + [_P.winter_melon] + [_P.gatling_pea] * 4 + [_P.tallnut] * 2
CLICK_X = 600
SLEEPER = (_P.fumeshroom, 0, 0)
SMASHED = (_P.wallnut, 5, 0)
GRIDITEMS = [(_G.ladder, 0, 0), (_G.grave, 5, 1), (_G.crater, 3, 8)]
EXTRA_ZOMBIES = [
    _Z.conehead, _Z.buckethead, _Z.football,
    _Z.screendoor, _Z.newspaper, _Z.ladder,
    _Z.balloon, _Z.balloon,
    _Z.ducky_tube, _Z.snorkel,
]
FREE_CELL = (0, 1)
SCRIPT = {  # tick -> action, besides the cob fires
    50: (int(_P.pea_shooter), *FREE_CELL),
    2000: (SHOVEL, *FREE_CELL),
    2100: (int(_P.wallnut), *FREE_CELL),
}
SCENARIO_TICKS = 4500
STRIDE = 10
COB_LEAD = 5  # compare every tick from this many ticks before a charge starts
COB_TAIL = 5  # ... and for this many ticks after a cob is back to unarmed
BALLOON_CAP = 5000
BALLOON_BOMB_X = 500  # plant the bomb once the balloon's hit box is this far in
BOMB_WATCH = 200  # updates watched after the bomb is planted
MIDGAME_TICKS = 1500
TIMING_FILLS = 2000
BATCH = 64
FILL_US_BOUND = 100.0  # loose: fill_state is a few thousand stores
BATCH_MS_BOUND = 10.0
WAVE_10_CAP = 20_000


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


def _scenario_world(seed=SEED):
    w = pvzemu.World(pvzemu.SceneType.pool, seed)
    assert w.select_plants(CARDS, _P.none) is True
    assert w.build(_defence_list()) is True
    sleeper = w.plant_factory.create(*SLEEPER)
    assert sleeper.is_sleeping
    smashed = w.plant_factory.create(*SMASHED)
    smashed.is_smashed = True
    for t, r, c in GRIDITEMS:
        w.griditem_factory.create(t, r, c)
    for t in EXTRA_ZOMBIES:
        w.zombie_factory.create(t, None)
    return w


def _cobs(w):
    return [p for p in w.scene.plants if p.type == _P.cob_cannon]


def _scenario_step(w, tick):
    """One update: fire as soon as a cob is armed, else the scripted action, else none."""
    if any(p.status == _PS.cob_cannon_armed_idle for p in _cobs(w)):
        w.update((COB_FIRE, tick % POOL_ROWS, CLICK_X))
    elif tick in SCRIPT:
        w.update(SCRIPT[tick])
    else:
        w.update()


def _cob_window(w):
    """True from COB_LEAD ticks before a charge through COB_TAIL ticks after re-arming starts."""
    for p in _cobs(w):
        if p.status != _PS.cob_cannon_unarmed_idle:
            return True
        if p.countdown.status <= COB_LEAD or p.countdown.status >= COB_REARM - COB_TAIL:
            return True
    return False


def _midgame_world(ticks=MIDGAME_TICKS):
    w = _scenario_world()
    for tick in range(ticks):
        _scenario_step(w, tick)
    return w


def _buf(n=1, fill=0):
    b = np.empty(n, dtype=D)
    b.view(np.uint8)[:] = fill
    return b


def _filled(w, fill=0):
    b = _buf(1, fill)
    w.fill_state(b)
    return b


# ---------------------------------------------------------------- expected values

def _expected_valid():
    bits = [getattr(schema, n) for n in dir(schema) if n.startswith("VALID_BIT_")]
    assert sorted(bits) == list(range(29))
    return sum(1 << b for b in bits if b != schema.VALID_BIT_MOWERS)


EXPECTED_VALID = _expected_valid()


def _card(c):
    t = int(c.type)
    if t == schema.CARD_TYPE_EMPTY:
        cd_total = 0
    else:
        target = int(c.imitater_type) if c.type == _P.imitater else t
        cd_total = pvzemu.CD_TABLE[target]
    return dict(type=t, imitater_type=int(c.imitater_type), cd_remaining=c.cold_down,
                cd_total=cd_total, usable=int(t != schema.CARD_TYPE_EMPTY and c.cold_down == 0))


def _plant(lst, p):
    cob = COB_STATES.get(int(p.status), schema.COB_STATE_NOT_COB) \
        if p.type == _P.cob_cannon else schema.COB_STATE_NOT_COB
    flags = (schema.PLANT_FLAG_SLEEPING if p.is_sleeping else 0) | \
        (schema.PLANT_FLAG_SMASHED if p.is_smashed else 0)
    return dict(
        id=lst.get_index(p), type=int(p.type), row=p.row, col=p.col, status_raw=int(p.status),
        x=np.float32(p.x), y=np.float32(p.y), hp=p.hp, max_hp=p.max_hp,
        state_countdown=p.countdown.status, shoot_countdown=p.countdown.generate,
        effect_countdown=p.countdown.effect, cob_state=cob,
        cob_recharge_countdown=p.countdown.status if cob == schema.COB_STATE_UNARMED else 0,
        cob_charge_progress=np.float32(
            p.reanim.progress if cob == schema.COB_STATE_CHARGING else 0.0),
        flags=flags)


def _zombie(lst, z):
    a1, a2 = z.accessory_1, z.accessory_2
    n1, n2 = a1.type == _A1.none, a2.type == _A2.none
    flags = 0
    for on, bit in ((z.is_eating, schema.ZOMBIE_FLAG_EATING),
                    (z.is_in_water, schema.ZOMBIE_FLAG_IN_WATER),
                    (z.is_flying_or_falling(), schema.ZOMBIE_FLAG_FLYING),
                    (z.has_death_status(), schema.ZOMBIE_FLAG_DYING),
                    (z.is_hypno, schema.ZOMBIE_FLAG_HYPNO)):
        flags |= bit if on else 0
    return dict(
        id=lst.get_index(z), type=int(z.type), row=z.row, status_raw=int(z.status),
        x=np.float32(z.x), y=np.float32(z.y), hp=z.hp, max_hp=z.max_hp,
        acc1_type=int(a1.type), acc1_hp=0 if n1 else a1.hp, acc1_max_hp=0 if n1 else a1.max_hp,
        acc2_type=int(a2.type), acc2_hp=0 if n2 else a2.hp, acc2_max_hp=0 if n2 else a2.max_hp,
        slow_countdown=z.countdown.slow, freeze_countdown=z.countdown.freeze,
        butter_countdown=z.countdown.butter, spawn_wave=z.spawn_wave, flags=flags, _pad=0)


def _griditem(lst, g):
    return dict(type=int(g.type), row=g.row, col=g.col, countdown_raw=g.countdown)


_MAKERS = {"plants": _plant, "zombies": _zombie, "griditems": _griditem}


def _records(dt, dicts):
    return np.array([tuple(d[n] for n in dt.names) for d in dicts], dtype=dt)


def _expected(w):
    """(header, cards, {member: entries}, raw accessories of the zombie entries) for w."""
    s = w.scene
    lists = {"plants": s.plants, "zombies": s.zombies, "griditems": s.griditems}
    arrays, counts = {}, {}
    raw_acc = []
    for member, lst in lists.items():
        live = list(lst)  # the iterator: live slots in ascending order
        cap = CAP[member]
        arrays[member] = [_MAKERS[member](lst, x) for x in live[:cap]]
        counts[member] = (min(len(live), cap), max(len(live) - cap, 0))
        if member == "zombies":
            raw_acc = [(z.accessory_1.type, z.accessory_1.max_hp,
                        z.accessory_2.type, z.accessory_2.max_hp) for z in live[:cap]]
    sp = s.spawn
    spawn_flags = [1 if f else 0 for f in sp.spawn_flags]
    assert len(spawn_flags) == N_SPAWN_FLAGS
    hdr = dict(
        magic=schema.MAGIC, schema_version=schema.SCHEMA_VERSION,
        layout_hash=schema.LAYOUT_HASH_U32, struct_size=schema.STATE_SIZE,
        world=schema.WORLD_EMULATOR, valid=EXPECTED_VALID, tick=s.tick,
        phase=schema.PHASE_GAME_OVER if s.is_game_over else schema.PHASE_PLAYING, paused=0,
        scene=int(s.type), flag=sp.total_flags - START_FLAGS, wave=sp.wave,
        next_wave_countdown=sp.countdown.next_wave, next_wave_is_huge=int(sp.wave % 10 == 9),
        sun=s.sun.sun,
        n_plants=counts["plants"][0], plants_overflow=counts["plants"][1],
        n_zombies=counts["zombies"][0], zombies_overflow=counts["zombies"][1],
        n_griditems=counts["griditems"][0], griditems_overflow=counts["griditems"][1],
        n_mowers=0, mowers_overflow=0,
        round_zombie_types=spawn_flags + [0] * (36 - N_SPAWN_FLAGS))
    hdr = _records(HDR_DT, [hdr])
    cards = _records(CARD_DT, [_card(c) for c in s.cards])
    arrays = {m: _records(REC_DT[m], arrays[m]) for m in arrays}
    return hdr, cards, arrays, raw_acc


def _compare(where, name, got, exp):
    assert got.shape == exp.shape, "%s: %s shape %s, expected %s" % (where, name, got.shape,
                                                                     exp.shape)
    for f in exp.dtype.names:
        g, e = got[f], exp[f]
        if not np.array_equal(g, e):
            diff = np.nonzero((g != e).reshape(len(e), -1).any(axis=1))[0][0]
            raise AssertionError("%s: %s[%d].%s: got %r, expected %r" % (
                where, name, diff, f, g[diff], e[diff]))
    assert got.tobytes() == exp.tobytes(), "%s: %s bytes differ" % (where, name)


def _check(w, where, buf=None):
    """Fill (unless buf is given), compare every written field with _expected(w)."""
    if buf is None:
        buf = _filled(w)
    hdr, cards, arrays, raw_acc = _expected(w)
    _compare(where, "hdr", buf["hdr"], hdr)
    _compare(where, "cards", buf["cards"][0], cards)
    for m in LIVE_ARRAYS:
        n = len(arrays[m])
        _compare(where, m, buf[m][0][:n], arrays[m])
    return buf, raw_acc


# ---------------------------------------------------------------- tests

def test_constants_and_header():
    assert pvzemu.STATE_VERSION == schema.SCHEMA_VERSION
    assert pvzemu.STATE_SIZE == schema.STATE_SIZE
    assert pvzemu.STATE_LAYOUT_HASH == schema.LAYOUT_HASH_U32
    assert pvzemu.STATE_MAGIC == schema.MAGIC
    explicit = 0
    for name in ("CORE", "PHASE", "GAME_OVER", "WAVE", "FLAG", "ROUND_TYPES", "CARDS",
                 "PLANT_BASIC", "PLANT_POS", "PLANT_HP", "PLANT_STATUS", "PLANT_COUNTDOWNS",
                 "COB_STATE", "COB_CHARGE", "Z_BASIC", "Z_HP", "Z_MAX_HP", "ACC_HP",
                 "ACC_TYPE", "Z_STATUS", "Z_EATING", "Z_IN_WATER", "Z_FLYING", "Z_HYPNO",
                 "Z_DEBUFF", "Z_SPAWN_WAVE", "GRID", "GRID_COUNTDOWN"):
        explicit |= getattr(schema, "VALID_" + name)
    assert explicit == (1 << 28) - 1
    assert not explicit & schema.VALID_MOWERS
    assert pvzemu.STATE_VALID == explicit == EXPECTED_VALID

    h = _filled(_scenario_world())["hdr"][0]
    assert h["magic"] == schema.MAGIC
    assert h["schema_version"] == schema.SCHEMA_VERSION
    assert h["layout_hash"] == schema.LAYOUT_HASH_U32
    assert h["struct_size"] == schema.STATE_SIZE
    assert h["world"] == schema.WORLD_EMULATOR == 1
    assert h["valid"] == explicit
    assert h["n_mowers"] == 0 and h["mowers_overflow"] == 0


def _observe(cov, buf, raw_acc):
    h = buf["hdr"][0]
    cards = buf["cards"][0]
    for u in cards["usable"]:
        cov["usable_%d" % u] += 1
    pl = buf["plants"][0][:h["n_plants"]]
    is_cob = pl["type"] == int(_P.cob_cannon)
    assert (pl["cob_state"][~is_cob] == schema.COB_STATE_NOT_COB).all()
    for state in pl["cob_state"][is_cob]:
        assert state != schema.COB_STATE_NOT_COB, "cob with an unmapped status"
        cov["cob_%d" % state] += 1
    cov["sleeping"] += int(np.count_nonzero(pl["flags"] & schema.PLANT_FLAG_SLEEPING))
    cov["smashed"] += int(np.count_nonzero(pl["flags"] & schema.PLANT_FLAG_SMASHED))
    zb = buf["zombies"][0][:h["n_zombies"]]
    for name in ("EATING", "IN_WATER", "FLYING", "DYING", "HYPNO"):
        bit = getattr(schema, "ZOMBIE_FLAG_" + name)
        cov[name] += int(np.count_nonzero(zb["flags"] & bit))
    cov["acc1"] += int(np.count_nonzero(zb["acc1_type"]))
    cov["acc2"] += int(np.count_nonzero(zb["acc2_type"]))
    for i, (t1, m1, t2, m2) in enumerate(raw_acc):
        if t1 == _A1.none and m1 != 0:
            assert zb["acc1_max_hp"][i] == 0 and zb["acc1_hp"][i] == 0
            cov["acc1_stale"] += 1
        if t2 == _A2.none and m2 != 0:
            assert zb["acc2_max_hp"][i] == 0 and zb["acc2_hp"][i] == 0
            cov["acc2_stale"] += 1


def _track_cobs(cov, last, tick, buf):
    """Per cob id: charge progress rises every tick and ends at exactly 1; count transitions."""
    pl = buf["plants"][0][:buf["hdr"][0]["n_plants"]]
    for e in pl[pl["type"] == int(_P.cob_cannon)]:
        cid, state, prog = int(e["id"]), int(e["cob_state"]), e["cob_charge_progress"]
        prev = last.get(cid)
        last[cid] = (tick, state, prog)
        if prev is None or prev[0] != tick - 1:
            continue
        _, pstate, pprog = prev
        if pstate != state:
            cov["step_%d_%d" % (pstate, state)] += 1
        if pstate == state == schema.COB_STATE_CHARGING:
            assert prog > pprog, "cob %d charge progress %r after %r" % (cid, prog, pprog)
            cov["charge_rising"] += 1
        if pstate == schema.COB_STATE_CHARGING and state != pstate:
            assert pprog == np.float32(1.0), "cob %d left charging at %r" % (cid, pprog)
            cov["charge_end"] += 1


def _ash_balloon_run(cov):
    """A flying balloon hit by a cherry bomb stays on the board, dying, for a while."""
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    w.scene.stop_spawn = True
    w.zombie_factory.create(_Z.balloon, None)
    (z,) = list(w.scene.zombies)
    hit = pvzemu.Rect()
    for _ in range(BALLOON_CAP):
        z.get_hit_box(hit)
        if hit.x <= BALLOON_BOMB_X:
            break
        w.update()
    assert hit.x <= BALLOON_BOMB_X and z.is_flying_or_falling()

    # The column whose bomb centre (damage.cpp:390) is nearest the balloon's hit box centre,
    # read from bombs created in a scratch world rather than from a pixel formula.
    scratch = pvzemu.World(pvzemu.SceneType.pool, SEED)
    bombs = [scratch.plant_factory.create(_P.cherry_bomb, z.row, c) for c in range(9)]
    centres = [b.attack_box.width // 2 + b.x for b in bombs]
    col = min(range(9), key=lambda c: abs(centres[c] - (hit.x + hit.width // 2)))
    w.plant_factory.create(_P.cherry_bomb, z.row, col)

    dying = 0
    for _ in range(BOMB_WATCH):
        w.update()
        buf, raw_acc = _check(w, "ash balloon, tick %d" % w.scene.tick)
        _observe(cov, buf, raw_acc)
        dying += z.has_death_status() and not z.is_dead
    assert dying, "the cherry bomb did not leave the balloon dying"
    assert z.status == pvzemu.ZombieStatus.dying_from_instant_kill


def test_fields_match_bindings():
    w = _scenario_world()
    cov = collections.Counter()
    last = {}
    compared = dense = 0
    for tick in range(SCENARIO_TICKS + 1):
        if tick:
            _scenario_step(w, tick)
        window = _cob_window(w)
        if tick % STRIDE == 0 or window:
            buf, raw_acc = _check(w, "tick %d" % w.scene.tick)
            _observe(cov, buf, raw_acc)
            _track_cobs(cov, last, w.scene.tick, buf)
            compared += 1
            dense += window
    assert not w.scene.is_game_over
    _ash_balloon_run(cov)
    print("compared %d scenario fills (%d in cob windows), wave %d, coverage %s" % (
        compared, dense, w.scene.spawn.wave, dict(cov)))

    U, C, A, F = (schema.COB_STATE_UNARMED, schema.COB_STATE_CHARGING,
                  schema.COB_STATE_ARMED, schema.COB_STATE_FIRING)
    for state in (U, C, A, F):
        assert cov["cob_%d" % state], "cob state %d never seen" % state
    for a, b in ((U, C), (C, A), (A, F), (F, U)):
        assert cov["step_%d_%d" % (a, b)], "cob step %d -> %d never seen tick by tick" % (a, b)
    assert cov["charge_rising"] and cov["charge_end"]
    for key in ("EATING", "IN_WATER", "FLYING", "DYING", "sleeping", "smashed",
                "acc1", "acc2", "acc1_stale", "usable_0", "usable_1"):
        assert cov[key], "%s never seen" % key


def test_empty_and_imitater_cards():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    assert w.select_plants([_P.pea_shooter, _P.imitater, _P.sunflower], _P.cherry_bomb) is True
    assert len(pvzemu.CD_TABLE) == N_CD_TABLE
    # The binding's table is the one plant() charges (plant_factory.cpp:543).
    assert w.plant(_P.sunflower, 0, 4) is True
    assert w.scene.cards[2].cold_down == pvzemu.CD_TABLE[int(_P.sunflower)]

    buf, _ = _check(w, "small world")
    cards = buf["cards"][0]
    assert cards["type"][1] == int(_P.imitater)
    assert cards["imitater_type"][1] == int(_P.cherry_bomb)
    assert cards["cd_total"][1] == pvzemu.CD_TABLE[int(_P.cherry_bomb)]
    assert cards["usable"][2] == 0 and cards["cd_remaining"][2] > 0
    for e in cards[3:]:
        assert e["type"] == schema.CARD_TYPE_EMPTY == -1
        assert e["imitater_type"] == -1
        assert e["cd_remaining"] == 0 and e["cd_total"] == 0 and e["usable"] == 0


def test_game_over_phase():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    for _ in range(20_000):
        if w.scene.is_game_over:
            break
        w.update()
    assert w.scene.is_game_over
    buf, _ = _check(w, "game over")
    assert buf["hdr"][0]["phase"] == schema.PHASE_GAME_OVER


def _flag(w, buf):
    w.fill_state(buf)
    return int(buf["hdr"][0]["flag"])


def test_flag_is_zero_based():
    buf = _buf()
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    assert w.scene.spawn.total_flags == START_FLAGS
    assert _flag(w, buf) == 0
    w.scene.spawn.total_flags = START_FLAGS + 6
    assert _flag(w, buf) == 6
    _check(w, "flag 6")
    w.scene.spawn.total_flags = START_FLAGS - 2
    assert _flag(w, buf) == -2  # signed, not wrapped
    w.reset(SEED)
    assert w.scene.spawn.total_flags == START_FLAGS
    assert _flag(w, buf) == 0

    # The wave-10 step: flag 0 through wave 9, 1 from wave 10 on. Clearing the zombies before
    # every update keeps the run alive and the waves quick (a test choice).
    seen = set()
    for _ in range(WAVE_10_CAP):
        if w.scene.spawn.wave >= 10:
            break
        for z in list(w.scene.zombies):
            w.zombie_factory.destroy(z)
        w.update()
        seen.add((w.scene.spawn.wave, _flag(w, buf)))
    assert w.scene.spawn.wave == 10 and not w.scene.is_game_over
    assert seen == {(wave, int(wave >= 10)) for wave in range(11)}
    _check(w, "wave 10")

    # The round end: flag 2 = 2 x one round completed, wave back to 0.
    w.scene.spawn.countdown.endgame = 1
    assert w.update() is True
    assert w.scene.spawn.wave == 0
    assert _flag(w, buf) == 2
    _check(w, "round end")


def test_every_live_byte_written():
    w = _midgame_world()
    a, b = _filled(w, 0x00), _filled(w, 0xCD)
    h = a["hdr"][0]
    assert 0 < h["n_plants"] < CAP["plants"]
    assert 0 < h["n_zombies"] < CAP["zombies"]
    assert 0 < h["n_griditems"] < CAP["griditems"]
    assert schema.summary(a) == schema.summary(b)
    ra, rb = a.tobytes(), b.tobytes()
    assert ra[:HEAD_BYTES] == rb[:HEAD_BYTES]
    for m in LIVE_ARRAYS:
        n = int(h["n_" + m])
        assert a[m][0][:n].tobytes() == b[m][0][:n].tobytes(), m


def test_tail_never_touched():
    w = _midgame_world()
    b = _filled(w, 0xCD)
    h = b["hdr"][0]
    for m in LIVE_ARRAYS:
        tail = b[m][0][int(h["n_" + m]):].tobytes()
        assert tail and set(tail) == {0xCD}, m
    assert set(b["mowers"][0].tobytes()) == {0xCD}


def test_overflow_counts_and_ids():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    for i in range(300):
        w.plant_factory.create(_P.pea_shooter, i % POOL_ROWS, i % 9)
    for _ in range(600):
        w.zombie_factory.create(_Z.zombie, None)
    for i in range(128):
        w.griditem_factory.create(_G.crater, i % POOL_ROWS, i % 9)
    buf, _ = _check(w, "overflow")
    h = buf["hdr"][0]
    assert (h["n_plants"], h["plants_overflow"]) == (256, 44)
    assert (h["n_zombies"], h["zombies_overflow"]) == (512, 88)
    assert (h["n_griditems"], h["griditems_overflow"]) == (128, 0)
    for m, n in (("plants", 256), ("zombies", 512)):
        lst = getattr(w.scene, m)
        ids = [lst.get_index(x) for x in lst][:n]
        assert ids == list(range(n))
        assert list(buf[m][0][:n]["id"]) == ids


def test_destroyed_not_shrunk_is_absent():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    for col in range(6):
        w.plant_factory.create(_P.pea_shooter, 0, col)
    for _ in range(4):
        w.zombie_factory.create(_Z.zombie, None)
    for col in range(3):
        w.griditem_factory.create(_G.crater, 5, col)
    w.plant_factory.destroy([p for p in w.scene.plants if p.col == 2][0])
    w.zombie_factory.destroy(list(w.scene.zombies)[1])
    w.griditem_factory.destroy(list(w.scene.griditems)[0])
    # size() still counts them until the next update shrinks the lists (obj_list.h:158-188).
    assert len(w.scene.plants) == 6 and len(w.scene.zombies) == 4
    buf, _ = _check(w, "destroyed")
    h = buf["hdr"][0]
    assert h["n_plants"] == 5 and h["n_zombies"] == 3 and h["n_griditems"] == 2
    assert list(buf["plants"][0][:5]["id"]) == [0, 1, 3, 4, 5]
    assert list(buf["zombies"][0][:3]["id"]) == [0, 2, 3]
    assert list(buf["plants"][0][:5]["col"]) == [0, 1, 3, 4, 5]
    assert list(buf["plants"][0][:5]["id"]) == [w.scene.plants.get_index(p)
                                                for p in w.scene.plants]


def _script_run(worlds, start, stop):
    for tick in range(start, stop):
        for w in worlds:
            _scenario_step(w, tick)


def test_fill_has_no_side_effects():
    w = _midgame_world()
    before = w.to_json()
    a = _filled(w)
    assert w.to_json() == before
    assert _filled(w).tobytes() == a.tobytes()

    c = w.clone()
    assert schema.summary(_filled(c)) == schema.summary(_filled(w))
    for start in range(MIDGAME_TICKS, MIDGAME_TICKS + 3000, 1000):
        _script_run([w, c], start, start + 1000)
        assert schema.summary(_filled(c)) == schema.summary(_filled(w)), start + 1000

    x, y = _scenario_world(), _scenario_world()
    for start in range(0, 3000, 100):
        _script_run([x, y], start, start + 100)
        sx, sy = schema.summary(_filled(x)), schema.summary(_filled(y))
        assert sx == sy, start + 100
    assert sx["n_zombies"] > 0


def test_batch_and_zero_copy():
    worlds = [_midgame_world(t) for t in (0, 300, 700, 1200)]
    arr = _buf(len(worlds))
    pvzemu.World.fill_states(worlds, arr)
    one = _buf(len(worlds))
    for i, w in enumerate(worlds):
        w.fill_state(one[i:i + 1])
        assert arr[i].tobytes() == one[i].tobytes(), i
        _check(w, "batch row %d" % i, buf=arr[i:i + 1])

    arr = _buf(6)
    worlds[2].fill_state(arr[3:4])
    for i in range(6):
        if i == 3:
            assert arr[i]["hdr"]["magic"] == schema.MAGIC
            assert arr[i].tobytes() == one[2].tobytes()
        else:
            assert set(arr[i].tobytes()) == {0}, i

    zero_d = np.zeros((), dtype=D)
    worlds[1].fill_state(zero_d)
    assert zero_d.tobytes() == one[1].tobytes()
    pvzemu.World.fill_states([], _buf(0))


def test_validation_errors():
    w = _scenario_world()
    good = _buf(2)

    with pytest.raises(ValueError, match=str(schema.STATE_SIZE)):
        w.fill_state(np.zeros(schema.STATE_SIZE, np.uint8))
    with pytest.raises(ValueError, match=str(schema.STATE_SIZE)):
        pvzemu.World.fill_states([w], np.zeros(schema.STATE_SIZE, np.uint8))

    strided = _buf(4)[::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        pvzemu.World.fill_states([w, w], strided)

    ro = _buf(2)
    ro.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        w.fill_state(ro[:1])
    with pytest.raises(ValueError, match="writeable"):
        pvzemu.World.fill_states([w, w], ro)

    with pytest.raises(ValueError):
        w.fill_state(good)
    with pytest.raises(ValueError):
        w.fill_state(_buf(1).reshape(1, 1))
    with pytest.raises(ValueError):
        pvzemu.World.fill_states([w], good)
    with pytest.raises(ValueError):
        pvzemu.World.fill_states([w, w, w], good)
    with pytest.raises(ValueError):
        pvzemu.World.fill_states([w, w], good.reshape(2, 1))
    with pytest.raises(ValueError, match="None"):
        pvzemu.World.fill_states([w, None], good)

    raw = np.zeros(schema.STATE_SIZE + 8, np.uint8)
    mis = raw[1:1 + schema.STATE_SIZE].view(D)
    assert mis.ctypes.data % 4 == 1 and mis.shape == (1,) and mis.flags.c_contiguous
    with pytest.raises(ValueError, match="align"):
        w.fill_state(mis)
    with pytest.raises(ValueError, match="align"):
        pvzemu.World.fill_states([w], mis)

    with pytest.raises(TypeError):
        w.fill_state([0] * schema.STATE_SIZE)
    with pytest.raises(TypeError):
        w.fill_state(good[0])  # np.void, not an array: a copy would lose the writes
    assert set(good.tobytes()) == {0}


def test_timing():
    w = _midgame_world()
    buf = _buf()
    h = _filled(w)["hdr"][0]
    start = time.perf_counter()
    for _ in range(TIMING_FILLS):
        w.fill_state(buf)
    per_fill = (time.perf_counter() - start) / TIMING_FILLS * 1e6

    worlds = [w.clone() for _ in range(BATCH)]
    arr = _buf(BATCH)
    start = time.perf_counter()
    for _ in range(20):
        pvzemu.World.fill_states(worlds, arr)
    per_batch = (time.perf_counter() - start) / 20 * 1e3
    print("fill_state: %.2f us (plants %d, zombies %d, grid items %d); fill_states x%d: %.3f ms"
          % (per_fill, h["n_plants"], h["n_zombies"], h["n_griditems"], BATCH, per_batch))
    assert per_fill < FILL_US_BOUND
    assert per_batch < BATCH_MS_BOUND
