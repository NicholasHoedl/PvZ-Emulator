"""Step 1.2: the Cob Cannon fire action (op = -3, row, x), its mask rule and PlantFactory.create.

Emulator facts used here (cited as emulator <path>:<line>):
- system/plant/cob_cannon.cpp:43-48: a new cannon starts in cob_cannon_unarmed_idle with
  countdown.status = 500; cob_cannon.cpp:9-21: it goes unarmed -> charge when the countdown
  hits 0, and charge -> armed_idle when the charge animation has played once (animation-driven,
  so the tests tick with a cap instead of assuming a tick count).
- system/plant/cob_cannon.cpp:28-41: launch() works only when armed, sets cob_cannon_launch and
  countdown.launch = 206, and stores cannon.x = x - 47, cannon.y = y.
- system/plant/plant_system.cpp:333: countdown.launch is decremented once per update;
  plant_system.cpp:365 + 408-409: the projectile is launched when it reaches 1;
  plant_system.cpp:302-311: a cob projectile gets cannon_row = get_row_by_x_and_y(cannon.x, cannon.y).
- system/plant/plant_system.cpp:434-439: after the launch animation the cannon goes back to
  unarmed_idle with countdown.status = 3000, then re-arms as above.
- system/plant/plant_factory.cpp:440-442: a cannon occupies plant_map[row][col] and [col + 1].
- object/obj_list.h:34-38: scene.plants iterates in slot order; slots are handed out in creation
  order on a fresh list (obj_list.h:106-110).
- object/scene.h:170: pool has 6 rows.
- world.cpp:115-117: the mask has len(actions) + 1 slots, the last (no-op) always 1.
- system/plant/plant_base.cpp:70: countdown.dead starts at 200; plant_system.cpp:450-453: a smashed
  plant is destroyed only once that counts below 1, and plant_system.cpp:461-466 skips its status
  updates, so a cannon marked smashed stays alive and armed_idle for the ticks below.
"""
import pvzemu

COB_FIRE = -3  # step 1.2 action op
POOL_ROWS = 6  # emulator object/scene.h:170
CLICK_X = 400  # test choice: a click pixel inside the lawn
ARM_CAP = 5000  # test choice: generous cap for the first arming (500 countdown + charge anim)
REARM_CAP = 10000  # test choice: re-arm waits 3000 (plant_system.cpp:436) + launch and charge anims

PS = pvzemu.PlantStatus


def _pool_world():
    w = pvzemu.World(pvzemu.SceneType.pool)
    assert w.scene.rows == POOL_ROWS
    # No zombies: keeps the cannons alive while the tests tick thousands of updates.
    w.scene.stop_spawn = True
    return w


def _mask(w, actions):
    m = pvzemu.IntVector()
    w.get_available_actions(actions, m)
    assert len(m) == len(actions) + 1
    return list(m)


def _fire_actions():
    return [(COB_FIRE, r, CLICK_X) for r in range(POOL_ROWS)]


def _cobs(w):
    return [p for p in w.scene.plants if p.type == pvzemu.PlantType.cob_cannon]


def _tick_until(w, cond, cap):
    for n in range(1, cap + 1):
        w.update()
        if cond():
            return n
    return None


def _all_armed(w):
    return lambda: all(p.status == PS.cob_cannon_armed_idle for p in _cobs(w))


def _two_cob_world():
    w = _pool_world()
    a = w.plant_factory.create(pvzemu.PlantType.cob_cannon, 2, 1)
    b = w.plant_factory.create(pvzemu.PlantType.cob_cannon, 2, 5)
    return w, a, b


def _armed_two_cob_world():
    w, a, b = _two_cob_world()
    n = _tick_until(w, _all_armed(w), ARM_CAP)
    assert n is not None, "cannons did not arm within the cap"
    return w, a, b, n


def _status_by_col(w):
    return {p.col: p.status for p in _cobs(w)}


def _cob_projectiles(w):
    return [q for q in w.scene.projectiles if q.type == pvzemu.ProjectileType.cob_cannon]


def test_create_returns_live_reference_and_mask_zero_before_armed():
    w, a, b = _two_cob_world()
    assert len(_cobs(w)) == 2
    assert a.status == PS.cob_cannon_unarmed_idle
    m = _mask(w, _fire_actions())
    assert m[:-1] == [0] * POOL_ROWS
    assert m[-1] == 1
    # reference_internal: the returned Plant is the live object, so it sees later updates.
    w.update()
    assert a.countdown.status == 499  # cob_cannon.cpp:47 sets 500; plant_system.cpp:470-471 -1


def test_cannons_arm_within_cap():
    w, a, b, n = _armed_two_cob_world()
    print("ticks to armed:", n)
    assert a.status == PS.cob_cannon_armed_idle
    assert b.status == PS.cob_cannon_armed_idle


def test_mask_when_armed():
    w, _, _, _ = _armed_two_cob_world()
    actions = _fire_actions() + [(COB_FIRE, POOL_ROWS, CLICK_X), (COB_FIRE, -1, CLICK_X)]
    m = _mask(w, actions)
    assert m[:POOL_ROWS] == [1] * POOL_ROWS
    assert m[POOL_ROWS] == 0
    assert m[POOL_ROWS + 1] == 0
    assert m[-1] == 1


def test_fire_launches_one_cannon_and_spawns_cob():
    w, a, b, _ = _armed_two_cob_world()
    w.update((COB_FIRE, 2, CLICK_X))
    statuses = [a.status, b.status]
    assert statuses.count(PS.cob_cannon_launch) == 1
    assert statuses.count(PS.cob_cannon_armed_idle) == 1
    fired = a if a.status == PS.cob_cannon_launch else b
    # launch() sets 206 (cob_cannon.cpp:34); the update() in the same call decrements it once
    # (plant_system.cpp:333).
    assert fired.countdown.launch == 205
    assert fired.cannon.x == CLICK_X - 47  # cob_cannon.cpp:37
    n = _tick_until(w, lambda: len(_cob_projectiles(w)) > 0, 400)
    assert n is not None, "no cob projectile appeared"
    projs = _cob_projectiles(w)
    assert len(projs) == 1
    assert projs[0].cannon_row == 2


def test_second_fire_uses_other_cannon_then_mask_zero():
    w, a, b, _ = _armed_two_cob_world()
    w.update((COB_FIRE, 2, CLICK_X))
    w.update((COB_FIRE, 4, CLICK_X))
    assert a.status == PS.cob_cannon_launch
    assert b.status == PS.cob_cannon_launch
    m = _mask(w, _fire_actions())
    assert m[:-1] == [0] * POOL_ROWS
    assert m[-1] == 1


def test_rotation_wraps_to_lowest_slot():
    w, a, b, _ = _armed_two_cob_world()
    # a was created first, so it holds the lower slot (obj_list.h:106-110).
    w.update((COB_FIRE, 1, CLICK_X))
    assert a.status == PS.cob_cannon_launch and b.status == PS.cob_cannon_armed_idle
    w.update((COB_FIRE, 1, CLICK_X))
    assert b.status == PS.cob_cannon_launch
    n = _tick_until(w, _all_armed(w), REARM_CAP)
    assert n is not None, "cannons did not re-arm within the cap"
    print("ticks to re-arm:", n)
    w.update((COB_FIRE, 1, CLICK_X))
    assert a.status == PS.cob_cannon_launch
    assert b.status == PS.cob_cannon_armed_idle


def test_fire_with_no_armed_cannon_is_noop():
    w = _pool_world()
    assert isinstance(w.update((COB_FIRE, 2, CLICK_X)), bool)
    w2, a, b = _two_cob_world()
    assert isinstance(w2.update((COB_FIRE, 2, CLICK_X)), bool)
    assert a.status != PS.cob_cannon_launch and b.status != PS.cob_cannon_launch


def test_fire_out_of_range_row_is_noop():
    w, a, b, _ = _armed_two_cob_world()
    w.update((COB_FIRE, POOL_ROWS, CLICK_X))
    w.update((COB_FIRE, -1, CLICK_X))
    assert a.status == PS.cob_cannon_armed_idle
    assert b.status == PS.cob_cannon_armed_idle


def test_update_all_fires_only_in_its_world():
    w1, a1, b1, _ = _armed_two_cob_world()
    w2, a2, b2, _ = _armed_two_cob_world()
    all_actions = _fire_actions()
    # (-1, 0, 0) shovels an empty cell: a harmless action for the other world.
    actions = [(COB_FIRE, 2, CLICK_X), (-1, 0, 0)]
    masks = pvzemu.BatchActionMasks()
    check_result = pvzemu.IntVector()
    done = pvzemu.IntVector()
    pvzemu.World.update_all([w1, w2], all_actions, actions, masks, [], check_result, done, 1)
    assert [a1.status, b1.status].count(PS.cob_cannon_launch) == 1
    assert a2.status == PS.cob_cannon_armed_idle
    assert b2.status == PS.cob_cannon_armed_idle
    assert list(masks[0])[:-1] == [1] * POOL_ROWS  # one cannon still armed in w1
    assert list(masks[1])[:-1] == [1] * POOL_ROWS


def test_smashed_cannon_is_not_fired():
    w = _pool_world()
    c = w.plant_factory.create(pvzemu.PlantType.cob_cannon, 2, 1)
    n = _tick_until(w, _all_armed(w), ARM_CAP)
    assert n is not None, "cannon did not arm within the cap"
    c.is_smashed = True
    m = _mask(w, _fire_actions())
    assert m[:-1] == [0] * POOL_ROWS
    assert m[-1] == 1
    w.update((COB_FIRE, 2, CLICK_X))
    assert not c.is_dead
    assert c.status == PS.cob_cannon_armed_idle
    assert _cob_projectiles(w) == []
