"""Step 0.9 smoke tests: random legal play never crashes; the first wave spawns on update 600.

Emulator facts used here (cited as emulator <path>:<line>):
- world.cpp:71-104: update((op, row, col)) plants the card whose type id equals op,
  op == -1 shovels the coffee bean / content / base, op == -2 shovels the pumpkin.
- world.cpp:122-124: the mask has len(actions) + 1 slots and the last slot is always 1 (no-op).
- world.h:115-117 / pybind.cpp:17,84: masks is an out parameter of the opaque type IntVector.
- spawn.cpp:462 and spawn.cpp:377,404,412-414: next_wave starts at 600, is decremented once per
  update, and the wave's zombies are created on the update where it reaches 0.
"""
import random
import time

import pvzemu

# Design choice for the smoke test, not a game fact: ten distinct cards that cover land,
# water (lily_pad) and the pumpkin layer, so every op kind (plant, -1, -2) can become legal.
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

SHOVEL = -1  # emulator world.cpp:86
SHOVEL_PUMPKIN = -2  # emulator world.cpp:82
POOL_ROWS = 6  # emulator object/scene.h:175
COLS = 9  # emulator world.cpp:81


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


def _pool_world():
    w = pvzemu.World(pvzemu.SceneType.pool)
    assert w.scene.rows == POOL_ROWS
    return w


def test_random_legal_actions_10000_ticks():
    # Seeds only the action choice. The emulator's own RNG is seeded from
    # std::random_device (emulator object/scene.h:142) and is not reachable from Python.
    rng = random.Random(9)
    w = _pool_world()
    # The binding drops the C++ default for imitater_type (pybind.cpp:140), so pass it.
    assert w.select_plants(CARDS, pvzemu.PlantType.none) is True
    assert [c.type for c in w.scene.cards] == CARDS

    actions = _candidate_actions()
    masks = pvzemu.IntVector()
    ticks = 0
    taken = 0
    taken_by_kind = {"plant": 0, "shovel": 0, "shovel_pumpkin": 0}

    start = time.perf_counter()
    for _ in range(10_000):
        w.get_available_actions(actions, masks)
        assert len(masks) == len(actions) + 1
        assert masks[len(actions)] == 1
        legal = [i for i, m in enumerate(masks) if m]
        choice = rng.choice(legal)
        if choice == len(actions):
            done = w.update()
        else:
            op = actions[choice][0]
            if op == SHOVEL:
                taken_by_kind["shovel"] += 1
            elif op == SHOVEL_PUMPKIN:
                taken_by_kind["shovel_pumpkin"] += 1
            else:
                taken_by_kind["plant"] += 1
            taken += 1
            done = w.update(actions[choice])
        assert isinstance(done, bool)
        ticks += 1
    elapsed = time.perf_counter() - start

    assert ticks == 10_000
    assert isinstance(w.scene.is_game_over, bool)
    print("is_game_over:", w.scene.is_game_over)
    print("ticks:", ticks, "non-no-op actions:", taken, taken_by_kind)
    print("wave:", w.scene.spawn.wave, "zombies:", len(w.scene.zombies),
          "plants:", len(w.scene.plants))
    print("ticks/s: %.0f (random legal actions incl. get_available_actions)" % (ticks / elapsed))
    # Random play must have found at least one legal non-no-op action: sun starts at 9990
    # (emulator object/scene.h:103) and every card starts with cold_down 0 (world.cpp:295-297).
    assert taken > 0


def test_plain_update_10000_ticks():
    # Once is_game_over is set, update() returns at once (emulator world.cpp:23-25), so the
    # rate below mixes real ticks with early returns; game_over_at says where the split is.
    w = _pool_world()
    game_over_at = None
    start = time.perf_counter()
    for tick in range(1, 10_001):
        assert isinstance(w.update(), bool)
        if game_over_at is None and w.scene.is_game_over:
            game_over_at = tick
    elapsed = time.perf_counter() - start
    print("is_game_over:", w.scene.is_game_over, "game_over_at:", game_over_at)
    print("ticks/s: %.0f (plain update())" % (10_000 / elapsed))


def test_first_wave_after_600_updates():
    # README.md:38-41: "The first wave of zombies will spawn after 6 seconds." above 600 updates.
    # spawn.cpp:462 sets next_wave = 600; spawn.update -> next_spawn_countdown_update
    # decrements it once per world.update (spawn.cpp:377); at 0 it creates the wave's 50
    # zombies (spawn.cpp:404-414) and increments wave (spawn.cpp:421). So the zombie list is
    # empty after 599 updates and non-empty after exactly 600.
    w = _pool_world()
    assert w.scene.spawn.wave == 0
    assert w.scene.spawn.countdown.next_wave == 600
    assert len(w.scene.zombies) == 0

    for tick in range(1, 600):
        assert w.update() is False
        assert len(w.scene.zombies) == 0, "zombies present after update %d" % tick
        assert w.scene.spawn.wave == 0
        assert w.scene.spawn.countdown.next_wave == 600 - tick

    assert w.update() is False  # update 600
    assert w.scene.spawn.wave == 1
    assert len(w.scene.zombies) == 50  # spawn.cpp:412-415
