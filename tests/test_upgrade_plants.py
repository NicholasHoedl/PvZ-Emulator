"""Step 1.10: upgrade plants are plantable by card on their base plant.

Before this step (line numbers in this paragraph are pre-fix, fork commit 99255da) can_plant
rejected every occupied cell (plant_factory.cpp:249-252) before it reached the upgrade rule
(plant_factory.cpp:259-266 -> can_plant_advanced_plant, :9-67), and the Gatling Pea rule
compared with gatling_pea instead of the repeater (:21-22). Now the upgrades are checked
(:249-255) before the occupied-cell rule (:257-260); the cattail stays in the third switch
(:267-268).

Real-game rule (wiki, retrieved 2026-10-05): https://plantsvszombies.fandom.com/wiki/Upgrade_plants
"To activate an upgrade, it must be planted on its weaker variant."; Gatling_Pea_(PvZ): "must be
planted on top of a Repeater"; Cob_Cannon_(PvZ): "He must be planted on two side-by-side
Kernel-pults, which must be placed one in front of the other in the same lane (horizontally)".

Emulator facts used here (cited as emulator <path>:<line>):
- system/plant/plant_factory.cpp:24-34: twin_sunflower on sunflower, gloomshroom on fumeshroom,
  winter_melon on melonpult, gold_magnet on magnetshroom; :36-37: spikerock on spikeweed;
  :39-43: cattail on a lily pad with no content; :45-61: cob_cannon on a kernelpult with col < 8
  and another kernelpult at col + 1 in its row, both cells with a base or neither.
- plant_factory.cpp:13-18: every upgrade but the cattail needs a content plant.
- plant_factory.cpp:118-132: get_cost; :128: an upgrade costs 50 more per live copy;
  :157: can_plant refuses when get_cost > sun.
- plant_factory.cpp:180-182 and :231: in a pool water row a plant needs a lily pad.
- plant_factory.cpp:491-505: plant() destroys the content plant under the six single-cell
  upgrades and keeps a fume-shroom's wake state for a gloom-shroom (:535-541); :507-511: the
  lily pad under a cattail; :513-525: both cells' content under a cob; :531 spends get_cost;
  :543 sets the card's cooldown to CD_TABLE[type] (bound as pvzemu.CD_TABLE, pybind.cpp:161).
- plant_factory.cpp:436-447: create() puts a pumpkin in .pumpkin, a lily pad or flower pot in
  .base, a cob in .content of [row][col] and [row][col + 1], anything else in .content.
- world.cpp:71-74: update(action) is apply_action then update(); world.cpp:44-48: update()
  takes 1 off each card's cooldown; sun.cpp:23-24: natural sun lands only when its countdown
  reaches 0, so with a countdown > 1 one update adds no natural sun.
- system/plant/plant_base.cpp:96-105: a new sunflower or twin sunflower waits at least 300
  ticks to make sun, so none is made in the tick that plants it.
- system/plant/plant_base.cpp:219-223: a mushroom created in a pool scene goes to sleep.
- world.cpp:178-189: a card's mask is 1 when it is in hand, off cooldown and can_plant agrees.
- object/scene.h:171: pool water rows are 2 and 3.
"""
import pytest

import pvzemu

_P = pvzemu.PlantType

# Upgrade -> base plant: plant_factory.cpp:24-34 and :45-46; the Gatling Pea's repeater from the
# wiki sentence quoted above (before this step the emulator's line :22 compared with gatling_pea).
UPGRADES = {
    _P.gatling_pea: _P.repeater,
    _P.twin_sunflower: _P.sunflower,
    _P.gloomshroom: _P.fumeshroom,
    _P.winter_melon: _P.melonpult,
    _P.gold_magnet: _P.magnetshroom,
    _P.cob_cannon: _P.kernelpult,
}
COB = _P.cob_cannon

# Test choices, not game facts: the loadout, seed, sun and cells.
CARDS = list(UPGRADES) + [_P.spikerock, _P.cattail, _P.pea_shooter, _P.sunflower]
SEED = 7
SUN = 5000
LAND = (1, 4)
LAND2 = (4, 4)
WATER = (2, 4)  # emulator object/scene.h:171
EMPTY = (5, 8)

LAYOUTS = ("land", "pumpkin", "water")
CASES = [(t, lay) for t in UPGRADES for lay in LAYOUTS if not (t == COB and lay == "pumpkin")]


def _id(case):
    return "%s-%s" % (case[0].name, case[1])


def _world():
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    assert w.select_plants(CARDS, _P.none) is True
    w.scene.stop_spawn = True
    w.scene.sun.sun = SUN
    return w


def _cells(t, r, c):
    return [(r, c), (r, c + 1)] if t == COB else [(r, c)]


def _place(w, upgrade, r, c, lily=False, pumpkin=False):
    """The base plant of `upgrade` at (r, c) (and c + 1 for a cob), on a lily pad and under a
    pumpkin if asked; build order as world.cpp:512-515 says."""
    lst = []
    for rr, cc in _cells(upgrade, r, c):
        if lily:
            lst.append((_P.lily_pad, rr, cc))
        lst.append((UPGRADES[upgrade], rr, cc))
        if pumpkin:
            lst.append((_P.pumpkin, rr, cc))
    assert w.build(lst) is True


def _mask(w, actions):
    m = pvzemu.IntVector()
    w.get_available_actions(actions, m)
    return list(m)[:-1]


def _live(w, t):
    return sum(1 for p in w.scene.plants if p.type == t)


def _card(w, t):
    return next(card for card in w.scene.cards if card.type == t)


def _can(w, t, r, c):
    return w.plant_factory.can_plant(r, c, t, _P.none)


# ---------------------------------------------------------------- a, d: plant on the base

@pytest.mark.parametrize("case", CASES, ids=_id)
def test_upgrade_planted_on_base(case):
    t, layout = case
    w = _world()
    r, c = WATER if layout == "water" else LAND
    _place(w, t, r, c, lily=layout == "water", pumpkin=layout == "pumpkin")
    action = (int(t), r, c)

    assert _can(w, t, r, c) is True
    assert _mask(w, [action, (int(t), *EMPTY)]) == [1, 0]

    cost = w.plant_factory.get_cost(t)
    sun = w.scene.sun.sun
    assert w.scene.sun.natural_sun_countdown > 1
    w.update(action)

    pm = w.scene.plant_map
    for rr, cc in _cells(t, r, c):
        assert pm[rr][cc].content is not None and pm[rr][cc].content.type == t
        assert pm[rr][cc].content.col == c
        if layout == "pumpkin":
            assert pm[rr][cc].pumpkin is not None and pm[rr][cc].pumpkin.type == _P.pumpkin
        if layout == "water":
            assert pm[rr][cc].base is not None and pm[rr][cc].base.type == _P.lily_pad
    assert _live(w, t) == 1
    assert _live(w, UPGRADES[t]) == 0
    assert w.scene.sun.sun == sun - cost
    assert _card(w, t).cold_down == pvzemu.CD_TABLE[int(t)] - 1


# ---------------------------------------------------------------- b: only on its own base

def test_upgrade_rejected_off_its_base():
    w = _world()
    placed = {
        (0, 0): _P.repeater,
        (0, 2): _P.sunflower,
        (0, 4): _P.fumeshroom,
        (0, 6): _P.melonpult,
        (0, 8): _P.magnetshroom,
        (1, 0): _P.kernelpult,  # alone: no kernelpult beside it
        (1, 2): _P.pea_shooter,
        (4, 0): _P.gatling_pea,
        (4, 2): _P.twin_sunflower,
        (4, 4): _P.gloomshroom,
        (4, 6): _P.winter_melon,
        (4, 8): _P.gold_magnet,
        (5, 0): COB,  # also covers (5, 1)
    }
    assert w.build([(p, r, c) for (r, c), p in placed.items()]) is True
    placed[(5, 1)] = COB

    for t, base in UPGRADES.items():
        assert _can(w, t, *EMPTY) is False, t
        for (r, c), p in placed.items():
            if p == base and t != COB:
                continue
            assert _can(w, t, r, c) is False, (t, p)
    # The old rule (pre-fix plant_factory.cpp:22): a Gatling Pea card on a Gatling Pea.
    assert _can(w, _P.gatling_pea, 4, 0) is False


# ---------------------------------------------------------------- c: the cob's two cells

def test_cob_cannon_cells():
    w = _world()
    lst = [
        (_P.kernelpult, 1, 3), (_P.kernelpult, 1, 4),  # land pair
        (_P.kernelpult, 4, 3),  # single
        (_P.kernelpult, 0, 7), (_P.kernelpult, 0, 8),  # pair ending in column 8
        (_P.kernelpult, 4, 6), (_P.kernelpult, 5, 6),  # vertical pair
        (_P.flower_pot, 5, 0), (_P.kernelpult, 5, 0),  # land pair, pots under both
        (_P.flower_pot, 5, 1), (_P.kernelpult, 5, 1),
        (_P.flower_pot, 0, 0), (_P.kernelpult, 0, 0),  # pot under the left one only
        (_P.kernelpult, 0, 1),
        (_P.kernelpult, 0, 3),  # pot under the right one only
        (_P.flower_pot, 0, 4), (_P.kernelpult, 0, 4),
        (_P.lily_pad, 2, 0), (_P.kernelpult, 2, 0),  # water pair, lily pads under both
        (_P.lily_pad, 2, 1), (_P.kernelpult, 2, 1),
        (_P.lily_pad, 3, 0), (_P.kernelpult, 3, 0),  # water, lily pad under the left only
        (_P.kernelpult, 3, 1),
        (_P.kernelpult, 3, 4),  # water, lily pad under the right only
        (_P.lily_pad, 3, 5), (_P.kernelpult, 3, 5),
    ]
    assert w.build(lst) is True

    expected = {
        (1, 3): True, (1, 4): False,
        (4, 3): False,
        (0, 7): True, (0, 8): False,
        (4, 6): False, (5, 6): False,
        (5, 0): True, (0, 0): False, (0, 3): False,
        (2, 0): True, (3, 0): False, (3, 4): False,
    }
    for (r, c), legal in expected.items():
        assert _can(w, COB, r, c) is legal, (r, c)
    cands = [(int(COB), r, c) for r, c in expected]
    assert _mask(w, cands) == [int(v) for v in expected.values()]

    kernels = _live(w, _P.kernelpult)
    w.update((int(COB), 1, 3))
    pm = w.scene.plant_map
    for c in (3, 4):
        assert pm[1][c].content is not None and pm[1][c].content.type == COB
        assert pm[1][c].content.col == 3
    assert _live(w, COB) == 1
    assert _live(w, _P.kernelpult) == kernels - 2


# ---------------------------------------------------------------- e: gloom-shroom sleep

@pytest.mark.parametrize("awake", [False, True])
def test_gloomshroom_keeps_fume_sleep(awake):
    w = _world()
    r, c = LAND
    fume = w.plant_factory.create(_P.fumeshroom, r, c)
    assert fume.is_sleeping is True
    if awake:
        fume.set_sleep(False)
        assert fume.is_sleeping is False
    w.update((int(_P.gloomshroom), r, c))
    gloom = w.scene.plant_map[r][c].content
    assert gloom is not None and gloom.type == _P.gloomshroom
    assert gloom.is_sleeping is (not awake)


# ---------------------------------------------------------------- f: sun

@pytest.mark.parametrize("t", list(UPGRADES), ids=lambda t: t.name)
def test_upgrade_sun(t):
    w = _world()
    r, c = LAND
    _place(w, t, r, c)
    cost = w.plant_factory.get_cost(t)
    action = (int(t), r, c)

    w.scene.sun.sun = cost - 1
    assert _can(w, t, r, c) is False
    assert _mask(w, [action]) == [0]
    w.scene.sun.sun = cost
    assert _can(w, t, r, c) is True
    assert _mask(w, [action]) == [1]

    w.scene.sun.sun = SUN
    w.update(action)
    assert _live(w, t) == 1
    assert w.plant_factory.get_cost(t) == cost + 50  # plant_factory.cpp:128

    r2, c2 = LAND2
    _place(w, t, r2, c2)
    w.scene.sun.sun = cost + 49
    assert _can(w, t, r2, c2) is False
    w.scene.sun.sun = cost + 50
    assert _can(w, t, r2, c2) is True


# ---------------------------------------------------------------- g: spikerock and cattail

def test_spikerock_as_before():
    w = _world()
    r, c = LAND
    assert w.build([(_P.spikeweed, r, c)]) is True
    assert _can(w, _P.spikerock, r, c) is True
    assert _can(w, _P.spikerock, *EMPTY) is False
    assert _mask(w, [(int(_P.spikerock), r, c), (int(_P.spikerock), *EMPTY)]) == [1, 0]

    cost = w.plant_factory.get_cost(_P.spikerock)
    sun = w.scene.sun.sun
    assert w.scene.sun.natural_sun_countdown > 1
    w.update((int(_P.spikerock), r, c))
    assert w.scene.plant_map[r][c].content.type == _P.spikerock
    assert _live(w, _P.spikeweed) == 0
    assert w.scene.sun.sun == sun - cost


def test_cattail_as_before():
    w = _world()
    assert w.build([
        (_P.lily_pad, 2, 4),
        (_P.lily_pad, 2, 6), (_P.pea_shooter, 2, 6),
    ]) is True
    assert _can(w, _P.cattail, 2, 4) is True
    assert _can(w, _P.cattail, 2, 6) is False  # the lily pad holds a plant
    assert _can(w, _P.cattail, 3, 4) is False  # water, no lily pad
    assert _can(w, _P.cattail, *EMPTY) is False  # land

    cost = w.plant_factory.get_cost(_P.cattail)
    sun = w.scene.sun.sun
    assert w.scene.sun.natural_sun_countdown > 1
    w.update((int(_P.cattail), 2, 4))
    cell = w.scene.plant_map[2][4]
    assert cell.content is not None and cell.content.type == _P.cattail
    assert cell.base is None  # plant_factory.cpp:507-511 destroys the lily pad
    assert w.scene.sun.sun == sun - cost


# ---------------------------------------------------------------- h: nothing else moved

def test_other_cards_on_occupied_cells():
    w = _world()
    assert w.build([
        (_P.sunflower, 0, 0),
        (_P.gatling_pea, 0, 2),
        (_P.pumpkin, 0, 4),
    ]) is True
    assert _can(w, _P.pea_shooter, 0, 0) is False
    assert _can(w, _P.sunflower, 0, 2) is False
    # A pumpkin alone is not a content plant (plant_factory.cpp:13-18).
    for t in UPGRADES:
        assert _can(w, t, 0, 4) is False, t
    assert _mask(w, [(int(_P.pea_shooter), 0, 0), (int(_P.sunflower), 0, 2)]) == [0, 0]
