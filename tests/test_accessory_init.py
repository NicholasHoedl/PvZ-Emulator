"""Step 1.6: a zombie created in a reused slot starts with no second accessory and 0 HP for it.

Emulator facts used here (cited as emulator <path>:<line>):
- object/obj_list.h:115-123: alloc() hands back a freed slot as it is, without zeroing it;
  obj_list.h:158-183: shrink_to_fit() frees slots whose object is_freeable(), and
  object/zombie.h:255-256: a zombie is freeable once is_dead; world.cpp:16-21 and 31:
  world::update() runs shrink_to_fit() on the zombie list.
- system/zombie/zombie_factory.cpp:183-184: destroy() sets is_dead.
- system/zombie/zombie_factory.cpp:13-17: create(type, spawn_wave) takes its slot from
  scene.zombies.alloc(); zombie_factory.cpp:22-28: a plain zombie and a screen-door zombie are
  both built by common_zombie::init, newspaper.cpp:16-19 and ladder.cpp:43-46 by their own init,
  all through zombie_base::init.
- system/zombie/common_zombie.cpp:67-70: zombie_base::init resets both accessory types and
  accessory_1.hp (line 70 repeated line 68, so accessory_2.hp was never reset);
  common_zombie.cpp:102-105: set_common_fields copies accessory_2.hp into accessory_2.max_hp.
- common_zombie.cpp:238-240: a screen-door zombie sets accessory_2.hp; newspaper.cpp:28-29 and
  ladder.cpp:49-50 set it for the newspaper and ladder zombies.
- system/spawn.cpp:362-364: the wave HP total adds accessory_2.hp for every zombie.
- object/scene.h:138 and world.cpp:52-54: stop_spawn stops the spawn system, so the test's
  zombies are the only ones on the board.
"""
import pvzemu

_Z = pvzemu.ZombieType
_A2 = pvzemu.ZombieAccessoriesType2
SEED = 7  # test choice


def _only_zombie(w):
    zs = list(w.scene.zombies)
    assert len(zs) == 1
    return zs[0]


def _plain_zombie_after(first_type, first_accessory):
    w = pvzemu.World(pvzemu.SceneType.pool, SEED)
    w.scene.stop_spawn = True
    assert len(w.scene.plants) == 0

    w.zombie_factory.create(first_type, None)
    first = _only_zombie(w)
    assert first.type == first_type
    assert first.accessory_2.type == first_accessory
    assert first.accessory_2.hp > 0

    w.zombie_factory.destroy(first)
    assert first.is_dead
    w.update()
    assert len(w.scene.zombies) == 0

    w.zombie_factory.create(_Z.zombie, None)
    z = _only_zombie(w)
    assert z.type == _Z.zombie
    return z


def _assert_no_accessory_2(z):
    assert z.accessory_2.type == _A2.none
    assert z.accessory_2.hp == 0, "stale accessory_2.hp %d" % z.accessory_2.hp
    assert z.accessory_2.max_hp == 0, "stale accessory_2.max_hp %d" % z.accessory_2.max_hp


def test_plain_zombie_in_screendoor_slot():
    _assert_no_accessory_2(_plain_zombie_after(_Z.screendoor, _A2.screen_door))


def test_plain_zombie_in_newspaper_or_ladder_slot():
    _assert_no_accessory_2(_plain_zombie_after(_Z.newspaper, _A2.newspaper))
    _assert_no_accessory_2(_plain_zombie_after(_Z.ladder, _A2.ladder))
