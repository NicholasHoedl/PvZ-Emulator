#include "learning/pvz_state_adapter.h"

namespace pvz_emulator::learning {

using namespace pvz_emulator::object;

static_assert(PVZ_STATE_VERSION == 1, "adapter written for schema v1");
static_assert(std::tuple_size<decltype(scene::cards)>::value == PVZ_MAX_CARDS,
    "one schema card per emulator card slot");

const uint32_t EMULATOR_VALID =
    PVZ_VALID_CORE | PVZ_VALID_PHASE | PVZ_VALID_GAME_OVER | PVZ_VALID_WAVE |
    PVZ_VALID_FLAG | PVZ_VALID_ROUND_TYPES | PVZ_VALID_CARDS |
    PVZ_VALID_PLANT_BASIC | PVZ_VALID_PLANT_POS | PVZ_VALID_PLANT_HP |
    PVZ_VALID_PLANT_STATUS | PVZ_VALID_PLANT_COUNTDOWNS | PVZ_VALID_COB_STATE |
    PVZ_VALID_COB_CHARGE | PVZ_VALID_Z_BASIC | PVZ_VALID_Z_HP | PVZ_VALID_Z_MAX_HP |
    PVZ_VALID_ACC_HP | PVZ_VALID_ACC_TYPE | PVZ_VALID_Z_STATUS | PVZ_VALID_Z_EATING |
    PVZ_VALID_Z_IN_WATER | PVZ_VALID_Z_FLYING | PVZ_VALID_Z_HYPNO | PVZ_VALID_Z_DEBUFF |
    PVZ_VALID_Z_SPAWN_WAVE | PVZ_VALID_GRID | PVZ_VALID_GRID_COUNTDOWN;

// Writes the first cap live objects of list (the iterator skips freed and freeable slots, in
// ascending slot order: object/obj_list.h:26-39) and counts all of them; size() would also count
// objects freed since the last shrink_to_fit (obj_list.h:158-188).
template <typename T, size_t S, typename E, typename Write>
static void fill_array(
    const obj_list<T, S>& list,
    E* out,
    int32_t cap,
    int32_t& n,
    int32_t& overflow,
    Write write)
{
    int32_t live = 0;
    for (const auto& obj : list) {
        if (live < cap) {
            write(out[live], obj, list.get_index(obj));
        }
        ++live;
    }

    n = live < cap ? live : cap;
    overflow = live - n;
}

static int32_t cob_state(const plant& p) {
    if (p.type != plant_type::cob_cannon) {
        return PVZ_COB_STATE_NOT_COB;
    }

    // object/plant.h:96-99
    switch (p.status) {
    case plant_status::cob_cannon_unarmed_idle:
        return PVZ_COB_STATE_UNARMED;
    case plant_status::cob_cannon_charge:
        return PVZ_COB_STATE_CHARGING;
    case plant_status::cob_cannon_armed_idle:
        return PVZ_COB_STATE_ARMED;
    case plant_status::cob_cannon_launch:
        return PVZ_COB_STATE_FIRING;
    default:
        return PVZ_COB_STATE_NOT_COB;
    }
}

static void write_plant(pvz_plant_t& e, const plant& p, int id) {
    auto cob = cob_state(p);

    e.id = id;
    e.type = static_cast<int32_t>(p.type);
    e.row = static_cast<int32_t>(p.row);
    e.col = static_cast<int32_t>(p.col);
    e.status_raw = static_cast<int32_t>(p.status);
    e.x = static_cast<float>(p.x);
    e.y = static_cast<float>(p.y);
    e.hp = p.hp;
    e.max_hp = p.max_hp;
    e.state_countdown = p.countdown.status;
    e.shoot_countdown = p.countdown.generate;
    e.effect_countdown = p.countdown.effect;
    e.cob_state = cob;
    e.cob_recharge_countdown = cob == PVZ_COB_STATE_UNARMED ? p.countdown.status : 0;
    // the charge animation's progress: zeroed at the start (object/plant.cpp:252-254), stops at
    // exactly 1 (system/reanim.cpp:175-178)
    e.cob_charge_progress = cob == PVZ_COB_STATE_CHARGING ? p.reanim.progress : 0.0f;
    e.flags = (p.is_sleeping ? PVZ_PLANT_FLAG_SLEEPING : 0u) |
        (p.is_smashed ? PVZ_PLANT_FLAG_SMASHED : 0u);
}

static void write_zombie(pvz_zombie_t& e, const zombie& z, int id) {
    // a shot-off first accessory keeps its max_hp (system/damage.cpp:653-654), so an accessory
    // slot of type none is written as 0 HP and 0 max HP
    bool no_acc1 = z.accessory_1.type == zombie_accessories_type_1::none;
    bool no_acc2 = z.accessory_2.type == zombie_accessories_type_2::none;

    e.id = id;
    e.type = static_cast<int32_t>(z.type);
    e.row = static_cast<int32_t>(z.row);
    e.status_raw = static_cast<int32_t>(z.status);
    e.x = z.x;
    e.y = z.y;
    e.hp = z.hp;
    e.max_hp = static_cast<int32_t>(z.max_hp);
    e.acc1_type = static_cast<int32_t>(z.accessory_1.type);
    e.acc1_hp = no_acc1 ? 0 : static_cast<int32_t>(z.accessory_1.hp);
    e.acc1_max_hp = no_acc1 ? 0 : static_cast<int32_t>(z.accessory_1.max_hp);
    e.acc2_type = static_cast<int32_t>(z.accessory_2.type);
    e.acc2_hp = no_acc2 ? 0 : static_cast<int32_t>(z.accessory_2.hp);
    e.acc2_max_hp = no_acc2 ? 0 : static_cast<int32_t>(z.accessory_2.max_hp);
    e.slow_countdown = static_cast<int32_t>(z.countdown.slow);
    e.freeze_countdown = static_cast<int32_t>(z.countdown.freeze);
    e.butter_countdown = static_cast<int32_t>(z.countdown.butter);
    e.spawn_wave = static_cast<int32_t>(z.spawn_wave);
    e.flags = (z.is_eating ? PVZ_ZOMBIE_FLAG_EATING : 0u) |
        (z.is_in_water ? PVZ_ZOMBIE_FLAG_IN_WATER : 0u) |
        (z.is_flying_or_falling() ? PVZ_ZOMBIE_FLAG_FLYING : 0u) |
        (z.has_death_status() ? PVZ_ZOMBIE_FLAG_DYING : 0u) |
        (z.is_hypno ? PVZ_ZOMBIE_FLAG_HYPNO : 0u);
    e._pad = 0;
}

static void write_griditem(pvz_griditem_t& e, const griditem& g, int) {
    e.type = static_cast<int32_t>(g.type);
    e.row = static_cast<int32_t>(g.row);
    e.col = static_cast<int32_t>(g.col);
    e.countdown_raw = g.countdown;
}

static void write_card(pvz_card_t& e, const scene::card_data& c) {
    bool empty = c.type == plant_type::none;

    // a planted card's cooldown is CD_TABLE[target], target = the imitater's type for an
    // imitater card (system/plant/plant_factory.cpp:464, :541)
    auto target = static_cast<int>(
        c.type == plant_type::imitater ? c.imitater_type : c.type);
    bool in_table = target >= 0 && target < static_cast<int>(plant::CD_TABLE.size());

    e.type = empty ? PVZ_CARD_TYPE_EMPTY : static_cast<int32_t>(c.type);
    e.imitater_type = static_cast<int32_t>(c.imitater_type);
    e.cd_remaining = static_cast<int32_t>(c.cold_down);
    e.cd_total = !empty && in_table ? static_cast<int32_t>(plant::CD_TABLE[target]) : 0;
    e.usable = !empty && c.cold_down == 0 ? 1 : 0;
}

void fill_state(const world& w, pvz_state_t& out) noexcept {
    const auto& s = w.scene;
    auto& h = out.hdr;

    h.magic = PVZ_STATE_MAGIC;
    h.schema_version = PVZ_STATE_VERSION;
    h.layout_hash = PVZ_STATE_LAYOUT_HASH;
    h.struct_size = PVZ_STATE_SIZE;
    h.world = PVZ_WORLD_EMULATOR;
    h.valid = EMULATOR_VALID;
    h.tick = s.tick;
    h.phase = s.is_game_over ? PVZ_PHASE_GAME_OVER : PVZ_PHASE_PLAYING;
    h.paused = 0;
    h.scene = static_cast<int32_t>(s.type);
    h.flag = static_cast<int32_t>(s.spawn.total_flags);
    h.wave = static_cast<int32_t>(s.spawn.wave);
    h.next_wave_countdown = static_cast<int32_t>(s.spawn.countdown.next_wave);
    h.next_wave_is_huge = s.spawn.wave % 10 == 9 ? 1 : 0;
    h.sun = static_cast<int32_t>(s.sun.sun);

    fill_array(s.plants, out.plants, PVZ_MAX_PLANTS,
        h.n_plants, h.plants_overflow, write_plant);
    fill_array(s.zombies, out.zombies, PVZ_MAX_ZOMBIES,
        h.n_zombies, h.zombies_overflow, write_zombie);
    fill_array(s.griditems, out.griditems, PVZ_MAX_GRIDITEMS,
        h.n_griditems, h.griditems_overflow, write_griditem);

    h.n_mowers = 0;
    h.mowers_overflow = 0;

    constexpr size_t n_types = sizeof(h.round_zombie_types);
    for (size_t t = 0; t < n_types; t++) {
        bool on = t < s.spawn.spawn_flags.size() && s.spawn.spawn_flags[t];
        h.round_zombie_types[t] = static_cast<uint8_t>(on ? 1 : 0);
    }

    for (size_t i = 0; i < PVZ_MAX_CARDS; i++) {
        write_card(out.cards[i], s.cards[i]);
    }
}

void fill_states(const std::vector<world*>& worlds, pvz_state_t* out) {
    for (size_t i = 0; i < worlds.size(); i++) {
        fill_state(*worlds[i], out[i]);
    }
}

}
