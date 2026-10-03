#pragma once
#include <cstdint>
#include <vector>

#include "world.h"
#include "schema/pvz_state.h"

namespace pvz_emulator::learning {

// every PVZ_VALID_* group the adapter fills: all but PVZ_VALID_MOWERS (the emulator has none)
extern const uint32_t EMULATOR_VALID;

// Fills out from w (pvz-rl schema/pvz_state.h). Reads w only. Writes the header, all cards and
// entries [0, n) of plants, zombies and grid items; never [n, cap) of any array, nor mowers[].
void fill_state(const world& w, pvz_state_t& out) noexcept;

// out[i] from worlds[i], one after another; out holds worlds.size() states.
void fill_states(const std::vector<world*>& worlds, pvz_state_t* out);

}
