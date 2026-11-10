/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2026 Hamzah Ibrahim
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#ifndef SWIFTGPUPACKSIM_GPU_BUFFER_DUMP_H
#define SWIFTGPUPACKSIM_GPU_BUFFER_DUMP_H

/**
 * @file gpu_buffer_dump.h
 * @brief Correctness check for the GPU transfer buffer layouts
 * (--enable-buffer-dump).
 *
 * In this mode, the simulation
 *  - initialises every particle with deterministic, particle-dependent values,
 *  - replays all events serially (one OpenMP thread), so the result does not
 *    depend on thread scheduling,
 *  - fills the recv buffer region with deterministic values before each
 *    unpack (on a real run, the GPU would have written them),
 *  - writes the buffer region after each pack and each unpack, and the final
 *    particle state, to GPU_BUFFER_DUMP_FILE.
 *
 * Buffers are written field by field (in the canonical order of
 * gpu_part_buffer_access.h), particle by particle, independently of the buffer
 * layout. Dumps of builds that only differ in buffer layout must therefore be
 * bitwise identical.
 *
 * Record format (native endianness):
 *   pack/unpack: int32 {tag, subtype, pack_index, count}, then for each field
 *                count values of 4 bytes.
 *   particles:   int32 {tag, 0, 0, nr_parts}, then for each quantity
 *                nr_parts values (doubles for x, int8 for the time bins, floats
 *                otherwise).
 */

#include "../config.h"

#ifdef GPU_BUFFER_DUMP

#if !defined(SWIFT_PARTICLE_ACCESS_GLOBAL_VAR)
#error "--enable-buffer-dump requires --with-particle-access=global-var"
#endif

#include "gpu_part_buffer_access.h"
#include "hydro_part.h"
#include "hydro_part_init.h"
#include "swift_placeholders/error.h"
#include "swift_placeholders/task.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define GPU_BUFFER_DUMP_FILE "gpu_buffer_dump.bin"

enum gpu_buffer_dump_tag {
  gpu_buffer_dump_tag_pack = 1,
  gpu_buffer_dump_tag_unpack = 2,
  gpu_buffer_dump_tag_particles = 3,
};

/*! Output file. Only touched from the (single) replaying thread. */
static FILE* gpu_buffer_dump_fp = NULL;

/*! Counts unpack events, used to vary the recv buffer contents */
static int gpu_buffer_dump_unpack_counter = 0;

/**
 * @brief Open the dump file.
 */
static void gpu_buffer_dump_open(void) {
  gpu_buffer_dump_fp = fopen(GPU_BUFFER_DUMP_FILE, "wb");
  if (gpu_buffer_dump_fp == NULL)
    error("Could not open %s for writing", GPU_BUFFER_DUMP_FILE);
  gpu_buffer_dump_unpack_counter = 0;
}

/**
 * @brief Close the dump file.
 */
static void gpu_buffer_dump_close(void) {
  fclose(gpu_buffer_dump_fp);
  gpu_buffer_dump_fp = NULL;
  message("Written buffer dump to %s", GPU_BUFFER_DUMP_FILE);
}

/**
 * @brief Write a record header.
 */
static void gpu_buffer_dump_header(const int tag, const int subtype,
                                   const int pack_index, const int count) {
  const int32_t h[4] = {tag, subtype, pack_index, count};
  if (fwrite(h, sizeof(int32_t), 4, gpu_buffer_dump_fp) != 4)
    error("Failed to write buffer dump");
}

/**
 * @brief Write n bytes of data.
 */
static void gpu_buffer_dump_write(const void* data, const size_t size,
                                  const size_t n) {
  if (fwrite(data, size, n, gpu_buffer_dump_fp) != n)
    error("Failed to write buffer dump");
}

/* Write all fields of element range [0, count) of a buffer handle b, field by
 * field, through the accessors. Used with the GPU_<BUF>_FIELDS lists. */
#define GPU_BUFFER_DUMP_FIELD(buf, field, ctype)     \
  for (int j = 0; j < count; j++) {                  \
    const ctype val = gpu_##buf##_get_##field(b, j); \
    gpu_buffer_dump_write(&val, sizeof(ctype), 1);   \
  }

/* Fill element range [0, count) of recv buffer handle b with deterministic
 * values, distinct per field, element and unpack event. All values are exact
 * in single precision. min_ngb_time_bin must stay a valid time bin since the
 * unpack converts it to timebin_t. */
#define GPU_BUFFER_DUMP_FILL_FIELD(buf, field, ctype)                      \
  {                                                                        \
    const int is_time_bin = (strcmp(#field, "min_ngb_time_bin") == 0);     \
    for (int j = 0; j < count; j++) {                                      \
      const int n = (field_index * 977 + j * 13 + evt * 7) % 2048;         \
      const float val =                                                    \
          is_time_bin ? (float)((j + evt) % 9) : 0.25f * (float)n - 100.f; \
      gpu_##buf##_set_##field(b, j, (ctype)val);                           \
    }                                                                      \
    field_index++;                                                         \
  }

#define GPU_BUFFER_DUMP_DEFINE_PACK(buf, BUF, subtype)                       \
  static void gpu_buffer_dump_##buf(const gpu_##buf##_buf buffer,            \
                                    const int pack_index, const int count) { \
    gpu_buffer_dump_header(gpu_buffer_dump_tag_pack, subtype, pack_index,    \
                           count);                                           \
    const gpu_##buf##_buf b = gpu_##buf##_offset(buffer, pack_index);        \
    GPU_##BUF##_FIELDS(GPU_BUFFER_DUMP_FIELD)                                \
  }

#define GPU_BUFFER_DUMP_DEFINE_UNPACK(buf, BUF, subtype)                     \
  static void gpu_buffer_fill_##buf(const gpu_##buf##_buf buffer,            \
                                    const int pack_index, const int count) { \
    const gpu_##buf##_buf b = gpu_##buf##_offset(buffer, pack_index);        \
    const int evt = gpu_buffer_dump_unpack_counter++;                        \
    int field_index = 0;                                                     \
    GPU_##BUF##_FIELDS(GPU_BUFFER_DUMP_FILL_FIELD)                           \
  }                                                                          \
  static void gpu_buffer_dump_##buf(const gpu_##buf##_buf buffer,            \
                                    const int pack_index, const int count) { \
    gpu_buffer_dump_header(gpu_buffer_dump_tag_unpack, subtype, pack_index,  \
                           count);                                           \
    const gpu_##buf##_buf b = gpu_##buf##_offset(buffer, pack_index);        \
    GPU_##BUF##_FIELDS(GPU_BUFFER_DUMP_FIELD)                                \
  }

GPU_BUFFER_DUMP_DEFINE_PACK(send_d, SEND_D, task_subtype_density)
GPU_BUFFER_DUMP_DEFINE_PACK(send_g, SEND_G, task_subtype_gradient)
GPU_BUFFER_DUMP_DEFINE_PACK(send_f, SEND_F, task_subtype_force)
GPU_BUFFER_DUMP_DEFINE_UNPACK(recv_d, RECV_D, task_subtype_density)
GPU_BUFFER_DUMP_DEFINE_UNPACK(recv_g, RECV_G, task_subtype_gradient)
GPU_BUFFER_DUMP_DEFINE_UNPACK(recv_f, RECV_F, task_subtype_force)

/**
 * @brief Initialise all particles with deterministic values that differ
 * between particles and between quantities. Requires global_hydro_part_arrays
 * to be set. All values are exact in single precision.
 */
static void gpu_buffer_dump_init_particles(struct hydro_part_arrays* part_data,
                                           const size_t nr_parts) {

  for (size_t pid = 0; pid < nr_parts; pid++) {

    /* Start from the default initialisation, then make values distinct */
    hydro_part_init(part_data, pid);

    const float r = 0.0625f * (float)(pid % 1021);
    for (int k = 0; k < 3; k++) {
      part_set_x_ind_global(pid, k, (double)(k + 1) + 0.001 * (pid % 4099));
      part_set_v_ind_global(pid, k, 4.f + k + r);
      part_set_a_hydro_ind_global(pid, k, 7.f + k + r);
      part_set_rot_v_ind_global(pid, k, 10.f + k + r);
    }
    part_set_mass_global(pid, 13.f + r);
    part_set_h_global(pid, 14.f + r);
    part_set_u_global(pid, 15.f + r);
    part_set_u_dt_global(pid, 16.f + r);
    part_set_rho_global(pid, 17.f + r);
    part_set_div_v_global(pid, 18.f + r);
    part_set_alpha_av_global(pid, 19.f + r);
    part_set_v_sig_global(pid, 20.f + r);
    part_set_laplace_u_global(pid, 21.f + r);
    part_set_alpha_diff_global(pid, 22.f + r);
    part_set_wcount_global(pid, 23.f + r);
    part_set_wcount_dh_global(pid, 24.f + r);
    part_set_rho_dh_global(pid, 25.f + r);
    part_set_f_gradh_global(pid, 26.f + r);
    part_set_pressure_global(pid, 27.f + r);
    part_set_soundspeed_global(pid, 28.f + r);
    part_set_h_dt_global(pid, 29.f + r);
    part_set_balsara_global(pid, 30.f + r);
    part_set_alpha_visc_max_ngb_global(pid, 31.f + r);
    /* Keep everything active: time bins <= engine.max_active_bin */
    part_set_time_bin_global(pid, (timebin_t)(1 + pid % 7));
    part_set_timestep_limiter_min_ngb_time_bin_global(pid,
                                                      (timebin_t)(1 + pid % 5));
  }
}

/* Write a single particle quantity for all particles */
#define GPU_BUFFER_DUMP_PART_QUANTITY(type, expr) \
  for (size_t pid = 0; pid < nr_parts; pid++) {   \
    const type val = (expr);                      \
    gpu_buffer_dump_write(&val, sizeof(type), 1); \
  }

/**
 * @brief Write the final state of every particle quantity that is packed or
 * unpacked.
 */
static void gpu_buffer_dump_particles(const size_t nr_parts) {

  gpu_buffer_dump_header(gpu_buffer_dump_tag_particles, 0, 0, (int)nr_parts);

  for (int k = 0; k < 3; k++) {
    GPU_BUFFER_DUMP_PART_QUANTITY(double, part_get_x_ind_global(pid, k));
    GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_v_ind_global(pid, k));
    GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_a_hydro_global(pid)[k]);
    GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_rot_v_global(pid)[k]);
  }
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_mass_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_h_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_u_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_u_dt_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_rho_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_div_v_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_alpha_av_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_v_sig_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_laplace_u_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_alpha_diff_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_wcount_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_wcount_dh_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_rho_dh_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_f_gradh_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_pressure_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_soundspeed_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_h_dt_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_balsara_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(float, part_get_alpha_visc_max_ngb_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(timebin_t, part_get_time_bin_global(pid));
  GPU_BUFFER_DUMP_PART_QUANTITY(
      timebin_t, part_get_timestep_limiter_min_ngb_time_bin_global(pid));
}

#endif /* GPU_BUFFER_DUMP */

#endif /* SWIFTGPUPACKSIM_GPU_BUFFER_DUMP_H */
