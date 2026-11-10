/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *                    Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
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
#pragma once

#ifndef USE_LOOP_SPLIT_BY_ELEMENT
#error "How did we get here?"
#endif

/**
 * @file gpu_part_pack_functions_global_var_access_loop_split_by_struct.h
 * @brief Functions related to packing and unpacking particles to/from a cell.
 * Uses global variable for access in getters/setters. Loops are split by
 * element, i.e. SoA access.
 */

#include "active.h"
#include "cell.h"
#include "gpu_part_buffer_access.h"
#include "gpu_part_structs.h"
#include "hydro_part.h"

#include <math.h>


/* Apparently redundant declaration */
/* extern struct hydro_part_arrays global_hydro_part_arrays; */

/**
 * @brief Unpacks the density data from GPU buffers into cell's particle arrays
 *
 * @param c the #cell
 * @param parts_buffer the "receive" buffer to copy from
 * @param unpack_ind the index in parts_buffer to start copying from
 * @param count number of particles to unpack
 * @param e the #engine
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_density(
    struct cell* restrict c, const gpu_recv_d_buf parts_buffer,
    const int unpack_ind, const int count, const struct engine* restrict e) {

  const ptrdiff_t first =
      c->hydro.part_arrs._part - global_hydro_part_arrays._part;
  const gpu_recv_d_buf parts_recv = gpu_recv_d_offset(parts_buffer, unpack_ind);

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float rho = part_get_rho_global(pi) + gpu_recv_d_get_rho(parts_recv, i);
    part_set_rho_global(pi, rho);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float rho_dh =
        part_get_rho_dh_global(pi) + gpu_recv_d_get_rho_dh(parts_recv, i);
    part_set_rho_dh_global(pi, rho_dh);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float wcount =
        part_get_wcount_global(pi) + gpu_recv_d_get_wcount(parts_recv, i);
    part_set_wcount_global(pi, wcount);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float wcount_dh =
        part_get_wcount_dh_global(pi) + gpu_recv_d_get_wcount_dh(parts_recv, i);
    part_set_wcount_dh_global(pi, wcount_dh);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float* rot_v = part_get_rot_v_global(pi);
    rot_v[0] += gpu_recv_d_get_rot_vx(parts_recv, i);
    rot_v[1] += gpu_recv_d_get_rot_vy(parts_recv, i);
    rot_v[2] += gpu_recv_d_get_rot_vz(parts_recv, i);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float div_v =
        part_get_div_v_global(pi) + gpu_recv_d_get_div_v(parts_recv, i);
    part_set_div_v_global(pi, div_v);
  }
}

/**
 * @brief Unpacks the gradient data from GPU buffers into cell's particle arrays
 *
 * @param c the #cell
 * @param parts_buffer the "receive" buffer to copy from
 * @param unpack_ind the index in parts_buffer to start copying from
 * @param count number of particles to unpack
 * @param e the #engine
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_gradient(
    struct cell* restrict c, const gpu_recv_g_buf parts_buffer,
    const int unpack_ind, const int count, const struct engine* e) {

  const ptrdiff_t first =
      c->hydro.part_arrs._part - global_hydro_part_arrays._part;
  const gpu_recv_g_buf parts_recv = gpu_recv_g_offset(parts_buffer, unpack_ind);

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float avisc_old = part_get_alpha_visc_max_ngb_global(pi);
    float avisc =
        fmaxf(avisc_old, gpu_recv_g_get_alpha_visc_max_ngb(parts_recv, i));
    part_set_alpha_visc_max_ngb_global(pi, avisc);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float vsig =
        fmaxf(gpu_recv_g_get_v_sig(parts_recv, i), part_get_v_sig_global(pi));
    part_set_v_sig_global(pi, vsig);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float lu =
        gpu_recv_g_get_laplace_u(parts_recv, i) + part_get_laplace_u_global(pi);
    part_set_laplace_u_global(pi, lu);
  }
}

/**
 * @brief Unpacks the force data from GPU buffers into cell's particle arrays
 *
 * @param c the #cell
 * @param parts_buffer the "receive" buffer to copy from
 * @param unpack_ind the index in parts_buffer to start copying from
 * @param count number of particles to unpack
 * @param e the #engine
 */
__attribute__((always_inline)) INLINE static void gpu_unpack_part_force(
    struct cell* restrict c, const gpu_recv_f_buf parts_buffer,
    const int unpack_ind, const int count, const struct engine* e) {

  const ptrdiff_t first =
      c->hydro.part_arrs._part - global_hydro_part_arrays._part;
  const gpu_recv_f_buf parts_recv = gpu_recv_f_offset(parts_buffer, unpack_ind);

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float* a = part_get_a_hydro_global(pi);
    a[0] += gpu_recv_f_get_a_x(parts_recv, i);
    a[1] += gpu_recv_f_get_a_y(parts_recv, i);
    a[2] += gpu_recv_f_get_a_z(parts_recv, i);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float u_dt = gpu_recv_f_get_u_dt(parts_recv, i) + part_get_u_dt_global(pi);
    part_set_u_dt_global(pi, u_dt);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    float h_dt = gpu_recv_f_get_h_dt(parts_recv, i) + part_get_h_dt_global(pi);

    part_set_h_dt_global(pi, h_dt);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    if (!part_is_active_global(pi, e)) continue;
    timebin_t mintbin =
        (timebin_t)(gpu_recv_f_get_min_ngb_time_bin(parts_recv, i) + 0.5f);
    part_set_timestep_limiter_min_ngb_time_bin_global(pi, mintbin);
  }
}

/**
 * @brief Packs the cell particle data for density interactions into the
 * CPU-side buffers.
 *
 * @param c the #cell
 * @param parts_buffer the buffer to pack into
 * @param pack_ind the first free index in the buffer arrays to copy data into
 * @param shift periodic boundary shift
 * @param cjstart start index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 * @param cjend end index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_density(
    const struct cell* restrict c, const gpu_send_d_buf parts_buffer,
    const int pack_ind, const double shift[3], const int cjstart,
    const int cjend) {

  /* Grab handles */
  const int count = c->hydro.count;
  const ptrdiff_t first =
      c->hydro.part_arrs._part - global_hydro_part_arrays._part;
  const gpu_send_d_buf ps = gpu_send_d_offset(parts_buffer, pack_ind);

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    const double* x = part_get_const_x_global(pi);
    gpu_send_d_set_x(ps, i, x[0] - shift[0]);
    gpu_send_d_set_y(ps, i, x[1] - shift[1]);
    gpu_send_d_set_z(ps, i, x[2] - shift[2]);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    gpu_send_d_set_h(ps, i, part_get_h_global(pi));
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    const float* v = part_get_const_v_global(pi);
    gpu_send_d_set_vx(ps, i, v[0]);
    gpu_send_d_set_vy(ps, i, v[1]);
    gpu_send_d_set_vz(ps, i, v[2]);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    gpu_send_d_set_m(ps, i, part_get_mass_global(pi));
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    gpu_send_d_set_pjs(ps, i, cjstart);
    gpu_send_d_set_pje(ps, i, cjend);
  }
}

/**
 * @brief Packs the cell particle data for gradient interactions into
 * the CPU-side buffers.
 *
 * @param ci the #cell
 * @param parts_buffer the buffer to pack into
 * @param pack_ind the first free index in the buffer arrays to copy data into
 * @param shift periodic boundary shift
 * @param cjstart start index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 * @param cjend end index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_gradient(
    const struct cell* restrict ci, const gpu_send_g_buf parts_buffer,
    const int pack_ind, const double shift[3], const int cjstart,
    const int cjend) {

  /* Grab handles */
  const ptrdiff_t first =
      ci->hydro.part_arrs._part - global_hydro_part_arrays._part;
  const int count = ci->hydro.count;
  const gpu_send_g_buf ps = gpu_send_g_offset(parts_buffer, pack_ind);

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    const double* x = part_get_const_x_global(pi);
    gpu_send_g_set_x(ps, i, x[0] - shift[0]);
    gpu_send_g_set_y(ps, i, x[1] - shift[1]);
    gpu_send_g_set_z(ps, i, x[2] - shift[2]);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    gpu_send_g_set_h(ps, i, part_get_h_global(pi));
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    const float* v = part_get_const_v_global(pi);
    gpu_send_g_set_vx(ps, i, v[0]);
    gpu_send_g_set_vy(ps, i, v[1]);
    gpu_send_g_set_vz(ps, i, v[2]);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    gpu_send_g_set_m(ps, i, part_get_mass_global(pi));
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_FORCE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_FORCE_TYPE_PARTICLE)
    gpu_send_g_set_u(ps, i, part_get_u_global(pi));
#else
    gpu_send_g_set_u(ps, i, part_get_u_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_FORCE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_FORCE_TYPE_PARTICLE)
    gpu_send_g_set_rho(ps, i, part_get_rho_global(pi));
#else
    gpu_send_g_set_rho(ps, i, part_get_rho_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_FORCE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_FORCE_TYPE_PARTICLE)
    gpu_send_g_set_c(ps, i, part_get_soundspeed_global(pi));
#else
    gpu_send_g_set_c(ps, i, part_get_soundspeed_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_FORCE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_FORCE_TYPE_PARTICLE)
    gpu_send_g_set_alpha_visc(ps, i, part_get_alpha_av_global(pi));
#else
    gpu_send_g_set_alpha_visc_max_ngb(ps, i,
                                      part_get_alpha_visc_max_ngb_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_FORCE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_FORCE_TYPE_PARTICLE)
    gpu_send_g_set_alpha_visc_max_ngb(ps, i,
                                      part_get_alpha_visc_max_ngb_global(pi));
#else
    gpu_send_g_set_alpha_visc(ps, i, part_get_alpha_av_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_FORCE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_FORCE_TYPE_PARTICLE)
    gpu_send_g_set_v_sig(ps, i, part_get_v_sig_global(pi));
#else
    gpu_send_g_set_v_sig(ps, i, part_get_v_sig_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_FORCE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_FORCE_TYPE_PARTICLE)
    gpu_send_g_set_laplace_u(ps, i, part_get_laplace_u_global(pi));
#else
    gpu_send_g_set_laplace_u(ps, i, part_get_laplace_u_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    gpu_send_g_set_pjs(ps, i, cjstart);
    gpu_send_g_set_pje(ps, i, cjend);
  }
}

/**
 * @brief Packs the cell particle data for force interactions into the
 * CPU-side buffers.
 *
 * @param ci the #cell
 * @param parts_buffer the buffer to pack into
 * @param pack_ind the first free index in the buffer arrays to copy data into
 * @param shift periodic boundary shift
 * @param cjstart start index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 * @param cjend end index of cell cj's particles (which cell ci is to be
 * interacted with) in buffer
 */
__attribute__((always_inline)) INLINE static void gpu_pack_part_force(
    const struct cell* restrict ci, const gpu_send_f_buf parts_buffer,
    const int pack_ind, const double shift[3], const int cjstart,
    const int cjend) {

  const int count = ci->hydro.count;
  const ptrdiff_t first =
      ci->hydro.part_arrs._part - global_hydro_part_arrays._part;
  const gpu_send_f_buf ps = gpu_send_f_offset(parts_buffer, pack_ind);

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    const double* x = part_get_const_x_global(pi);
    gpu_send_f_set_x(ps, i, x[0] - shift[0]);
    gpu_send_f_set_y(ps, i, x[1] - shift[1]);
    gpu_send_f_set_z(ps, i, x[2] - shift[2]);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    gpu_send_f_set_h(ps, i, part_get_h_global(pi));
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    const float* v = part_get_const_v_global(pi);
    gpu_send_f_set_vx(ps, i, v[0]);
    gpu_send_f_set_vy(ps, i, v[1]);
    gpu_send_f_set_vz(ps, i, v[2]);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    gpu_send_f_set_m(ps, i, part_get_mass_global(pi));
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_u(ps, i, part_get_u_global(pi));
#else
    gpu_send_f_set_u(ps, i, part_get_u_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_rho(ps, i, part_get_rho_global(pi));
#else
    gpu_send_f_set_rho(ps, i, part_get_rho_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_c(ps, i, part_get_soundspeed_global(pi));
#else
    gpu_send_f_set_f_gradh(ps, i, part_get_f_gradh_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_alpha_visc(ps, i, part_get_alpha_av_global(pi));
#else
    gpu_send_f_set_pressure(ps, i, part_get_pressure_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_f_gradh(ps, i, part_get_f_gradh_global(pi));
#else
    gpu_send_f_set_balsara(ps, i, part_get_balsara_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_pressure(ps, i, part_get_pressure_global(pi));
#else
    gpu_send_f_set_c(ps, i, part_get_soundspeed_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_balsara(ps, i, part_get_balsara_global(pi));
#else
    gpu_send_f_set_alpha_visc(ps, i, part_get_alpha_av_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
#if defined(SPHENIX_PACK_GRADIENT_PARTICLE) || \
    defined(SPHENIX_PACK_GRADIENT_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_TYPE_PARTICLE) || \
    defined(SPHENIX_PACK_SHARED_PARTICLE)
    gpu_send_f_set_alpha_diff(ps, i, part_get_alpha_diff_global(pi));
#else
    gpu_send_f_set_alpha_diff(ps, i, part_get_alpha_diff_global(pi));
#endif
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    gpu_send_f_set_time_bin(ps, i, (int)part_get_time_bin_global(pi));
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    ptrdiff_t pi = i + first;
    int mintbin = (int)part_get_timestep_limiter_min_ngb_time_bin_global(pi);
    gpu_send_f_set_min_ngb_time_bin(ps, i, mintbin);
  }

#ifdef VECTORIZE
#pragma omp simd
#endif
  for (ptrdiff_t i = 0; i < count; i++) {
    gpu_send_f_set_pjs(ps, i, cjstart);
    gpu_send_f_set_pje(ps, i, cjend);
  }
}
