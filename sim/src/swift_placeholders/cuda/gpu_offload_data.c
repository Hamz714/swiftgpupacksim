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

/**
 * @file cuda/gpu_offload_data.c
 * @brief functions related to the gpu_offload_data struct, containing data
 * required for offloading
 */

#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

#include "../error.h"
#include "../memuse.h"
#include "gpu_offload_data.h"

#include <cuda.h>
#include <cuda_runtime.h>

#if defined(GPU_BUFFER_SOA)

/**
 * @brief View a SoA buffer struct as an array of its field pointers.
 *
 * All SoA buffer structs (see gpu_part_structs.h) consist only of pointers to
 * 4-byte scalars, so their members can be handled generically.
 */
static void** gpu_soa_buffer_fields(void* soa_struct) {
  return (void**)soa_struct;
}

/**
 * @brief Allocate every field array of a SoA buffer struct.
 *
 * @param soa_struct: pointer to the SoA struct whose pointers are set
 * @param n_fields: number of field arrays in the struct
 * @param n_elements: number of particles each array holds
 */
static void gpu_soa_buffer_alloc(void* soa_struct, const size_t n_fields,
                                 const size_t n_elements) {

  void** fields = gpu_soa_buffer_fields(soa_struct);

  for (size_t f = 0; f < n_fields; f++) {
    /* Stagger the start of each array, see GPU_BUFFER_SOA_STAGGER */
    const size_t offset = f * GPU_BUFFER_SOA_STAGGER;
    const size_t bytes = n_elements * GPU_BUFFER_SOA_ELEMENT_SIZE + offset;
    void* base = NULL;
#if defined(NO_CUDA_MALLOCHOST)
    if (swift_memalign("gpu_soa_buffer", &base, SWIFT_PART_ALIGNMENT, bytes) !=
        0)
      error("Error while allocating memory for SoA GPU buffer");
#else
    cudaError_t cu_error = cudaMallocHost(&base, bytes);
    swift_assert(cu_error == cudaSuccess);
#endif
    fields[f] = (char*)base + offset;
  }
}

/**
 * @brief Free every field array of a SoA buffer struct.
 */
static void gpu_soa_buffer_free(void* soa_struct, const size_t n_fields) {

  void** fields = gpu_soa_buffer_fields(soa_struct);

  for (size_t f = 0; f < n_fields; f++) {
    void* base = (char*)fields[f] - f * GPU_BUFFER_SOA_STAGGER;
#if defined(NO_CUDA_MALLOCHOST)
    free(base);
#else
    cudaError_t cu_error = cudaFreeHost(base);
    swift_assert(cu_error == cudaSuccess);
#endif
    fields[f] = NULL;
  }
}

#ifdef SWIFT_DEBUG_CHECKS
/**
 * @brief Zero out every field array of a SoA buffer struct.
 */
static void gpu_soa_buffer_zero(void* soa_struct, const size_t n_fields,
                                const size_t n_elements) {

  void** fields = gpu_soa_buffer_fields(soa_struct);
  for (size_t f = 0; f < n_fields; f++)
    memset(fields[f], 0, n_elements * GPU_BUFFER_SOA_ELEMENT_SIZE);
}
#endif

#endif /* GPU_BUFFER_SOA */

/**
 * @brief initialise GPU data buffers (including their associated metadata)
 *
 * @param buf: the buffers to be initialised
 * @param params: global gpu packing parameters
 * @param send_struct_size: bytes per particle of the send buffer (both host
 * and device). For AoS buffers, this is the size of the send struct; for SoA
 * buffers, it is the sum over all field arrays, i.e. n_fields *
 * GPU_BUFFER_SOA_ELEMENT_SIZE (see GPU_PART_SEND_*_BYTES).
 * @param recv_struct_size: bytes per particle of the recv buffer, same
 * convention as send_struct_size.
 */
void gpu_data_buffers_init(struct gpu_offload_data* buf,
                           const struct gpu_global_pack_params* params,
                           const size_t send_struct_size,
                           const size_t recv_struct_size) {

  /* Grab some handles */
  /* const size_t n_bundles = params->n_bundles; */
  const size_t part_buffer_size = params->part_buffer_size;

  /* Initialise and set up metadata */
  struct gpu_pack_metadata* md = &(buf->md);
  gpu_pack_metadata_init(md, params);

#if defined(GPU_BUFFER_SOA)
  /* One array per scalar field. The send/recv unions alias the density,
   * gradient and force SoA structs, so allocate through the density members
   * with the number of fields of whichever buffer this is. */
  buf->n_send_fields = send_struct_size / GPU_BUFFER_SOA_ELEMENT_SIZE;
  buf->n_recv_fields = recv_struct_size / GPU_BUFFER_SOA_ELEMENT_SIZE;
  gpu_soa_buffer_alloc(&buf->parts_send_d, buf->n_send_fields,
                       part_buffer_size);
  gpu_soa_buffer_alloc(&buf->parts_recv_d, buf->n_recv_fields,
                       part_buffer_size);
#elif defined(NO_CUDA_MALLOCHOST)
  /* Hack to get a working test version on my own machine. */
  /* buf->parts_send_d = malloc(part_buffer_size * send_struct_size); */
  /* swift_assert(buf->parts_send_d != NULL); */

  if (swift_memalign("parts_part", (void**)&buf->parts_send_d,
                     SWIFT_PART_ALIGNMENT,
                     part_buffer_size * send_struct_size) != 0)
    error("Error while allocating memory for SPH particles");

  buf->parts_recv_d = malloc(part_buffer_size * recv_struct_size);
  swift_assert(buf->parts_recv_d != NULL);
#else
  /* Now allocate arrays */
  cudaError_t cu_error = cudaSuccess;

  /* Now allocate memory for Buffer and GPU particle arrays */
  /* cu_error = cudaMalloc((void **)&buf->d_parts_send_d, */
  /*                       part_buffer_size * send_struct_size); */
  /* swift_assert(cu_error == cudaSuccess); */

  /* cu_error = cudaMalloc((void **)&buf->d_parts_recv_d, */
  /*                       part_buffer_size * recv_struct_size); */
  /* swift_assert(cu_error == cudaSuccess); */
  cu_error = cudaMallocHost((void**)&buf->parts_send_d,
                            part_buffer_size * send_struct_size);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaMallocHost((void**)&buf->parts_recv_d,
                            part_buffer_size * recv_struct_size);
  swift_assert(cu_error == cudaSuccess);
#endif

  /* Create space for cuda events */
  /* buf->event_end = (cudaEvent_t *)malloc(n_bundles * sizeof(cudaEvent_t)); */

  /* for (size_t i = 0; i < n_bundles; i++) { */
  /*   cu_error = cudaEventCreate(&(buf->event_end[i])); */
  /*   swift_assert(cu_error == cudaSuccess); */
  /* } */

#ifdef SWIFT_DEBUG_CHECKS
  md->send_struct_size = send_struct_size;
  md->recv_struct_size = recv_struct_size;
#endif
}

/**
 * @brief perform the initialisations required at the start of each step
 */
void gpu_data_buffers_init_step(struct gpu_offload_data* buf) {

  struct gpu_pack_metadata* md = &buf->md;
  gpu_pack_metadata_init_step(md);
  gpu_data_buffers_reset(buf);
}

/**
 * @brief reset (zero out) the data buffers.
 */
void gpu_data_buffers_reset(struct gpu_offload_data* buf) {

#ifdef SWIFT_DEBUG_CHECKS

  /* In principle, if our book-keeping is correct, we shouldn't ever
   * need to zero out the contents. So we don't do it outside of debug
   * mode. */

  const struct gpu_global_pack_params pars = buf->md.params;

#if defined(GPU_BUFFER_SOA)
  gpu_soa_buffer_zero(&buf->parts_send_d, buf->n_send_fields,
                      pars.part_buffer_size);
  gpu_soa_buffer_zero(&buf->parts_recv_d, buf->n_recv_fields,
                      pars.part_buffer_size);
#else
  const struct gpu_pack_metadata md = buf->md;
  memset(buf->parts_send_d, 0, pars.part_buffer_size * md.send_struct_size);
  memset(buf->parts_recv_d, 0, pars.part_buffer_size * md.recv_struct_size);
#endif

  /* Can't do this from the host side, would need to launch cuda kernel */
  /* bzero(buf->d_parts_recv_d, pars.part_buffer_size *
   * sizeof(md.send_struct_size)); */
  /* bzero(buf->d_parts_send_d, pars.part_buffer_size) *
   * sizeof(md.recv_struct_size); */

#endif
}

/**
 * @brief Free everything you allocated.
 */
void gpu_data_buffers_free(struct gpu_offload_data* buf) {

  struct gpu_pack_metadata* md = &(buf->md);
  gpu_pack_metadata_free(md);

#if defined(GPU_BUFFER_SOA)
  /* Same aliasing via the density members as in gpu_data_buffers_init */
  gpu_soa_buffer_free(&buf->parts_send_d, buf->n_send_fields);
  gpu_soa_buffer_free(&buf->parts_recv_d, buf->n_recv_fields);
#elif defined(NO_CUDA_MALLOCHOST)
  free(buf->parts_send_d);
  free(buf->parts_recv_d);
#else
  cudaError_t cu_error = cudaErrorMemoryAllocation;

  /* cu_error = cudaFree(buf->d_parts_send_d); */
  /* swift_assert(cu_error == cudaSuccess); */

  /* cu_error = cudaFree(buf->d_parts_recv_d); */
  /* swift_assert(cu_error == cudaSuccess); */

  cu_error = cudaFreeHost(buf->parts_send_d);
  swift_assert(cu_error == cudaSuccess);

  cu_error = cudaFreeHost(buf->parts_recv_d);
  swift_assert(cu_error == cudaSuccess);
#endif

  /* free((void *)buf->event_end); */
}

#ifdef __cplusplus
}
#endif
