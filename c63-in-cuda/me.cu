// clang-format off
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "c63.h"
#include "me.h"
#include "tables.h"

#include <cuda.h>
#include <cuda_runtime.h>

static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int u, v;

  *result = 0;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      *result += abs(block2[v*stride+u] - block1[v*stride+u]);
    }
  }
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad = INT_MAX;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      int sad;
      sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, &sad);

      /* printf("(%4d,%4d) - %d\n", x, y, sad); */

      if (sad < best_sad)
      {
        mb->mv_x = x - mx;
        mb->mv_y = y - my;
        best_sad = sad;
      }
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
          cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}


/* ~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~ */
/*                  CUDA SECTION                      */
/* ~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~ */

// Comment to diable
#define LINE_ERR

#ifdef LINE_ERR
#define LINE_CHECK()                                                           \
  {                                                                            \
    printf("\n(in %s at line %d)\n", __FILE__, __LINE__);                      \
  }

#else
#define LINE_CHECK()
#endif

/**************** CUDA ERR DEBUG ****************/

// Comment to diable
#define DEBUG_ERR_CUDA

// Cuda error check macro
#ifdef DEBUG_CUDA
#define CUDA_ERR_CHECK()                                                       \
  {                                                                            \
    cudaDeviceSynchronize();                                                   \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      printf("\nCuda err: %s (in %s at line %d)\n", cudaGetErrorString(err),   \
             __FILE__, __LINE__);                                              \
    }                                                                          \
  }
#else
#define CUDA_ERR_CHECK() // Macro does nothing when DEBUG_CUDA is not defined
#endif
/**************** CUDA ERR DEBUG ****************/


static void gpu_sad_block_8x8()
{

}

/* Motion estimation for 8x8 block */
static void gpu_me_block_8x8()
{

}

#define BLOCK_SIZE 8  // Process 8x8 macroblocks

__device__ int gpu_compute_sad_8x8(uint8_t *orig, uint8_t *ref, int stride)
{
    int sad = 0;
    for (int v = 0; v < 8; ++v)
    {
        for (int u = 0; u < 8; ++u)
        {
            sad += abs(orig[v * stride + u] - ref[v * stride + u]);
        }
    }
    return sad;
}

__global__ void gpu_c63_motion_estimate_kernel(uint8_t *orig, uint8_t *ref, struct macroblock *mbs,
                                               int w, int h, int range, int mb_cols)
{
    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;
    int mb_index = mb_y * mb_cols + mb_x;

    int mx = mb_x * BLOCK_SIZE;
    int my = mb_y * BLOCK_SIZE;
    
    int best_sad = INT_MAX;
    int best_mv_x = 0, best_mv_y = 0;

    int left = max(mx - range, 0);
    int top = max(my - range, 0);
    int right = min(mx + range, w - BLOCK_SIZE);
    int bottom = min(my + range, h - BLOCK_SIZE);

    for (int y = top; y < bottom; ++y)
    {
        for (int x = left; x < right; ++x)
        {
            int sad = gpu_compute_sad_8x8(&orig[my * w + mx], &ref[y * w + x], w);
            if (sad < best_sad)
            {
                best_sad = sad;
                best_mv_x = x - mx;
                best_mv_y = y - my;
            }
        }
    }

    mbs[mb_index].mv_x = best_mv_x;
    mbs[mb_index].mv_y = best_mv_y;
    mbs[mb_index].use_mv = 1;
}

void gpu_c63_motion_estimate(struct c63_common *cm)
{
    uint8_t *d_orig, *d_ref;
    struct macroblock *d_mbs;
    
    int size_y = cm->padw[Y_COMPONENT] * cm->padh[Y_COMPONENT];
    int mb_size = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);

    cudaMalloc(&d_orig, size_y);
    cudaMalloc(&d_ref, size_y);
    cudaMalloc(&d_mbs, mb_size);

    cudaMemcpy(d_orig, cm->curframe->orig->Y, size_y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, cm->refframe->recons->Y, size_y, cudaMemcpyHostToDevice);

    dim3 grid(cm->mb_cols, cm->mb_rows);
    gpu_c63_motion_estimate_kernel<<<grid, 1>>>(d_orig, d_ref, d_mbs,
                                               cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT],
                                               cm->me_search_range, cm->mb_cols);
    
    cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], d_mbs, mb_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_orig);
    cudaFree(d_ref);
    cudaFree(d_mbs);
}


/* Motion compensation for 8x8 block */
static void gpu_mc_block_8x8()
{

}

__global__ static void gpu_c63_motion_compensate_kernel()
{

}

void gpu_c63_motion_compensate(struct c63_common *cm)
{

} 
