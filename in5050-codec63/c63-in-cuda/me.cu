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

#include "me.h"
#include "tables.h"

extern struct c63_common *d_cm;
extern struct frame *d_refframe, *d_curframe;

extern yuv_t *d_curframe_orig, *d_refframe_recons, *d_curframe_predicted;
extern yuv_buf *d_curframe_origbuf, *d_refframe_reconsbuf, *d_curframe_predictedbuf;

extern struct macroblock *d_curframe_mby, *d_curframe_mbu, *d_curframe_mbv;

#define SHUFFLE_FULL_MASK 0xffffffff

__device__ static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  __shared__ int part_sad_sums[2];

  int lane_index = threadIdx.x
  int warp_index = threadIdx.y;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  __syncwarp();

  int shuffled_result = abs(block2[warp_index*stride+lane_index] - block1[warp_index*stride+lane_index])

  for (int offset = 16; offset > 0; offset /= 2) {
    shuffled_result += __shfl_down_sync(SHUFFLE_FULL_MASK, shuffled_result, offset);
  }

  __syncwarp();
  if (lane_index == 0) {
    part_sad_sums[warp_index] = shuffled_result;
  }

  __syncthreads();
  if (tid == 0){
    *result = part_sad_sums[0] + part_sad_sums[1];
  }
}

/* Motion estimation for 8x8 block */
__device__ static void me_block_8x8(
  struct c63_common *cm, 
  int mb_x, 
  int mb_y, 
  uint8_t *orig, 
  uint8_t *ref, 
  int color_component
)
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  struct macroblock *mb = &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;
  
  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }
  
  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;
  
  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

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

      if (tid == 0) {
        if (sad < best_sad)
        {
          mb->mv_x = x - mx;
          mb->mv_y = y - my;
          best_sad = sad;
        }
      }
    }
  }
  
  if (tid == 0) {
    mb->use_mv = 1;
  }
}

__device__ void c63_motion_estimate_gpu(struct c63_common *cm) {
  // /* Compare this frame with previous reconstructed frame */
  int color_component = gridDim.z;
  int mb_x = gridDim.x;
  int mb_y = gridDim.y;

  if (color_component == 0) {
    // Y component
    me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y, cm->refframe->recons->Y, Y_COMPONENT);
  } else if(color_component == 1) {
    // U component
    if (mb_x < cm->uv_mb_cols && mb_y < cm->uv_mb_rows) {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U, cm->refframe->recons->U, U_COMPONENT); 
    }
  } else {
    // V component
    if (mb_x < cm->uv_mb_cols && mb_y < cm->uv_mb_rows) {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V, cm->refframe->recons->V, V_COMPONENT); 
    }
  }
}

/* Motion compensation for 8x8 block */
__device__ static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
  uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb = &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) return;
  
  int w = cm->padw[color_component];

  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  
  int y = (mb_y * 8) + (tid >> 3);
  int x = (mb_x * 8) + (tid & 0x7);
  
  /* Bank conflicts? Bank conflicts may not be a problem for global memory */
  predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
}

__device__ void c63_motion_compensate_gpu(struct c63_common *cm) {
  int color_component = gridDim.z;
  int mb_x = gridDim.x;
  int mb_y = gridDim.y;

  __syncthreads();

  if (color_component == 0) {
    // Y component
    mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, Y_COMPONENT);
  } else if(color_component == 1) {
    // U component
    if (mb_x < cm->uv_mb_cols && mb_y < cm->uv_mb_rows) {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, U_COMPONENT);
    }
  } else {
    // V component
    if (mb_x < cm->uv_mb_cols && mb_y < cm->uv_mb_rows) {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

__global__ void c63_estimate_compensate_gpu(struct c63_common *cm) {
  c63_motion_estimate_gpu(cm);
  c63_motion_compensate_gpu(cm);
}

void c63_estimate_compensate(struct c63_common *cm) {
  /* Copy into device curframe original buffer */
  cudaMemcpy(
    d_curframe_origbuf, 
    cm->curframe->orig->buf,
    cm->total_yuv_buflen,
    cudaMemcpyHostToDevice
  );

  /* Copy into device refframe recons buffer */
  cudaMemcpy(
    d_refframe_reconsbuf, 
    cm->refframe->recons->buf,
    cm->total_yuv_buflen,
    cudaMemcpyHostToDevice
  );


  dim3 grid(cm->mb_cols, cm->mb_rows, 3);
  dim3 blk(32, 2, 1);

  c63_estimate_compensate <<< grid, blk >>>(d_cm);

  cudaDeviceSynchronize();

  /* Copy back curframe predicted buffer to the host */
  cudaMemcpy(
    cm->curframe->predicted->buf,
    d_curframe_predictedbuf,
    cm->total_yuv_buflen,  
    cudaMemcpyDeviceToHost
  );

  /* Copy back current frame macroblocks (Y, U and V) */
  cudaMemcpy(
    cm->curframe->mbs[Y_COMPONENT],
    d_curframe_mby,
    cm->y_mb_buflen, 
    cudaMemcpyDeviceToHost
  );
  
  cudaMemcpy(
    cm->curframe->mbs[U_COMPONENT],
    d_curframe_mbu, 
    cm->u_mb_buflen, 
    cudaMemcpyDeviceToHost
  );
  
  cudaMemcpy(
    cm->curframe->mbs[V_COMPONENT],
    d_curframe_mbv,
    cm->v_mb_buflen, 
    cudaMemcpyDeviceToHost
  );
}