#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

void destroy_frame(struct frame *f)
{
  /* First frame doesn't have a reconstructed frame to destroy */
  if (!f) { return; }

  free(f->recons->buf);
  free(f->recons);

  free(f->residuals->Ydct);
  free(f->residuals->Udct);
  free(f->residuals->Vdct);
  free(f->residuals);

  free(f->predicted->buf);
  free(f->predicted);

  free(f->mbs[Y_COMPONENT]);
  free(f->mbs[U_COMPONENT]);
  free(f->mbs[V_COMPONENT]);

  free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
  frame *f = (frame*)malloc(sizeof(struct frame));

  f->orig = image;

  f->recons      = (yuv_t*)malloc(sizeof(yuv_t));
  f->recons->buf = (yuv_buf)calloc(cm->total_yuv_buflen, 1);
  f->recons->Y   = f->recons->buf;
  f->recons->U   = f->recons->buf + cm->u_bufoff;
  f->recons->V   = f->recons->buf + cm->v_bufoff;

  f->predicted      = (yuv_t*)malloc(sizeof(yuv_t));
  f->predicted->buf = (yuv_buf)calloc(cm->total_yuv_buflen, 1);
  f->predicted->Y   = f->predicted->buf;
  f->predicted->U   = f->predicted->buf + cm->u_bufoff;
  f->predicted->V   = f->predicted->buf + cm->v_bufoff;

  f->residuals = (dct_t*)malloc(sizeof(dct_t));
  f->residuals->Ydct = (int16_t*)calloc(cm->ypw * cm->yph, sizeof(int16_t));
  f->residuals->Udct = (int16_t*)calloc(cm->upw * cm->uph, sizeof(int16_t));
  f->residuals->Vdct = (int16_t*)calloc(cm->vpw * cm->vph, sizeof(int16_t));

  f->mbs[Y_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows * cm->mb_cols, sizeof(struct macroblock));
  f->mbs[U_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));
  f->mbs[V_COMPONENT] =
    (macroblock*)calloc(cm->mb_rows/2 * cm->mb_cols/2, sizeof(struct macroblock));

  return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
  fwrite(image->Y, 1, w*h, fp);
  fwrite(image->U, 1, w*h/4, fp);
  fwrite(image->V, 1, w*h/4, fp);
}

static void printCudaErrReadable(cudaError_t err) {
  switch(err) {
    case cudaErrorInvalidValue:
      fprintf(stderr, "Cuda error invalid value\n");
      break;
    case cudaErrorMemoryAllocation:
      fprintf(stderr, "Cuda error memory allocation\n");
      break;
    case cudaErrorInitializationError:
      fprintf(stderr, "Cuda initialization error\n");
      break;
    default:
      fprintf(stderr, "Uncategorized cuda error (%d)\n", err);
  }
}

void cudaMemcpyErr(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  cudaError_t err = cudaMemcpy(dst, src, count, kind);

  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy: ");
    printCudaErrReadable(err);
    exit(EXIT_FAILURE);
  }
}

void cudaMallocErr(void** devPtr, size_t size) {
  cudaError_t err = cudaMalloc(devPtr, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc: ");
    printCudaErrReadable(err);
    exit(EXIT_FAILURE);
  }
}

void cudaFreeErr(void* devPtr) {
  cudaError_t err = cudaFree(devPtr);

  if (err != cudaSuccess) {
    fprintf(stderr, "cudaFree: ");
    printCudaErrReadable(err);
    exit(EXIT_FAILURE);
  }
}

void cudaDevSyncErr() {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize: ");
    printCudaErrReadable(err);
    exit(EXIT_FAILURE);
  }
}