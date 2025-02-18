#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include "c63.h"
#include "c63_write.h"
#include "quantdct.h"
#include "common.h"
#include "me.h"
#include "tables.h"

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

/* Created pointer globals for VRAM memory */

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static yuv_t* read_yuv(FILE *file, struct c63_common *cm)
{
  size_t len = 0;
  yuv_t *image = (yuv_t*)malloc(sizeof(yuv_t));
  yuv_buf img_yuv_buf;

  /* Allocating space for Y, U and v pixel components */
  img_yuv_buf = (yuv_buf)calloc(cm->total_yuv_buflen, 1);

  /* Read Y, U and V. U and V components are sampled 4:2:0 meaning 1/4 size of Y */
  len += fread(img_yuv_buf               , 1, cm->y_datalen, file);
  len += fread(img_yuv_buf + cm->u_bufoff, 1, cm->u_datalen, file);
  len += fread(img_yuv_buf + cm->v_bufoff, 1, cm->v_datalen, file);

  if (ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if (feof(file))
  {
    free(img_yuv_buf);
    free(image);
    return NULL;
  }
  else if (len != width*height*1.5)
  {
    fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
    fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);
    
    free(img_yuv_buf);
    free(image);
    return NULL;
  }

  image->Y   = img_yuv_buf;
  image->U   = img_yuv_buf + cm->u_bufoff;
  image->V   = img_yuv_buf + cm->v_bufoff;
  image->buf = img_yuv_buf;

  return image;
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{  
  /* Advance to next frame */
  destroy_frame(cm->refframe);
  cm->refframe = cm->curframe;
  cm->curframe = create_frame(cm, image);

  /* Check if keyframe */
  if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
  {
    cm->curframe->keyframe = 1;
    cm->frames_since_keyframe = 0;

    fprintf(stderr, " (keyframe) ");
  }
  else { cm->curframe->keyframe = 0; }

  if (!cm->curframe->keyframe)
  {
    c63_estimate_compensate(cm);
  }

  /* DCT and Quantization */
  dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT],
      cm->padh[Y_COMPONENT], cm->curframe->residuals->Ydct,
      cm->quanttbl[Y_COMPONENT]);

  dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT],
      cm->padh[U_COMPONENT], cm->curframe->residuals->Udct,
      cm->quanttbl[U_COMPONENT]);

  dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT],
      cm->padh[V_COMPONENT], cm->curframe->residuals->Vdct,
      cm->quanttbl[V_COMPONENT]);

  /* Reconstruct frame for inter-prediction */
  dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y,
      cm->ypw, cm->yph, cm->curframe->recons->Y, cm->quanttbl[Y_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U,
      cm->upw, cm->uph, cm->curframe->recons->U, cm->quanttbl[U_COMPONENT]);
  dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V,
      cm->vpw, cm->vph, cm->curframe->recons->V, cm->quanttbl[V_COMPONENT]);

  /* Function dump_image(), found in common.c, can be used here to check if the
     prediction is correct */

  write_frame(cm);

  ++cm->framenum;
  ++cm->frames_since_keyframe;
}

struct c63_common* init_c63_enc(int width, int height)
{
  int i;

  /* calloc() sets allocated memory to zero */
  c63_common *cm = (c63_common*)calloc(1, sizeof(struct c63_common));

  cm->width = width;
  cm->height = height;

  cm->padw[Y_COMPONENT] = cm->ypw = (uint32_t)(ceil(width/16.0f)*16);
  cm->padh[Y_COMPONENT] = cm->yph = (uint32_t)(ceil(height/16.0f)*16);
  cm->padw[U_COMPONENT] = cm->upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  cm->padh[U_COMPONENT] = cm->uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  cm->padw[V_COMPONENT] = cm->vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  cm->padh[V_COMPONENT] = cm->vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  cm->y_buflen = cm->padw[Y_COMPONENT]*cm->padh[Y_COMPONENT];
  cm->u_buflen = cm->padw[U_COMPONENT]*cm->padh[U_COMPONENT];
  cm->v_buflen = cm->padw[V_COMPONENT]*cm->padh[V_COMPONENT];

  cm->total_yuv_buflen = cm->y_buflen + cm->u_buflen + cm->v_buflen;

  cm->u_bufoff = cm->y_buflen;
  cm->v_bufoff = cm->y_buflen + cm->u_buflen;

  cm->y_datalen = width * height;
  cm->u_datalen = cm->y_datalen / 4;
  cm->v_datalen = cm->u_datalen;

  cm->mb_cols = cm->ypw / 8;
  cm->mb_rows = cm->yph / 8;

  cm->uv_mb_cols = cm->mb_cols / 2;
  cm->uv_mb_rows = cm->mb_rows / 2;

  cm->y_mb_buflen = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
  cm->u_mb_buflen = cm->y_mb_buflen / 4;
  cm->v_mb_buflen = cm->u_mb_buflen;

  /* Quality parameters -- Home exam deliveries should have original values,
   i.e., quantization factor should be 25, search range should be 16, and the
   keyframe interval should be 100. */
  cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
  cm->me_search_range = 16;     // Pixels in every direction
  cm->keyframe_interval = 100;  // Distance between keyframes

  /* Initialize quantization tables */
  for (i = 0; i < 64; ++i)
  {
    cm->quanttbl[Y_COMPONENT][i] = yquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[U_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
    cm->quanttbl[V_COMPONENT][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
  }

  return cm;
}

void free_c63_enc(struct c63_common* cm)
{
  destroy_frame(cm->curframe);
  free(cm);
}

struct c63_common *d_cm;
struct frame *d_refframe, *d_curframe;

yuv_t *d_curframe_orig, *d_refframe_recons, // Computing from this (copied from host per frame)
  *d_curframe_predicted; // Computing into this and copying back
yuv_buf d_curframe_origbuf, d_refframe_reconsbuf,
  d_curframe_predictedbuf;

struct macroblock *d_curframe_mby, *d_curframe_mbu, *d_curframe_mbv;

void init_device_state(struct c63_common *cm) {
  /* Allocating necessary memory on the device */
  cudaMallocErr((void**)&d_cm, sizeof(struct c63_common));

  cudaMallocErr((void**)&d_refframe, sizeof(struct frame));
  cudaMallocErr((void**)&d_curframe, sizeof(struct frame));

  cudaMallocErr((void**)&d_curframe_orig, sizeof(struct yuv));
  cudaMallocErr((void**)&d_refframe_recons, sizeof(struct yuv));
  cudaMallocErr((void**)&d_curframe_predicted, sizeof(struct yuv));

  cudaMallocErr((void**)&d_curframe_origbuf, cm->total_yuv_buflen);
  cudaMallocErr((void**)&d_refframe_reconsbuf, cm->total_yuv_buflen);
  cudaMallocErr((void**)&d_curframe_predictedbuf, cm->total_yuv_buflen);

  cudaMallocErr((void**)&d_curframe_mby, cm->y_mb_buflen);
  cudaMallocErr((void**)&d_curframe_mbu, cm->u_mb_buflen);
  cudaMallocErr((void**)&d_curframe_mbv, cm->v_mb_buflen);

  /* Initializing device memory structures */
  cudaMemcpyErr(d_cm, cm, sizeof(struct c63_common), cudaMemcpyHostToDevice);

  cudaMemcpyErr((uint8_t*)d_cm + offsetof(struct c63_common, refframe), &d_refframe, sizeof(d_refframe), cudaMemcpyHostToDevice);
  cudaMemcpyErr((uint8_t*)d_cm + offsetof(struct c63_common, curframe), &d_curframe, sizeof(d_curframe), cudaMemcpyHostToDevice);

  cudaMemcpyErr((uint8_t*)d_curframe + offsetof(struct frame, orig), &d_curframe_orig, sizeof(d_curframe_orig), cudaMemcpyHostToDevice);
  cudaMemcpyErr((uint8_t*)d_refframe + offsetof(struct frame, recons), &d_refframe_recons, sizeof(d_refframe_recons), cudaMemcpyHostToDevice);
  cudaMemcpyErr((uint8_t*)d_curframe + offsetof(struct frame, predicted), &d_curframe_predicted, sizeof(d_curframe_predicted), cudaMemcpyHostToDevice);  

  struct yuv curframe_orig_yuv = {
    .Y   = (uint8_t*)d_curframe_origbuf,
    .U   = (uint8_t*)d_curframe_origbuf + cm->u_bufoff,
    .V   = (uint8_t*)d_curframe_origbuf + cm->v_bufoff,
    .buf = d_curframe_origbuf
  };

  struct yuv refframe_recons_yuv = {
    .Y   = (uint8_t*)d_refframe_reconsbuf,
    .U   = (uint8_t*)d_refframe_reconsbuf + cm->u_bufoff,
    .V   = (uint8_t*)d_refframe_reconsbuf + cm->v_bufoff,
    .buf = d_refframe_reconsbuf
  };
  
  struct yuv curframe_predicted_yuv = {
    .Y   = (uint8_t*)d_curframe_predictedbuf,
    .U   = (uint8_t*)d_curframe_predictedbuf + cm->u_bufoff,
    .V   = (uint8_t*)d_curframe_predictedbuf + cm->v_bufoff,
    .buf = d_curframe_predictedbuf
  };

  cudaMemcpyErr(d_curframe_orig, &curframe_orig_yuv, sizeof(struct yuv), cudaMemcpyHostToDevice);
  cudaMemcpyErr(d_refframe_recons, &refframe_recons_yuv, sizeof(struct yuv), cudaMemcpyHostToDevice);
  cudaMemcpyErr(d_curframe_predicted, &curframe_predicted_yuv, sizeof(struct yuv), cudaMemcpyHostToDevice);

  cudaMemcpyErr(
    (uint8_t*)d_curframe + offsetof(struct frame, mbs), 
    &d_curframe_mby, sizeof(d_curframe_mby), 
    cudaMemcpyHostToDevice
  );
  
  cudaMemcpyErr(
    (uint8_t*)d_curframe + offsetof(struct frame, mbs) + sizeof(struct macroblock*), 
    &d_curframe_mbu, sizeof(d_curframe_mbu), 
    cudaMemcpyHostToDevice
  );
  
  cudaMemcpyErr(
    (uint8_t*)d_curframe + offsetof(struct frame, mbs) + 2 * sizeof(struct macroblock*), 
    &d_curframe_mbv, sizeof(d_curframe_mbv), 
    cudaMemcpyHostToDevice
  );
}

void fini_device_state() {
  cudaFreeErr(d_cm);
  cudaFreeErr(d_refframe);
  cudaFreeErr(d_curframe);
  cudaFreeErr(d_curframe_orig);
  cudaFreeErr(d_refframe_recons);
  cudaFreeErr(d_curframe_predicted);
  cudaFreeErr(d_curframe_origbuf);
  cudaFreeErr(d_refframe_reconsbuf);
  cudaFreeErr(d_curframe_predictedbuf);
  cudaFreeErr(d_curframe_mby);
  cudaFreeErr(d_curframe_mbu);
  cudaFreeErr(d_curframe_mbv);
}

static void print_help()
{
  printf("Usage: ./c63enc [options] input_file\n");
  printf("Commandline options:\n");
  printf("  -h                             Height of images to compress\n");
  printf("  -w                             Width of images to compress\n");
  printf("  -o                             Output file (.c63)\n");
  printf("  [-f]                           Limit number of frames to encode\n");
  printf("\n");

  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;
  yuv_t *image;

  if (argc == 1) { print_help(); }

  while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
  {
    switch (c)
    {
      case 'h':
        height = atoi(optarg);
        break;
      case 'w':
        width = atoi(optarg);
        break;
      case 'o':
        output_file = optarg;
        break;
      case 'f':
        limit_numframes = atoi(optarg);
        break;
      default:
        print_help();
        break;
    }
  }

  if (optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  /* Opening file descriptor for input file */
  input_file = argv[optind];
  FILE *infile = fopen(input_file, "rb");
  if (infile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Opening file descriptor for output file */
  outfile = fopen(output_file, "wb");
  if (outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Prompting information about limiting the # of encoded frames */
  if (limit_numframes) { printf("Limited to %d frames.\n", limit_numframes); }

  /* Initializing information about the frames that will be encoded */
  struct c63_common *cm = init_c63_enc(width, height);
  cm->e_ctx.fp = outfile;

  /* Initialize device state */
  init_device_state(cm);

  /* Encode input frames */
  int numframes = 0;

  while (1)
  {
    image = read_yuv(infile, cm);

    if (!image) { break; }

    printf("Encoding frame %d, ", numframes);
    c63_encode_image(cm, image);

    free(image->buf);
    free(image);

    printf("Done!\n");

    ++numframes;

    if (limit_numframes && numframes >= limit_numframes) { break; }
  }

  fini_device_state();
  free_c63_enc(cm);
  fclose(outfile);
  fclose(infile);

  return EXIT_SUCCESS;
}
