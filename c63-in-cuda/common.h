#ifndef C63_COMMON_H_
#define C63_COMMON_H_

#include <inttypes.h>

#include "c63.h"

// Declarations
struct frame* create_frame(struct c63_common *cm, yuv_t *image);

void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

void cudaMemcpyErr(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
void cudaMallocErr(void** devPtr, size_t size);
void cudaFreeErr(void* devPtr);
void cudaDevSyncErr(void);

#endif  /* C63_COMMON_H_ */
