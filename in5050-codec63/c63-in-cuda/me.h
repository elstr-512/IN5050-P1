#ifndef C63_ME_H_
#define C63_ME_H_

#include "c63.h"

// Declaration


void c63_motion_estimate(struct c63_common *cm);
__global__ void c63_motion_estimate_gpu(struct c63_common *cm);

void c63_motion_compensate(struct c63_common *cm);

#endif  /* C63_ME_H_ */
