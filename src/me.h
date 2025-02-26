#ifndef C63_ME_H_
#define C63_ME_H_

#include "c63.h"

#ifdef CUDA_OPTIMIZATION
void c63_estimate_compensate(struct c63_common *cm);
#else
void c63_motion_compensate(struct c63_common *cm);
#endif

#endif  /* C63_ME_H_ */
