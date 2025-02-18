#!/usr/bin/bash

cmake -DCMAKE_CUDA_FLAGS="-arch=sm_50 -gencode=arch=compute_50,code=sm_50" ..