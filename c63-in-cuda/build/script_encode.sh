#!/usr/bin/bash

IMAGE_FILE=foreman.yuv

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <frames to encode>"
    exit 1
fi

if [ ! -f $IMAGE_FILE ]; then
    echo "Can not find $IMAGE_FILE"
    exit 1
fi

if test -f Makefile; then
	make && ./c63enc -w 352 -h 288 -f $1 -o foremanout.c63 $IMAGE_FILE

else
    ./script_fix_cmake.sh
    make && ./c63enc -w 352 -h 288 -f $1 -o foremanout.c63 $IMAGE_FILE
fi
