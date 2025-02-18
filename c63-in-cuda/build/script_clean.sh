#!/usr/bin/bash
if test -f Makefile; then
	make clean && rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake foremanout.c63 Makefile output.yuv
fi
