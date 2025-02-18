#!/usr/bin/bash
if test -f foremanout.c63; then
	./c63dec foremanout.c63 output.yuv ; mplayer -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
fi
