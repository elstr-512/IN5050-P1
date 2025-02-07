# Codec63 #

The repository contains code for both native C code that is meant for compilation with GCC,
and native CUDA code that is meant for compilation with NVCC. Choose the subdirectory 
c63-in-c for the first and c63-in-cuda for the second.

The code is actually the same at this point (C++ is pretty much a superset of C,
and CUDA a superset of C++), only the filenames are different for convenience.

After changing either to c63-in-c or to c63-in-cuda, you do the following:

To build:
```
mkdir build
cd build
cmake ..
make
```

To encode a video:
```
./c63enc -w 352 -h 288 -f 10 -o foremanout.c63 foreman.yuv
```

We strongly recommend that clone the github repository in5050-codec63 twice.
Make your change in one of them, but use the c53dec and c63pred command from your
_other_ checkout. The reason is that nearly all files are shared between encoding
and decoding.

To decode the c63 file:
```
./c63dec foremanout.c63 output.yuv
```

To playback the raw YUV file before encoding or after decoding, you can use mplayer or ffplay.
```
mplayer -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
ffplay -f rawvideo -pixel_format yuv420p -video_size 352x288 -i output.yuv
```
When you are remotely logged into a Tegra, you may have to force the use of X11:
```
mplayer -vo x11 -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
```


### Description ###
This project is used in IN5050 (Programming Heterogeneous Multi-core Architectures) at the Department of Informatics, University of Oslo, Norway. For more information, see the [course page](https://www.uio.no/studier/emner/matnat/ifi/IN5050/).
