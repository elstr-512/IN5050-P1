# IN5050-P1

## How to test out implementation

There are four contained scripts within the src/build directory

1. script_clean.sh - wipes cmake files from the build directory
2. script_fix_cmake.sh - runs cmake with patch for running on IFI machines 
3. script_encode.sh - runs the encoder
4. script_playback.sh - plays the video


## Build
To build:

``` bash
cd src/build
```
```bash
cmake ..
```
```bash
make
```

To encode a video:

Foreman 
```bash
./c63enc -w 352 -h 288 -f <frames max 300> -o out-file.c63 <in-file>.yuv
```

Tractor
```bash
./c63enc -w 1980 -h 1080 -f <frames max 300> -o out-file.c63 <in-file>.yuv
```

To decode the c63 file:

```bash
./c63dec out-file.c63 output.yuv
```


Playback
Tip! Use mplayer or ffplay to playback raw YUV file:
 
Foreman
```bash
mplayer -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
```

Tractor
```bash
mplayer -demuxer rawvideo -rawvideo w=1920:h=1080 output.yuv
```

Dump the prediction buffer
```bash
./c63pred out-file.c63 output.yuv
```


Description
This project is used in IN5050 (Programming Heterogeneous Multi-core Architectures) at the Department of Informatics, University of Oslo, Norway. For more information, see the course page.
