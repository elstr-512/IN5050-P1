# IN5050-P1

## How to test out implementation

There are four contained scripts within the src/build directory

1. script_clean.sh - wipes cmake files from the build directory
2. script_fix_cmake.sh - runs cmake with patch for running on IFI machines 
3. script_encode.sh - runs the encoder
4. script_playback.sh - plays the video


## Build
To build:

cd src/build
cmake ..
make

To encode a video:

./c63enc -w 352 -h 288 -f <frames max 300> -o out-file.c63 <in-file>.yuv


To decode the c63 file:

./c63dec out-file.c63 output.yuv

Tip! Use mplayer or ffplay to playback raw YUV file:

Playback
Foreman
mplayer -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
Tractor
mplayer -demuxer rawvideo -rawvideo w=1920:h=1080 output.yuv

Dump the prediction buffer
./c63pred out-file.c63 output.yuv


Description
This project is used in IN5050 (Programming Heterogeneous Multi-core Architectures) at the Department of Informatics, University of Oslo, Norway. For more information, see the course page.
