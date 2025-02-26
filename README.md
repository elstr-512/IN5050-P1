# IN5050-P1

This project is home exam one IN5050 (Programming Heterogeneous Multi-core Architectures) at the Department of Informatics, University of Oslo, Norway. For more information, see the course page.


NOTE: The CMakeList is tailored for the IN5050 Quadro K2200 machines.

## Build:

``` bash
cd src/build
```
```bash
cmake ..
```
```bash
make
```

## To encode a video:

Foreman 
```bash
./c63enc -w 352 -h 288 -f <frames max 300> -o out-file.c63 <in-file>.yuv
```

Tractor
```bash
./c63enc -w 1980 -h 1080 -f <frames max 300> -o out-file.c63 <in-file>.yuv
```

## To decode the c63 file:

```bash
./c63dec out-file.c63 output.yuv
```


## Playback the videos:
 
Foreman
```bash
mplayer -demuxer rawvideo -rawvideo w=352:h=288 output.yuv
```

Tractor
```bash
mplayer -demuxer rawvideo -rawvideo w=1920:h=1080 output.yuv
```

## Dump the prediction buffer:
```bash
./c63pred out-file.c63 output.yuv
```
