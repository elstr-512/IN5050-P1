from what nsight systems is reporting, the final thing we could possibly do is use some sort of pinned memory.
In order to test this we could just use one frame of the video over and over again in order to remove I/O latency.and then there is the streaming-cores, we spend a lot of the time waiting on the transfer of frames from the cpu-gpu and gpu-cpu, we should be able to have three frames in a buffer that get fed into the streaming processors.
nsight-systems: are we using the same stream for every image?
