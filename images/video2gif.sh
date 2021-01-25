#!/usr/bin/env sh
# Usage: sh video2gif.sh input.mp4 output.gif
#
# Filters:
# Slowdown 8 times: setpts=8.0*PTS
# Crop: crop=out_w:out_h:x:y
# Scale: scale=-1:240
ffmpeg -i "$1" -pix_fmt rgb8 -r 10 -filter:v "crop=640:240:0:120, scale=-1:240, setpts=8.0*PTS" /tmp/temp.gif
convert -layers Optimize /tmp/temp.gif "$2"
rm /tmp/temp.gif
