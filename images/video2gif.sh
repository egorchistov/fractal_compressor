#!/usr/bin/env sh
# Usage: sh video2gif.sh input.mp4 output.gif
ffmpeg -i "$1" -vf "crop=550:550:640:40" /tmp/temp.mp4
ffmpeg -i /tmp/temp.mp4 -pix_fmt rgb24 -r 10 -s 280x280 /tmp/temp.gif
convert -layers Optimize /tmp/temp.gif "$2"
rm /tmp/temp.mp4 /tmp/temp.gif