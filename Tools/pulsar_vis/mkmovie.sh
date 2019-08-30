#!/usr/bin/bash

parallel --keep-order --tag "python3 pulsar_plot.py {} $@" ::: `ls -d plt?????`

echo "Finished plotting"

parallel --keep-order --tag "python3 ffmpeg_make_mp4 plt*{}.png -s -ifps 10 -ofps 30 -name {} > /dev/null 2>&1" ::: `cat plot_types.txt`

echo "Finished making movies"
