# Some vis scripts for the pulsar setup

Dependencies: yt, Python 3, GNU Parallel

A good way to use these scripts is to make symbolic links to them in the directory containing the plotfiles.

## Plotting slices for individual plotfiles

To plot particles shaded by a particle variable in a slice, use the `pulsar_plot.py` script.

Use the `-h` argument to see the help options. For example, the following will plot electrons with an x-y slice shaded by Ez.

```
python3 pulsar_plot.py plt00250 -s e -c x y -v Ez
```

Any arguments not supplied will take all possible values in a loop.

For example, the following will plot all components of momentum, E, and B for both electrons and positrons in a y-z slice:

```
python3 pulsar_plot.py plt00250 -c y z
```

## Making movies

To make a movie, pass the arguments to the `pulsar_plot.py` script to the `mkmovie.sh` script, which will loop over the plotfiles.

This will create the slice plots necessary for the movies. If you already have the slice plots, you can use the ffmpeg command from the `mkmovie.sh` script.

The loop over plotfiles and the loop over plot types are both parallelized with GNU Parallel.

For example, the following will loop over plotfiles to make x-y slice plots with electrons shaded by Ez, and will then create the movie.

```
sh mkmovie.sh -s e -v Ez -c x y
```
