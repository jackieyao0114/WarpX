import yt ; yt.funcs.mylog.setLevel(50)
import numpy as np
import scipy.constants as scc
import matplotlib.pyplot as plt
import argparse
import os
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="Name of input plotfile to process.")
parser.add_argument("-s", "--species", type=str, default='both', help="Species to plot (e.g. 'e', 'p', or the default 'both')")
parser.add_argument("-v", "--variable", type=str, default='all', help="Particle variable to plot (e.g. 'momentum_x', 'Ex', 'Bx', ...). Defaults to 'all'")
parser.add_argument("-c", "--coordinates", type=str, nargs=2, default=['c', 'c'], help="Coordinate axes to plot against. Takes 2 values out of 'x', 'y', and 'z'. Defaults to all combinations.")

args = parser.parse_args()

suffixes = []

def doplot(filename):
    basename = os.path.basename(filename)

    ds = yt.load(filename)

    if args.species == 'both':
        species = ['e', 'p']
    else:
        species = [args.species]

    if args.coordinates[0] == 'c' or args.coordinates[1] == 'c':
        coordinates = itertools.combinations(['x', 'y', 'z'], 2)
    else:
        coordinates = [args.coordinates]

    if args.variable == 'all': 
        variables = ['momentum_x', 'momentum_y', 'momentum_z', 'Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz']
    else:
        variables = [args.variable]

    combinations = itertools.product(species, coordinates, variables)

    for s, c, v in combinations:
        species_name = 'plasma_{}'.format(s)
        c1, c2 = c
        position_name_1 = 'particle_position_{}'.format(c1)
        position_name_2 = 'particle_position_{}'.format(c2)
        variable_name = 'particle_{}'.format(v)
        p = yt.ParticlePlot(ds,(species_name, position_name_1),
                (species_name, position_name_2),
                (species_name, variable_name))
        suffix = "{}{}_{}_{}".format(c1, c2, s, variable_name)
        suffixes.append(suffix)
        p.save("{}_{}.png".format(basename, suffix))

if __name__ == "__main__":
    fname = args.infile
    doplot(fname)

    fsuff = open("plot_types.txt", "w")
    for s in suffixes:
        fsuff.write("{}\n".format(s))
    fsuff.close()
