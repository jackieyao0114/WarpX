#!/bin/bash
for timestep in $(seq 5 5 100000); do
    newnumber='00000'${timestep}      # get number, pack with zeros
    newnumber=${newnumber:(-5)}       # the last five characters
    ./WritePlotfileToASCII3d.gnu.MPI.ex infile=20201221/diags/zline_BernadoFilter1221_${newnumber} | tee ${newnumber}.txt
done
