#!/bin/bash
for timestep in $(seq 10000 10000 20000); do
    newnumber='00000'${timestep}      # get number, pack with zeros
    newnumber=${newnumber:(-5)}       # the last five characters
    ./WritePlotfileToASCII2d.gnu.MPI.ex infile=diags/zline_BernadoFilter1221_${newnumber} | tee ${newnumber}.txt
done
