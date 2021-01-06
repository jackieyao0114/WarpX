#!/bin/bash
for timestep in $(seq 100000 2 300000); do
    newnumber='00000'${timestep}      # get number, pack with zeros
    newnumber=${newnumber:(-6)}       # the last five characters
    ./WritePlotfileToASCII3d.gnu.MPI.ex infile=diags/zline_BernadoFilter0105_${newnumber} | tee raw_data/${newnumber}.txt
done
