
#prepare input files
ln -s ../../../4.frames/pdbmovie_1.pdb 

#have a copy of the .ini files in the folder before starting

#to calculate ccmPSN (with Rg correction)
pyinteraph -s pdbmovie_1.pdb -t ../../../3.filt_trjs/traj_prot_dt1000.xtc -r pdbmovie_1.pdb --cmpsn --cmpsn-co 2.5 --cmpsn-csv cmpsn.csv --cmpsn-graph cmpsn.dat --cmpsn-correction rg --ff-masses charmm27


