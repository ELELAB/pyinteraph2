
#prepare input files
ln -s ../../../4.frames/pdbmovie_1.pdb 

#have a copy of the .ini files in the folder before starting

pyinteraph -s pdbmovie_1.pdb -t ../../../3.filt_trjs/traj_prot_dt1000.xtc -r pdbmovie_1.pdb -p --kbp-kbt 1 --ff-masses charmm27 -v --kbp-graph kbp-graph.dat

