

#prepare input files
ln -s ../../../4.frames/pdbmovie_1.pdb 

#have a copy of the .ini files in the folder before starting

#to calculate acPSN 
pyinteraph -s pdbmovie_1.pdb -t ../../../3.filt_trjs/traj_prot_dt1000.xtc -r pdbmovie_1.pdb -a --acpsn-co 4.5  --acpsn-imin 3 --ff-masses charmm27 --acpsn-proxco 2  --acpsn-nf-file normalization_factors.ini --acpsn-graph acpsn-graph.dat

