#prepare input files
ln -s ../../../4.frames/pdbmovie_1.pdb 

#have a copy of the .ini files in the folder before starting

#to calculate salt bridges
pyinteraph -s pdbmovie_1.pdb -t ../../../3.filt_trjs/traj_prot_dt1000.xtc -r pdbmovie_1.pdb --sb-co 5 -b --sb-graph sb-graph.dat --ff-masses charmm27 -v --sb-cg-file charged_groups.ini

#to calculate hbonds sc-sc

pyinteraph -s pdbmovie_1.pdb -t ../../../3.filt_trjs/traj_prot_dt1000.xtc -r pdbmovie_1.pdb -f --hb-graph hb-graph.dat --ff-masses charmm27 -v --hb-ad-file hydrogen_bonds.ini


#to calculate hydrophobic interactions

pyinteraph -s pdbmovie_1.pdb -t ../../../3.filt_trjs/traj_prot_dt1000.xtc -r pdbmovie_1.pdb --hc-co 5 -f --hc-graph hb-graph.dat --ff-masses charmm27 -v --hc-residues ALA,VAL,LEU,ILE,PHE,PRO,MET,TRP
