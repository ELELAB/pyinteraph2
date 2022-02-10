source /usr/local/gromacs-5.1.2_plumed-2.3b/bin/GMXRC.bash
modelID=$1
variant=$2
xtc=center_traj.xtc
simdir=../../../../../../simulations/$2/$1/CHARMM22star/md/Mol_An
#this cannot be reproduced in the tutorial since it would require the full length trajectory (too big for github to handle)
#we can provide the full trajectory upon request if the users are interested
#xtc skipped on protein dt 1000
gmx_mpi trjcat -f $simdir/$xtc -n ../1.index_files/protein.ndx -o traj_prot_dt1000.xtc -dt 1000
