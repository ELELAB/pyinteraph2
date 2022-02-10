source /usr/local/gromacs-5.1.2_plumed-2.3b/bin/GMXRC.bash
modelID=$1
variant=$2
tpr=sim
#this cannot be reproduced in the tutorial since it would require the full length trajectory (too big for github to handle)
#we can provide the full trajectory upon request if the users are interested
simdir=../../../../../../simulations/$2/$1/CHARMM22star/md
#create the protein ndx file
gmx_mpi make_ndx -f $simdir/$tpr.tpr -o protein.ndx << eof
keep 1 
q
eof
