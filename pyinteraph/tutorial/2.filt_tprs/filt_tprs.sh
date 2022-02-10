source /usr/local/gromacs-5.1.2_plumed-2.3b/bin/GMXRC.bash
modelID=$1
variant=$2
tpr=sim
simdir=../../../../../../simulations/$2/$1/CHARMM22star/md
#this cannot be reproduced in the tutorial since it would require the full length trajectory (too big for github to handle)
#we can provide the full trajectory upon request if the users are interested
#tpr filtered on  protein
#in reality this is not used by PyInKnife2, we left it for consistency with the contents of the full workflow we have in house
gmx_mpi convert-tpr -s $simdir/$tpr.tpr -n ../1.index_files/protein.ndx -o md_prot.tpr
