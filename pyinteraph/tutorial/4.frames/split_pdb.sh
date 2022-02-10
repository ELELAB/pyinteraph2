modelID=$1
variant=$2
#this cannot be reproduced in the tutorial since it would require the full length trajectory (too big for github to handle)
#we can provide the full trajectory upon request if the users are interested
ln -s simdir=../../../../../../simulations/$2/$1/CHARMM22star/md/Mol_An/pdbmovie.pdb .
module load python/3.7
pdb_splitmodel pdbmovie.pdb
