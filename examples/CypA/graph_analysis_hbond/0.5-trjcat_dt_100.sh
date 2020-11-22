trajectory_path="../../../../../../../../../../../simulations/cypA/free/3K0Na_2-164/replicate1/CHARMM22star/md/Mol_An/center_traj.xtc"
reference_path="../../model0.pdb"

# Source gromacs
source /usr/local/gromacs-5.1.5/bin/GMXRC

# Take every hundredth frame
gmx trjcat -f $trajectory_path -dt 100 -o center_traj_dt_100.xtc

