# Requirements
# Topology file (e.g. sim.prot.A.pdb)
# Trajectory file (e.g. traj.xtc)

# Load python
module load python

# Checking different options
python motion_correlation.py -h

# Running DCCM and naming the outputs dccm_example
python motion_correlation.py -s sim.prot.A.pdb -t traj.xtc -m dccm -o dccm_example.dat -c dccm_example.csv

# Running LMI and naming the outpus lmi_example
python motion_correlation.py -s sim.prot.A.pdb -t traj.xtc -m lmi -o lmi_example.dat -c lmi_example.csv

