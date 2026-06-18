# Requirements
# Topology file (e.g. sim.prot.A.pdb)
# Trajectory file (e.g. traj.xtc)

# Load environment
source /usr/local/envs/pyinteraph/bin/activate

# Checking different options
motion_correlation -h

# Running DCCM and naming the outputs dccm_example
motion_correlation -s sim.prot.A.pdb -t traj.xtc -m dccm -o dccm_example.dat -c dccm_example.csv

# Running LMI and naming the outpus lmi_example
motion_correlation -s sim.prot.A.pdb -t traj.xtc -m lmi -o lmi_example.dat -c lmi_example.csv

