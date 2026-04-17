#Activate pyinteraph environment
source /usr/local/envs/pyinteraph/bin/activate

#Copy reference pdb file to working directory
cp /data/user/shared_projects/mavisp/GENE_NAME/simulations_analysis/free/RANGE/replicateX/CHARMM36/md/8.psn/pyinteraph2/acPSN/full_traj/reference_A.pdb .

#Copy trajectory file to working directory
cp /data/user/shared_projects/mavisp/GENE_NAME/simulations_analysis/free/RANGE/replicateX/CHARMM36/md/8.psn/pyinteraph2/acPSN/full_traj/traj_prot.xtc .

#Run the python program
python dccm_calc.py reference_A.pdb traj_prot.xtc

#Running the tests
pytest -v

# example usage
cp /data/user/shared_projects/mavisp/GRB2/simulations_analysis/free/AF_160-214/replicate1/CHARMM36/md/8.psn/pyinteraph2/acPSN/full_traj/reference_A.pdb .
cp /data/user/shared_projects/mavisp/GRB2/simulations_analysis/free/AF_160-214/replicate1/CHARMM36/md/8.psn/pyinteraph2/acPSN/full_traj/traj_prot.xtc .
python dccm_calc.py reference_A.pdb traj_prot.xtc

