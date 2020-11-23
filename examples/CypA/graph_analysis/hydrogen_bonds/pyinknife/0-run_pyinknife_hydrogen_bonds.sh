# PyinKnife2 Pipeline
# First step: run the program 

reference_path="../../model0.pdb"

# Run
pyinknife_run -f "center_traj_dt_100.xtc" -s $reference_path -r $reference_path -c only_hydrogen_bonds_10_resamplings.yaml -n 4 -d raw_results_hydrogen_bonds
