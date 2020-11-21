
# Final step in the Pyinteraph pipeline: Analysis of the final PSN

reference_path="model0.pdb"

# Run graph_analysis
graph_analysis -r $reference_path -a hc_graph_filtered.dat -c -cb cypa_components.pdb  
graph_analysis -r $reference_path -a hc_graph_filtered.dat -u -ub cypa_hubs.pdb 

# Calculate paths of communication from S99/T99 to the catalytic site, which is at R55.
graph_analysis -r $reference_path -a hc_graph_filtered.dat -p -r1 SYSTEM-99SER -r2 SYSTEM-55ARG > communication_paths.txt 

