#prepare input files
ln -s ../../../4.frames/pdbmovie_1.pdb 

#have a copy of the .ini files in the folder before starting

#to calculate cmPSN (no correction)
pyinteraph -s pdbmovie_1.pdb -t ../../../3.filt_trjs/traj_prot_dt1000.xtc -r pdbmovie_1.pdb --cmpsn --cmpsn-co 5 --cmpsn-csv cmpsn.csv --cmpsn-graph cmpsn.dat --cmpsn-correction null --ff-masses charmm27  

#to estimate the cutoff for occurrence percentage

filter_graph -d cmpsn_all.dat -c cluster_size.dat -p clusters_plot.pdf

#filter the grap using cutoff of occurrence perc. of 20%

filter_graph -d cmpsn_all.dat -o cmpsn_all_filtered.dat -t 20.0

#calculate connected components
graph_analysis -a cmpsn_all_filtered.dat -r pdbmovie_1.pdb -c -cb ccs.pdb

#calculate hubs
graph_analysis -a cmpsn_all_filtered.dat -r pdbmovie_1.pdb -u -ub hubs.pdb -k 3

#metapath
path_analysis -i cmpsn_all_filtered.dat -r pdbmovie_1.pdb -m -g 3 -e 0.1 -n 0.1 -d metapath_hb

#centrality
centrality_analysis -i cmpsn_all_filtered.dat -r pdbmovie_1.pdb -c all -o centrality.csv

#cytoscape-compatible output
dat2graphml -a cmpsn_all_filtered.dat -r pdbmovie_1.pdb -o  graph.graphml


