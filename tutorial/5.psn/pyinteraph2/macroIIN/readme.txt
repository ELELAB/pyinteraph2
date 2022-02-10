
#prepare input files
#ln -s ../../../4.frames/pdbmovie_1.pdb 

cp ../interactions/hb-graph_all.dat .
cp ../interactions/hc-graph_all.dat .
cp ../interactions/sb-graph_all.dat .

#filter the graphs individually
filter_graph -d hb-graph_all.dat -o hb-graph_filtered.dat -t 20.0
filter_graph -d hc-graph_all.dat -o hc-graph_filtered.dat -t 20.0
filter_graph -d sb-graph_all.dat -o sb-graph_filtered.dat -t 20.0

#generate a macroIIN (unweigthed)
filter_graph -d hb-graph_filtered.dat -d hc-graph_filtered.dat -d sb-graph_filtered.dat -o macro_IIN.dat

#generate macroIIN (weighted)
cp ../energy_net/kbp-graph.dat .
filter_graph -d hb-graph_filtered.dat -d hc-graph_filtered.dat -d sb-graph_filtered.dat -o macro_IIN_w.dat -w kbp-graph.dat
