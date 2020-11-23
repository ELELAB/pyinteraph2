# Pyinknife2 pipeline step 3: Plot

pyinknife_plot -c only_hydrophobic_10_resamplings.yaml -cp plot_hubs_barplot.yaml -d aggregated_results_hydrophobic -p hubs -od hubs_plot_hydrophobic

pyinknife_plot -c only_hydrophobic_10_resamplings.yaml -cp plot_ccs_barplot.yaml -d aggregated_results_hydrophobic  -p ccs -od cc_plot_hydrophobic

