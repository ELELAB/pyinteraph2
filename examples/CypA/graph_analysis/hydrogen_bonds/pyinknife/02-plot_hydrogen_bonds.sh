# Pyinknife2 pipeline step 3: Plot

pyinknife_plot -c only_hydrogen_bonds_10_resamplings.yaml -cp plot_hubs_barplot.yaml -d aggregated_results_hydrogen_bonds -p hubs -od hubs_plot_hydrogen_bonds

pyinknife_plot -c only_hydrogen_bonds_10_resamplings.yaml -cp plot_ccs_barplot.yaml -d aggregated_results_hydrogen_bonds -p ccs -od cc_plot_hydrogen_bonds

