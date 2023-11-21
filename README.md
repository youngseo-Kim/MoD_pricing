# MoD_pricing

This is the implementation for this paper (to be updated).

---
## prerequisite
Gurobi


## Project Structure

- `data/`: json, csv, text file to indicate transporation network. Currently, SiouxFall network is considered. 
- `log/`: save logging file for intermediate solutions of MILP. `src/plot_log.ipynb` uses these logs to plot the change of optimality gap over time. 
- `model/`: save optimization models as mps files. It can be used to tune the parameters. 
- `output/`: save output solutions after solving an optimization problem. `src/analyze_result.ipynb` uses this output to visualize prices, market share, vehicle flows, and road congestions. 
- `src/`: main implementation to build and solve model. Follow the Guideline for more details. 


## Guideline

I) Run single script SiouxFalls.py
`src/SiouxFalls.py` is the main implementation to solve the convex programming using Gurobi. 

```
python SiouxFalls.py --alpha 0 --beta 1 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 10
```

II) Run `src/runsim.sh` to repeat the experiments. 

III) Visualize resuylts
`src/analyze_result.ipynb` is to analyze and visualize the output results. `src/plot_log.ipynb` is to plot the change of optimality gap over time. 


## Tune the parameter

[Resource](https://www.gurobi.com/documentation/current/refman/parameter_tuning_tool.html)
```
grbtune NonConvex=2 TuneTimeLimit=30000 output/model_name.mps
```

