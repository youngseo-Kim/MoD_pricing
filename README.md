# MoD_pricing

This repository provides the Python implementation to characterize the company-traveler equilibrium defined in the [following paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4950974). 

Strategic Pricing and Routing to Maximize Profit in Congested Roads Considering Interactions with Travelers
Youngseo Kim, Ning Duan, Gioele Zardini, Samitha Samaranayake, and Damon Wischik, Transactions on Control of Network Systems, 2024.

---
## Prerequisite
Gurobi: You can get free academic licence in the [following link](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Project Structure

- `data/`: JSON, CSV, and text files are used to represent the transportation network. We utilize the Sioux Falls network obtained from [this source](https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls).
- `src/`: Main implementation to build and solve the model. Follow the guidelines for more details. 
- `log/`: Saving logging file for intermediate solutions of MILP. 
- `model/`: Saving optimization models as mps files. It can be used to tune the parameters. 
- `output/`: Saving output solutions after solving an optimization problem. 

## Guideline

Before you start, download Gurobi license and create empty directories `log/`, `model/`, `output/`. Move to the `src/` directory. 

I) Run single script SiouxFalls.py
`src/SiouxFalls.py` is the main implementation for solving the proposed formulation using Gurobi with a piecewise linear approximation.

```
python SiouxFalls.py --n_ods 50 --dist 1 --alpha 0.15 --beta 4 --n_alter 1 --transit_scenario 2 --n_bins 15 --oper_cost 0.08 --exo_private 0 --mip_gap 0.05 --time_limit 18000
```

II) Run `src/runsim.sh` to repeat the experiments. 

```
sh runsim.sh 
```

III) Visualize results
`src/analyze_result.ipynb` is to analyze and visualize the output results. 



## Contact
If you have any questions or suggestions about the code, please contact us at the following email: yk796@cornell.edu
