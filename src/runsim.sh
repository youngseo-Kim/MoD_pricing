# #!/bin/bash

# scenario b
python SiouxFalls.py --alpha 0.15 --beta 4 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 10
python SiouxFalls.py --alpha 0.15 --beta 4 --mip_gap 0.01 --n_ods 50 --transit_scenario 2 --n_bins 10
python SiouxFalls.py --alpha 0.15 --beta 4 --mip_gap 0.01 --n_ods 100 --transit_scenario 2 --n_bins 10
python SiouxFalls.py --alpha 0.15 --beta 4 --mip_gap 0.01 --n_ods 528 --transit_scenario 2 --n_bins 10

# # scenario a
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 5 --transit_scenario 2 --n_bins 10
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 10
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 50 --transit_scenario 2 --n_bins 10
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 100 --transit_scenario 2 --n_bins 10
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 528 --transit_scenario 2 --n_bins 10

# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 5 --transit_scenario 2 --n_bins 100
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 100
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 50 --transit_scenario 2 --n_bins 100
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 100 --transit_scenario 2 --n_bins 100
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 528 --transit_scenario 2 --n_bins 100

# # scenario b
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 10
# python SiouxFalls.py --alpha 0.15 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 10
# python SiouxFalls.py --alpha 0.5 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 10

# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 10 --transit_scenario 1 --n_bins 10
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 10 --transit_scenario 2 --n_bins 10
# python SiouxFalls.py --alpha 0 --mip_gap 0.01 --n_ods 10 --transit_scenario 3 --n_bins 10

# #ideal
# python SiouxFalls.py --alpha 0.15 --mip_gap 0.01 --n_ods 528 --transit_scenario 2 --n_bins 100

