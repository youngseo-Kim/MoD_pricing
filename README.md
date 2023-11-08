# MoD_pricing

This is the implementation for this paper (to be updated).

---
## prerequisite
Gurobi

## Guideline

I) SiouxFalls.py is to solve the convex programming in SiouxFalls network. Change ```file_name``` to save optimal solutions in text file. Use the following command to save log as a text file. 
```
python SiouxFalls.py > log_sampled_10.txt
```

II) analyze_result.ipynb is to analyze and visualize the output results.