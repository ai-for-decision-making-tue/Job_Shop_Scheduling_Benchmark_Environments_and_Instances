## Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods

### 📖 Overview:
This repository provides a comprehensive benchmarking environment for a variety of machine scheduling problems, including Job Shop Scheduling (JSP), Flow Shop Scheduling (FSP), Flexible Job Shop Scheduling (FJSP), FJSP with Assembly constraints (FAJSP), FJSP with Sequence-Dependent Setup Times (FJSP-SDST), and the online FJSP (with online job arrivals). It aims to be a centralized hub for researchers, practitioners, and enthusiasts interested in tackling machine scheduling challenges.

### 🛠 Solution Methods:
The repository includes exact, heuristic and learning based solution methods, each compatible with one or more machine scheduline problem variants:

| Solution methods | Type | JSP | FSP | FJSP | FJSP SDST | FAJSP | Online (F)JSP |
| :----: | :---:| :---:| :---: | :---: | :---: | :---: | :---: |
| MILP | Exact | ✓ | ✓ | ✓ | ✓ | | | 
| CP-SAT | Exact | ✓ | ✓ | ✓ | ✓ | | |
| Dispatching Rules | Heuristic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Genetic Algorithm | Heuristic |✓ | ✓ | ✓ | ✓ | ✓ | |
| FJSP-DRL | DRL | ✓ | ✓ | ✓ | |  | |
| L2D | DRL |✓ | ✓ | | | | |
| DANIEL | DRL | ✓ | ✓ | ✓ | | | |  

### 🚀 How to use:

Here we provide some short examples on how to use the solution methods in this repository. For more detailed information and more examples, please refer to the tutorials [here][2] and [here][3].

1. **Dispatching Rules:** 
  ```python
  from solution_methods.dispatching_rules import run_dispatching_rules
  from solution_methods.helper_functions import load_job_shop_env, load_parameters
  
  parameters = load_parameters("configs/dispatching_rules.toml")
  jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
  
  makespan, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)
  ```

2. **Genetic Algorithm:**  
  ```python
  from solution_methods.helper_functions import load_job_shop_env, load_parameters
  from solution_methods.GA.run_GA import run_GA
  from solution_methods.GA.src.initialization import initialize_run
  
  parameters = load_parameters("configs/genetic_algorithm.toml")
  jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))

  population, toolbox, stats, hof = initialize_run(jobShopEnv, **parameters)
  makespan, jobShopEnv = run_GA(jobShopEnv, population, toolbox, stats, hof, **parameters)  
```

3. **L2D (DRL-based):**
  ```python
  from solution_methods.L2D.src.run_L2D import run_L2D
  from solution_methods.helper_functions import load_job_shop_env, load_parameters
   
  parameters = load_parameters("configs/L2D.toml")
  jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
  makespan, jobShopEnv = run_L2D(jobShopEnv, **parameters)
  ```
  

### 🏗️ Repository Structure
The repository is structured to provide ease of use and flexibility:
- **Configs**: Contains the configuration files for the solution methods.
- **Data**: Contains the problem instances for benchmarking for different problem variants.
- **Data Parsers**: Parsers for configuring the benchmarking instances in the scheduling environment.
- **Plotting**: Contains the plotting functions for visualizing the results.
- **Scheduling Environment**: Defines the core environment components (`job`, `operation`, `machine`, and `jobShop`). Also contains the `simulationEnv` for dynamic scheduling problems with online job arrivals.
- **Solution Methods**: Contains the solution methods, including exact, heuristic, and learning-based approaches.


### 📄 Reference
For more detailed information, please refer to our paper. If you use this repository in your research, please consider citing the following paper:

> Reijnen, R., van Straaten, K., Bukhsh, Z., & Zhang, Y. (2023). 
> Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods. 
> arXiv preprint arXiv:2308.12794.
> https://doi.org/10.48550/arXiv.2308.12794

Or, using the following BibTeX entry:
```bibtex
@article{reijnen2023job,
  title={Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods},
  author={Reijnen, Robbert and van Straaten, Kjell and Bukhsh, Zaharah and Zhang, Yingqian},
  journal={arXiv preprint arXiv:2308.12794},
  year={2023}
}
```
A preprint of this paper is available or [arXiv][1]. Please note that this version is a placeholder, and will be updated shortely with the final version.

[1]: https://arxiv.org/abs/2308.12794
[2]: https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances/blob/main/tutorial_benchmark_environment.ipynb
[3]: https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances/blob/main/tutorial_custom_problem_instance.ipynb