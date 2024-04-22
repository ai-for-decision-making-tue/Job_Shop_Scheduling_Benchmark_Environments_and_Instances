# Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods 
Welcome to the **Job Shop Scheduling Benchmark**

This GitHub repository serves as a comprehensive benchmark for a wide range of machine scheduling problems, including  Job Shop Scheduling (JSP), Flow Shop Scheduling (FSP), Flexible Job Shop Scheduling (FJSP), FJSP with Assembly constraints (FAJSP), FJSP with Sequence-Dependent Setup Times (FJSP-SDST), and the online FJSP (with online job arrivals). Our primary goal is to provide a centralized hub for researchers, practitioners, and enthusiasts interested in tackling machine scheduling challenges. 



## 🛠 Solution methods
We aim to include a wide range of solution methods capable of solving machine scheduling problems with various constraints and characteristics. This selection ranges from **load-balancing heuristics**, **dispatching rules** and **genetic algorithms** to end-to-end **Deep Reinforcement Learning** solutions. The repo currently contains the following solution methods, each capable of solving machine scheduling problems with the corresponding characteristics:  



| Solution methods | Job Shop (JSP) | Flow Show (FSP) | Flexible Job Shop (FJSP) | FJSP SDST | FAJSP | Online (F)JSP |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: |
| Dispatching Rules | ✓ | ✓ | ✓ | | ✓ | ✓* |
| Genetic Algorithm | ✓ | ✓ | ✓ | ✓ | ✓ | |
| MILP | ✓ | ✓ | ✓ | ✓ | | | 
| CP-SAT | ✓ | ✓ | ✓ | ✓ | | |
| FJSP-DRL | ✓ | ✓ | ✓ | |  | |
| L2D | ✓ | ✓ | | | | |
| DANIEL | ✓ | ✓ | ✓ | | | |  

*Capable of online arrivals of FJSP problems 

🔜 We have a few DRL-based solutions in the pipeline, which will be published here upon completion. 

📢 We encourage you to make use of our repository to get started with your own solutions, and, when possible, release your solution method in this repository.

## 📝 Cite our repository:
Please consider citing our paper if you use code or ideas from this project:

Robbert Reijnen, Kjell van Straaten, Zaharah Bukhsh, and Yingqian Zhang (2023) [Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods](https://arxiv.org/abs/2308.12794). arXiv preprint arXiv:2308.12794 
```
@misc{reijnen2023job,
      title={Job Shop Scheduling Benchmark: Environments and Instances for Learning and Non-learning Methods}, 
      author={Robbert Reijnen and Kjell van Straaten and Zaharah Bukhsh and Yingqian Zhang},
      year={2023},
      eprint={2308.12794},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
