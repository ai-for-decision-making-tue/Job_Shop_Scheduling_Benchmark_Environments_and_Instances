PRINTS = False
OUTPUT = False

import os
import scheduling_environment.simulationEnv as sim
from data_parsers.custom_instance_parser import parse
from plotting.drawer import plot_gantt_chart
from solution_methods.GA.src.initialization import initialize_run
from solution_methods.GA.run_GA import run_GA
from solution_methods.CP_SAT.run_cp_sat import run_CP_SAT
import numpy as np


os.chdir("/home/cole/madrid/sites")
import frappe

frappe.connect("test", db_name="cole")


class Workstations:
    def __init__(self) -> None:
        self._workstations = []
        self._mapped_names = {}

    def old_add_workstation(self, machine: dict):
        for ma in machine.keys():
            print(f"adding: {ma} {self._workstations}")
            if ma not in self._workstations:
                self._workstations.append(ma)
        return self._workstations

    def add_workstation(self, machine: dict):
        for ma in machine.keys():
            print(f"adding: {ma} {self._workstations}")
            if ma not in self._workstations:
                self._workstations.append(ma)

    @property
    def workstations(self):
        return self._workstations

    @property
    def mapped_names(self):
        return self._mapped_names

    def processing_times(self, machine):
        # all other workstations are 0 except in machine
        # processing_time = self._workstations.copy()
        processing_time = {}
        for i, wo in enumerate(self._workstations):
            self._mapped_names[f"machine_{i+1}"] = wo

            if wo not in machine:
                processing_time[f"machine_{i+1}"] = 1
            else:
                processing_time[f"machine_{i+1}"] = int(machine[wo])

        # processing_times = {}
        # for i, wo in enumerate(processing_time):
        #     self._mapped_names[f"machine_{i+1}"] = wo
        #     processing_times[f"machine_{i+1}"] = int(processing_time[wo])

        return processing_time

    @property
    def num_of_workstations(self):
        return len(self._workstations)


# Indirect approach
import frappe

frappe.connect("test", db_name="cole")

number_jobs = 0
number_total_machines = 1
number_operations = 0
workstations = {}
test_wkn = 0
cole = []
y = Workstations()

for i, wos in enumerate(frappe.get_all("Work Order")):
    wo = frappe.get_doc("Work Order", wos["name"]).as_dict()
    job = {"job_id": i, "operations": []}
    number_jobs += 1
    operations = []
    pred = 0

    for j, ops in enumerate(wo["operations"]):
        if PRINTS == True:
            print(ops["operation"])
            print(ops["time_in_mins"])
            print(ops["workstation"])
        if pred == 0:
            _pred = None
            pred += 1
        else:
            _pred = number_operations - 1

        operation = {
            "operation_id": number_operations,
            "processing_times": {},
            "predecessor": _pred,
        }
        number_operations += 1

        machine_name = f'{ops["workstation"]}'
        # machine_name = f'machine_{number_total_machines}'
        processing_time = {machine_name: ops["time_in_mins"]}
        y.add_workstation(processing_time)
        test_wkn += 1

        number_total_machines += 1
        operation["processing_times"] = processing_time
        # operation['processing_times'] = y.processing_times(processing_time)
        operations.append(operation)

    job["operations"] = operations

    cole.append(job)
    if i == 3:
        break

for i in cole:
    for j in i["operations"]:
        j["processing_times"] = y.processing_times(j["processing_times"])


processing_info = {
    "instance_name": "custom_problem_instance",
    "nr_machines": y.num_of_workstations,
    "jobs": cole,
    "sequence_dependent_setup_times": {
        f"machine_{i+1}": np.zeros(
            (number_operations, number_operations), dtype=int
        ).tolist()
        for i in range(y.num_of_workstations)
    },
}

print(y.mapped_names)
print(test_wkn)
import json

with open(
    r"/home/cole/cole_scripts/job_scheduling_env/processing_info.json", "w"
) as file:
    json.dump(processing_info, file, indent=4, sort_keys=True)


import matplotlib

matplotlib.use("Qt5Agg")  # or 'Qt5Agg' depending on your preference


mode = 0
if mode == 1:
    parameters = {
        "instance": {"problem_instance": "custom_problem_instance"},
        "algorithm": {
            "population_size": 10,
            "ngen": 10,
            "seed": 5,
            "cr": 0.7,
            "indpb": 0.2,
            "multiprocessing": True,
        },
        "output": {"logbook": True},
    }

    jobShopEnv = parse(processing_info)
    population, toolbox, stats, hof = initialize_run(jobShopEnv, **parameters)

    makespan, jobShopEnv = run_GA(
        jobShopEnv, population, toolbox, stats, hof, **parameters
    )

    plt = plot_gantt_chart(jobShopEnv)
    plt.show()
else:
    parameters = {
        "instance": {"problem_instance": "custom_problem_instance"},
        "solver": {"time_limit": 10000, "model": "fjsp_sdst"},
        "output": {"logbook": True},
    }

    jobShopEnv = parse(processing_info)
    results, jobShopEnv = run_CP_SAT(jobShopEnv, **parameters)
    plt = plot_gantt_chart(jobShopEnv)
    plt.show()

if OUTPUT is True:
    with open(r"/home/cole/cole_scripts/job_scheduling_env/results.json", "w") as file:
        json.dump(results, file, indent=4, sort_keys=True)
    print(results)

    import pickle

    with open(
        r"/home/cole/cole_scripts/job_scheduling_env/jobShopEnv.pkl", "wb"
    ) as file:
        pickle.dump(jobShopEnv, file)
