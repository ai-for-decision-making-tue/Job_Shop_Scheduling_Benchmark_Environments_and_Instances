import numpy as np

from solution_methods.DANIEL.src.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from solution_methods.helper_functions import initialize_device


class FJSPEnv_test(FJSPEnvForSameOpNums):
    def __init__(self, JobShop_module, parameters):
        n_j = JobShop_module.nr_of_jobs
        n_m = JobShop_module.nr_of_machines
        device = initialize_device(parameters, method="DANIEL")
        super().__init__(n_j=n_j, n_m=n_m, device=device)

        # Assign values to the job_length_list and op_pt_list
        job_length_list = np.asarray([[job.nr_of_ops for job in JobShop_module.jobs]])
        op_pt_list = np.zeros((1, JobShop_module.nr_of_operations, n_m), dtype=np.int32)

        for job in JobShop_module.jobs:
            for operation in job.operations:
                for machine, pt in operation.processing_times.items():
                    op_pt_list[0, operation.operation_id, machine] = pt

        super(FJSPEnv_test, self).set_initial_data(job_length_list, op_pt_list)

        self.JSP_instance = JobShop_module

    def step(self, actions):
        chosen_job = (actions // self.number_of_machines).item()
        chosen_mch = (actions % self.number_of_machines).item()
        chosen_op = self.candidate[self.env_idxs, chosen_job].item()
        operation = self.JSP_instance.operations[chosen_op]
        duration = operation.processing_times[chosen_mch]
        self.JSP_instance.schedule_operation_on_machine(operation, chosen_mch, duration)
        self.JSP_instance.get_job(operation.job_id).scheduled_operations.append(operation)
        
        state, reward, done = super(FJSPEnv_test, self).step(actions)
        return state, reward, done

    def reset(self):
        self.JSP_instance.reset()
        super(FJSPEnv_test, self).reset()
