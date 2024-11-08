import numpy as np

import numpy as np


def permute_rows(x):
    # Permutes the rows of a numpy array `x` randomly.
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uniform_instance_generator(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines


def override(fn):
    """
    override decorator
    """
    return fn


if __name__ == "__main__":
    # Set parameters
    j = 20              # Number of jobs
    m = 10              # Number of machines
    l = 1               # Minimum processing time
    h = 99              # Maximum processing time
    batch_size = 100    # nr of instances to generate
    seed = 201          # Random seed

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate data for batch_size number of instances
    data = np.array([uniform_instance_generator(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
    print(f"Generated data shape: {data.shape}")
    np.save(f'generated_data/test_generatedData{j}_{m}_Seed{seed}.npy', data)