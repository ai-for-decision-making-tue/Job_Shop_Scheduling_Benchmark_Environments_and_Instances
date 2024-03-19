import argparse
from solution_methods.FJSP_DRL import training
from solution_methods.FJSP_DRL import test

PARAM_FILE = "configs/FJSP_DRL.toml"


def main(execute_mode, param_file: str):
    if execute_mode == 'train':
        training.train_FJSP_DRL(param_file)
    if execute_mode == 'test':
        test.test_instance(param_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FJSP_DRL")
    parser.add_argument(
        "config_file",
        metavar='-f',
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()

    # if train the FJSP_DRL model with a tensor env, please specify the mode as 'train'
    # mode = 'train'
    # the size of training cases (number of jobs and machines) can be modified in /configs/FJSP_DRL.toml

    # if test a saved model using a simulation env, please specify the mode as 'test'
    # the test instance can be specified in /configs/FJSP_DRL.toml
    mode = 'test'

    main(mode, param_file=args.config_file)
