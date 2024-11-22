from setuptools import setup, find_packages

setup(
    name="Job_Shop_Scheduling_Benchmark_Environments_and_Instances",
    version="0.1.0",
    description="A benchmarking repo with various solution methods to various machine scheduling problems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=[
        "data",
        "frappe",
        "configs",
        "plotting",
        "data_parsers",
        "solution_methods",
        "scheduling_environment",
    ],
    install_requires=[],  # Add your dependencies here
)
