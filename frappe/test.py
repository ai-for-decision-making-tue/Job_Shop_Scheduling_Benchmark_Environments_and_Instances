import re


def compare_schema(reference, target, path="root", flexible_types=None):
    # Define flexible types if not provided
    if flexible_types is None:
        flexible_types = {
            "root.jobs[\\d+].operations[\\d+].predecessor": (
                int,
                type(None),
            )  # Allow `predecessor` to be `int` or `None`
        }

    # Handle dictionaries
    if isinstance(reference, dict):
        if not isinstance(target, dict):
            return f"Mismatch at {path}: Expected dict, got {type(target).__name__}"

        for key in reference:
            if key not in target:
                return f"Key missing at {path}: {key}"
            result = compare_schema(
                reference[key], target[key], f"{path}.{key}", flexible_types
            )
            if result:
                return result

        # Allow extra keys in `processing_times` and `sequence_dependent_setup_times`
        if (
            path.endswith(".processing_times")
            or path == "root.sequence_dependent_setup_times"
        ):
            return None  # Allow extra keys

        for key in target:
            if key not in reference:
                return f"Extra key found at {path}: {key}"

    # Handle lists
    elif isinstance(reference, list):
        if not isinstance(target, list):
            return f"Mismatch at {path}: Expected list, got {type(target).__name__}"

        if reference:
            for i, (ref_item, tgt_item) in enumerate(zip(reference, target)):
                result = compare_schema(
                    ref_item, tgt_item, f"{path}[{i}]", flexible_types
                )
                if result:
                    return result

    # Handle flexible types
    for pattern, allowed_types in flexible_types.items():
        if re.match(pattern, path):
            if not isinstance(target, allowed_types):
                return f"Type mismatch at {path}: Expected one of {allowed_types}, got {type(target).__name__}"

    # Handle base case
    if not isinstance(target, type(reference)):
        return f"Type mismatch at {path}: Expected {type(reference).__name__}, got {type(target).__name__}"

    return None


# Example Usage
processing_info_test = {
    "instance_name": "custom_problem_instance",
    "nr_machines": 2,
    "jobs": [
        {
            "job_id": 0,
            "operations": [
                {
                    "operation_id": 0,
                    "processing_times": {"machine_1": 10, "machine_2": 20},
                    "predecessor": None,
                },
                {
                    "operation_id": 1,
                    "processing_times": {"machine_1": 25, "machine_2": 19},
                    "predecessor": 0,
                },
            ],
        },
        {
            "job_id": 1,
            "operations": [
                {
                    "operation_id": 2,
                    "processing_times": {"machine_1": 23, "machine_2": 21},
                    "predecessor": None,
                },
                {
                    "operation_id": 3,
                    "processing_times": {"machine_1": 12, "machine_2": 24},
                    "predecessor": 2,
                },
            ],
        },
        {
            "job_id": 2,
            "operations": [
                {
                    "operation_id": 4,
                    "processing_times": {"machine_1": 37, "machine_2": 21},
                    "predecessor": None,
                },
                {
                    "operation_id": 5,
                    "processing_times": {"machine_1": 23, "machine_2": 34},
                    "predecessor": 4,
                },
            ],
        },
    ],
    "sequence_dependent_setup_times": {
        "machine_1": [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        "machine_2": [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    },
}


import json

with open(
    r"/home/cole/cole_scripts/job_scheduling_env/processing_info.json", "r"
) as file:
    processing_info = json.load(file)


# Validate
result = compare_schema(processing_info_test, processing_info)
if result:
    print(f"Schema mismatch: {result}")
else:
    print("Schemas match.")


# processing_info = {
#     "instance_name": "custom_problem_instance",
#     "nr_machines": 4,
#     "jobs": [
#         {
#             "job_id": 0,
#             "operations": [
#                 {
#                     "operation_id": 0,
#                     "processing_times": {
#                         "machine_1": 12,
#                         "machine_2": 1,
#                         "machine_3": 1,
#                         "machine_4": 1,
#                     },
#                     "predecessor": None,
#                 },
#                 {
#                     "operation_id": 1,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 300,
#                         "machine_3": 1,
#                         "machine_4": 1,
#                     },
#                     "predecessor": 0,
#                 },
#                 {
#                     "operation_id": 2,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 1,
#                         "machine_3": 180,
#                         "machine_4": 1,
#                     },
#                     "predecessor": 1,
#                 },
#                 {
#                     "operation_id": 3,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 1,
#                         "machine_3": 1,
#                         "machine_4": 6,
#                     },
#                     "predecessor": 2,
#                 },
#             ],
#         },
#         {
#             "job_id": 1,
#             "operations": [
#                 {
#                     "operation_id": 4,
#                     "processing_times": {
#                         "machine_1": 12,
#                         "machine_2": 1,
#                         "machine_3": 1,
#                         "machine_4": 1,
#                     },
#                     "predecessor": None,
#                 },
#                 {
#                     "operation_id": 5,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 300,
#                         "machine_3": 1,
#                         "machine_4": 1,
#                     },
#                     "predecessor": 0,
#                 },
#                 {
#                     "operation_id": 6,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 1,
#                         "machine_3": 180,
#                         "machine_4": 1,
#                     },
#                     "predecessor": 1,
#                 },
#                 {
#                     "operation_id": 7,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 1,
#                         "machine_3": 1,
#                         "machine_4": 6,
#                     },
#                     "predecessor": 2,
#                 },
#             ],
#         },
#         {
#             "job_id": 2,
#             "operations": [
#                 {
#                     "operation_id": 8,
#                     "processing_times": {
#                         "machine_1": 12,
#                         "machine_2": 1,
#                         "machine_3": 1,
#                         "machine_4": 1,
#                     },
#                     "predecessor": None,
#                 },
#                 {
#                     "operation_id": 9,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 300,
#                         "machine_3": 1,
#                         "machine_4": 1,
#                     },
#                     "predecessor": 0,
#                 },
#                 {
#                     "operation_id": 10,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 1,
#                         "machine_3": 180,
#                         "machine_4": 1,
#                     },
#                     "predecessor": 1,
#                 },
#                 {
#                     "operation_id": 11,
#                     "processing_times": {
#                         "machine_1": 1,
#                         "machine_2": 1,
#                         "machine_3": 1,
#                         "machine_4": 6,
#                     },
#                     "predecessor": 2,
#                 },
#             ],
#         },
#     ],
#     "sequence_dependent_setup_times": {
#         "machine_1": [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#         "machine_2": [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#         "machine_3": [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#         "machine_4": [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         ],
#     },
# }
