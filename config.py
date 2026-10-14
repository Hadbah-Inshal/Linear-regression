# Edit this file if you pick different datasets.
# csv    -> path to the dataset file
# target -> name of the column you want to predict (must be numeric)
DATASETS = {
    "insurance": {
        "title": "Health_insurance",
        "csv": "data/Health_insurance.csv",
        "target": "charges",
    },
    "student": {
        "title": "Student_performance_data",
        "csv": "data/Student_performance_data.csv",
        "target": "GradeClass",
    },
}
