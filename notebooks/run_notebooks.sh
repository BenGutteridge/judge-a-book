#!/bin/bash

# Run IAM pre-processing
python notebooks/00_iam_preprocessing.py

# Run experiments
python notebooks/01_experiments.py experiment_mode iam_multipage_minpages=02 gpt_model gpt-4o
python notebooks/01_experiments.py experiment_mode iam_multipage_minpages=02 gpt_model gpt-4o-mini
python notebooks/01_experiments.py experiment_mode iam_multipage_2-10_pages_10_docs gpt_model gpt-4o

# Error handling
python notebooks/02_error_catching.py
python notebooks/02_error_catching.py mode pagecount

# Generate tables and figures
python notebooks/03_performance_tables.py # table latex copied to clipboard
python notebooks/03_dropfirst_tables.py # # table latex copied to clipboard
python notebooks/03_pagecount_plot.py # figure saved to /plots
python notebooks/03_pareto_plot.py # figure saved to /plots