#!/bin/sh
python3 plot_matchings.py
python3 benchmark_amazon.py
python3 benchmark_eu.py
python3 benchmark_village.py
python3 benchmark_wikicats.py
python3 benchmark_regularized_amazon.py
python3 benchmark_regularized_eu.py
python3 benchmark_regularized_village.py
python3 benchmark_regularized_wikicats.py
python3 plot_gwa_village.py
python3 plot_energy.py
python3 plot_node_correctness.py