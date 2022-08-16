# -*- coding: utf-8 -*-

"""

@author: Liz Verbeek

TODO: write header comment

"""
import os
import time

import numpy as np
import pandas as pd

from model import EnergyModel

# -- HYPERPARAMETERS -- #
steps = 50
# random_seeds = [0]
random_seeds = np.linspace(0, 10e6, 10, dtype="int")

# -- MODEL PARAMETERS -- #
# TODO: make user input
n_households = 10000
decision_making_model = "Rational"  # "Rational" or "TPB"

for i, random_seed in enumerate(random_seeds):
    print("Run num", i, "with random seed", random_seed)
    print("Number of households:", n_households)
    
    tic = time.time()
    model = EnergyModel(n_households, decision_making_model, random_seed)

    for j in range(steps):
        print("# ------------ Step", j+1, "------------ #")
        model.step()

    toc = time.time()
    print("MODEL runtime:", toc-tic)

    model_vars = model.datacollector.get_model_vars_dataframe()
    print(model_vars)
    agent_vars = model.datacollector.get_agent_vars_dataframe()
    print(agent_vars)

    # -- STORING OUTPUT -- #
    if not os.path.isdir("results"):
        os.makedirs(("results"))

    # EFFICIENT: store pickles
    model_vars.to_pickle("results/model_variables_[seed" +
                         str(random_seed) + "].pickle")
    agent_vars.to_pickle("results/agent_variables_[seed" +
                         str(random_seed) + "].pickle")

    # # DIRECTLY READABLE: store csv files
    # model_vars.to_csv("results/model_variables_[seed" + str(random_seed) + "].csv")
    # agent_vars.to_csv("results/agent_variables_[seed" + str(random_seed) + "].csv")

    print()

print("# -------- FINISHED", len(random_seeds), "runs -------- #")