# -*- coding: utf-8 -*-

"""

@author: Liz Verbeek

TODO: write header comment

"""
import time

import random
import numpy as np
import pandas as pd
import networkx as nx

from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from collections import Counter

from household import Household


class EnergyModel(Model):
    """Model class for the energy model. """

    def __init__(self, n_households=10000, decision_making_model="TPB",
                 random_seed=12345678):
        """Initialization of the energy model.

        Args:
            n_households            : Number of households in the model
            decision_making_model   : Type of household decision making theory

        """
        self.current_id = 0                         # Agent ID count
        self.schedule = BaseScheduler(self)         # Model scheduler

        # -- Regulate stochasticity -- #
        self.random_seed = random_seed

        # -- Model parameters: energy-related --#
        self.PV_lifespan = 25
        self.energy_cost = (-228.7, 0.25618)   # Fixed cost; cost per kWh
        self.CO2_emission = 0.425

        # -- Model parameters: financial --#
        self.interest_rate = -0.0054877
        self.hh_savings_ratio = 0.235
        PV_timesteps = np.linspace(0, self.PV_lifespan - 1 , self.PV_lifespan)
        self.discount_rates = (1 + self.interest_rate)**PV_timesteps

        # -- Initialize social network -- #
        self.G = nx.watts_strogatz_graph(n=n_households, k=7, p=1,
                                         seed=int(random_seed))
        # Relabel nodes to be consistent with agent IDs
        self.G = nx.relabel_nodes(self.G, lambda x: x + 1)

        # -- Initialize households -- #
        self.decision_making_model = decision_making_model
        if self.decision_making_model == "TPB":
            # Generate weight distribution for TPB utility function weights
            TPB_weights = self.generate_TPB_weights(n_households)
            # Add households to model
            for n in range(n_households):
                self.add_household(TPB_weights[n])
        elif self.decision_making_model == "Rational":
            # Add households to model
            for n in range(n_households):
                self.add_household()
        # Assign social network connections for each household
        for hh in self.schedule._agents.values():
            hh.neighbors = [self.schedule._agents[hh_id] for hh_id
                              in self.G.neighbors(hh.unique_id)]

        # -- Initialize normalization factors -- #
        self.max_NPV = max(max(hh.NPV) for hh
                           in self.schedule._agents.values())
        self.max_CE = max(max(hh.CE_ratio) for hh
                          in self.schedule._agents.values())

        # -- Initialize ouput collection -- #
        model_reporters = {"Diffusion rate": self.diffusion_rate,
                           "CO2_saved (kilotonne)": self.total_CO2_saved,
                           "PV_investment (Euros)": self.total_PV_investment}
        agent_reporters = {"Income (Euros)": "income",
                           # "Attitude": lambda hh: hh.TPB_attributes["PV"][0],
                           "PV_installed": "PV_installed",
                           "PV_investment (Euros)": lambda hh: hh.PV_costs if hh.PV_installed else 0,
                           "CO2_saved (kg)": "CO2_saved"}
        self.datacollector = DataCollector(model_reporters=model_reporters,
                                           agent_reporters=agent_reporters)
        self.datacollector.collect(self)

    # ------------------------------------------------ #
    # -------- MODEL INITIALIZATION FUNCTIONS -------- #
    # ------------------------------------------------ #
    def generate_TPB_weights(self, n_households):
        """Generate weights for TPB decision-making module.

        Args:
            n_households        : Number of households to add to model

        Returns:
            TPB_weights         : List of TPB weights for entire population
        """

        # Read weight distributions for PV installation
        df = pd.read_csv("TPB_weights_PV.csv")
        # Generate weights for total population
        TPB_weights = np.repeat([row.tolist() for key, row in
                                 df.set_index("Pop share").iterrows()],
                                round(df["Pop share"] *
                                      n_households).astype(int).values, axis=0)
        # Correct for rounding errors in population shares
        # Solution: add/remove single household with most common weight type
        if len(TPB_weights) < n_households:
            for _ in range(n_households - len(TPB_weights)):
                w_idx = np.argmax(df["Pop share"] * n_households)
                weights = df.loc[w_idx][1:].tolist()
                TPB_weights = np.append(TPB_weights, [weights], axis=0)
        elif len(TPB_weights) > n_households:
            for _ in range(len(TPB_weights) - n_households):
                w_idx = np.argmax(df["Pop share"] * n_households)
                weights = df.loc[w_idx][1:].tolist()
                w_idx_new = np.where(TPB_weights == weights)[0][0]
                TPB_weights = np.delete(TPB_weights, w_idx_new, axis=0)

        return TPB_weights

    def add_household(self, TPB_weights=None):
        """Add a single household to model, return household object."""
        household = Household(self, TPB_weights)
        self.schedule.add(household)
        return household

    # ------------------------------------------------ #
    # --------- OUTPUT COLLECTION FUNCTIONS ---------- #
    # ------------------------------------------------ #
    def diffusion_rate(self):
        """Return diffusion rate of PV installation in current timestep."""
        hh_PV = sum(hh.PV_installed for hh in self.schedule._agents.values())
        hh_total = self.schedule.get_agent_count()
        return hh_PV/hh_total

    def total_CO2_saved(self):
        """Return annual total CO2 savings (in kilotonne per year)."""
        total_CO2_saved = sum(hh.CO2_saved for hh
                              in self.schedule._agents.values()
                              if hh.PV_installed) * 1e-6
        return total_CO2_saved

    def total_PV_investment(self):
        """ Return total amount of money invested in PVs. """
        total_PV_investment = sum(hh.PV_costs for hh
                                  in self.schedule._agents.values()
                                  if hh.PV_installed)
        return total_PV_investment

    def step(self):
        """Describes a single model step. """
        self.schedule.step()
        self.datacollector.collect(self)
