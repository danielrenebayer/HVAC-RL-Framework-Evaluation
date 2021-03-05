#
# This training script initializes a complete new simulation run
# and trains all agents
#
# Building              5ZoneAirCooled
# Wheather              Chicage O'Hare Airport
# Episode period        1. July to 30. July
# Number of occupants   40
# Time resolution       5 minutes
# Number of training episodes   100
# Critic hidden layer size      40
#

import os
import sys
import shutil

glp = os.path.abspath("../code")
if not glp in sys.path: sys.path.append( glp )

from global_paths import global_paths

if not global_paths["COBS"] in sys.path: sys.path.append( global_paths["COBS"] )

import cobs
import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from BuildingOccupancy import Person, Meeting, WeeklyMeeting, OneTimeMeeting, BuildingOccupancy
from DefaultBuildings import Building_5ZoneAirCooled
from Agents import agent_constructor
from CentralController import ddpg_episode_mc
import RLCritics
import StateUtilities as SU

cobs.Model.set_energyplus_folder(global_paths["eplus"])

checkpoint_dir = "checkpoints/001" + datetime.datetime.now().strftime("%Y%m%d-%H%M")

#
# Define the building and the occupants
building = Building_5ZoneAirCooled()
building_occ = BuildingOccupancy()
building_occ.set_room_settings(building.room_names[:-1], {building.room_names[-1]: 40}, 40)
building_occ.generate_random_occupants(40)
building_occ.generate_random_meetings(15,0)

#
# Define the agents
agents = []
# HINT: a device can be a zone, too
for agent_name, (controlled_device, controlled_device_type) in building.agent_device_pairing.items():
    new_agent = agent_constructor( controlled_device_type )
    new_agent.initialize(name = agent_name)
    agents.append(new_agent)

#
# Define the critics
critics = []
ciritic_input_variables=["Minutes of Day","Day of Week","Calendar Week",
                         "Outdoor Air Temperature","Outdoor Air Humidity",
                         "Outdoor Wind Speed","Outdoor Wind Direction",
                         "Outdoor Solar Radi Diffuse","Outdoor Solar Radi Direct"]
for vartype in ["Zone Temperature","Zone People Count",
                "Zone Relative Humidity",
                "Zone VAV Reheat Damper Position","Zone CO2"]:
    ciritic_input_variables.extend( [f"SPACE{k}-1 {vartype}" for k in range(1,6)] )
for agent in agents:
    new_critic = RLCritics.CriticMergeAndOnlyFC(
                    hidden_size = 40,
                    input_variables=ciritic_input_variables,
                    agents = agents)
    critics.append(new_critic)

#
# Set model parameters
building.model.set_runperiod(30, 2020, 7, 1)
building.model.set_timestep(12) # 5 Min interval, 12 steps per hour

n_episode_runs = 100

os.makedirs(checkpoint_dir, exist_ok=True)

for n_episode in range(n_episode_runs):
    output_lists = {
        "episode_list": [],
        "timestamp_list": [],
        "loss_list": [],

        "room_temp_list": [],
        "outd_temp_list": [],
        "outd_humi_list": [],
        "outd_solar_radi_list": [],
        "outd_wspeed_list": [],
        "outd_wdir_list": [],
        "occupancy_list": [],
        "humidity_list": [],
        "co2_ppm_list": [],
        "energy_list": [],
        "rewards_list": [],
        "n_manual_stp_ch_list": [],

        "vav_pos_list": []
    }
    
    ddpg_episode_mc(building, building_occ, agents, critics, output_lists, n_episode)
    
    # save agent/critic networks every 10th run
    if n_episode+1 % 10 == 0:
        for agent in agents: agent.save_models_to_disk(checkpoint_dir, prefix=f"episode_{n_episode}_")
        for critic in critics: agent.save_models_to_disk(checkpoint_dir, prefix=f"episode_{n_episode}_")
    # save the output_lists
    f = open(os.path.join(checkpoint_dir, f"epoch_{n_episode}_output_lists.pickle"), "wb")
    pickle.dump(datalist, f)
    f.close()

#
# save the building_occ object
f = open(os.path.join(checkpoint_dir, "building_occ.pickle"), "wb")
pickle.dump(datalist, f)
f.close()


