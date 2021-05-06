
import os
import sys
import shutil

glp = os.path.abspath("../../code")
if not glp in sys.path: sys.path.append( glp )

from global_paths import global_paths

if not global_paths["COBS"] in sys.path: sys.path.append( global_paths["COBS"] )

import cobs
import torch
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from copy import deepcopy

from BuildingOccupancy import Person, Meeting, WeeklyMeeting, OneTimeMeeting, BuildingOccupancy
import DefaultBuildings
from Agents import agent_constructor
from CentralController import ddpg_episode_mc
import RLCritics
import StateUtilities as SU
from Options import get_argparser
from ReplayBuffer import ReplayBufferStd

args = get_argparser().parse_args([
        "--algorithm", "ddqn",
        "--idf_file",  "",
        "--epw_file",  os.path.abspath("../../../COBS/cobs/data/weathers/8.epw"),
        "--checkpoint_dir", "/tmp",
        "--shared_network_per_agent_class"
    ])

cobs.Model.set_energyplus_folder(global_paths["eplus"])
building = DefaultBuildings.Building_5ZoneAirCooled_SingleSetpoint(args)
building_occ = BuildingOccupancy()
building_occ.set_room_settings(building.room_names[:-1], {building.room_names[-1]: 40}, 40)
building_occ.generate_random_occupants(20)
building_occ.generate_random_meetings(10,10)



# Define the agents
agents = []
idx    = 0
for agent_name, (controlled_device, controlled_device_type) in building.agent_device_pairing.items():
    new_agent = agent_constructor( controlled_device_type )
    new_agent.initialize(
                         name = agent_name,
                         args = args,
                         controlled_element = controlled_device,
                         global_state_keys  = building.global_state_variables)
    agents.append(new_agent)
    idx += 1


building.model.set_runperiod(30, 2020, 1, 1)
building.model.set_timestep(1)

episode_output_lists = []
ts_diff_in_min = 60
evaluation_epoch = False
hyper_params = args
LAMBDA_REWARD_ENERGY = hyper_params.lambda_rwd_energy
LAMBDA_REWARD_MANU_STP_CHANGES = hyper_params.lambda_rwd_mstpc
TAU_TARGET_NETWORKS  = hyper_params.tau
DISCOUNT_FACTOR = hyper_params.discount_factor
BATCH_SIZE      = hyper_params.batch_size
RPB_BUFFER_SIZE = hyper_params.rpb_buffer_size
LEARNING_RATE   = hyper_params.lr

episode_number = 0



epsilon = 0.05
for agent in agents:
    agent.epsilon = epsilon


# Define the replay ReplayBuffer
rpb = ReplayBufferStd(size=RPB_BUFFER_SIZE, number_agents=len(agents))
# Define the loss
loss = torch.nn.MSELoss()
# prepare the simulation
state = building.model_reset()
SU.fix_year_confussion(state)
norm_state_ten = SU.unnormalized_state_to_tensor(state, building)
#
current_occupancy = building_occ.draw_sample( state["time"] )
timestep   = 0
last_state = None
# start the simulation loop


while not building.model_is_terminate():
        actions = list()

        currdate = state['time']
        #
        # request occupancy for the next state
        nextdate = state['time'] + datetime.timedelta(minutes=ts_diff_in_min)
        next_occupancy = building_occ.draw_sample(nextdate)
        #
        # propagate occupancy values to COBS / EnergyPlus
        for zonename, occd in next_occupancy.items():
            actions.append({"priority":        0,
                            "component_type": "Schedule:Constant",
                            "control_type":   "Schedule Value",
                            "actuator_key":  f"OCC-SCHEDULE-{zonename}",
                            "value":           next_occupancy[zonename]["relative number occupants"],
                            "start_time":      state['timestep'] + 1})

        #
        # request new actions from all agents
        agent_actions_dict = {}
        agent_actions_list = []
        add_random_process = True
        if evaluation_epoch and not add_epsilon_greedy_in_eval_epoch:
            add_random_process = False
        if agents[0].shared_network_per_agent_class:
            new_actions = agents[0].next_action(norm_state_ten, add_random_process)
            agent_actions_list = new_actions
            # decode the actions for every agent using the individual agent objects
            for idx, agent in enumerate(agents):
                agent_actions_dict[agent.name] = agent.output_action_to_action_dict(new_actions[idx])
        else:
            for agent in agents:
                new_action = agent.next_action(norm_state_ten, add_random_process)
                agent_actions_list.append( new_action )
                agent_actions_dict[agent.name] = agent.output_action_to_action_dict(new_action)
            # no backtransformation of variables needed, this is done in agents definition already

        #
        # send agent actions to the building object and obtaion the actions for COBS/eplus
        actions.extend( building.obtain_cobs_actions( agent_actions_dict, state["timestep"]+1 ) )

        #
        # send actions to EnergyPlus and obtian the new state
        norm_state_ten_last = norm_state_ten
        last_state = state
        timestep  += 1
        state      = building.model_step(actions)
        current_occupancy = next_occupancy
        SU.fix_year_confussion(state)

        current_energy_Wh = state["energy"] / 360

        #
        # modify state
        norm_state_ten = SU.unnormalized_state_to_tensor(state, building)

        #
        # send current temp/humidity values for all rooms
        # obtain number of manual setpoint changes
        _, n_manual_stp_changes = building_occ.manual_setpoint_changes(state['time'], state["temperature"], None)

        #
        # reward computation
        if hyper_params is None or hyper_params.reward_function == "sum_energy_mstpc":
            reward = -( LAMBDA_REWARD_ENERGY * current_energy_Wh + LAMBDA_REWARD_MANU_STP_CHANGES * n_manual_stp_changes )
        elif hyper_params.reward_function == "rulebased_roomtemp":
            reward = - reward_fn_rulebased_roomtemp(state, building)
        #elif hyper_params.reward_function == "rulebased_agent_output":
        else:
            reward = - reward_fn_rulebased_agent_output(state, agent_actions_dict)
        if not hyper_params is None and hyper_params.log_reward:
            reward = - np.log(-reward + 1)

        #
        # save (last_state, actions, reward, state) to replay buffer
        rpb.add_transition(norm_state_ten_last, agent_actions_list, reward, norm_state_ten)

        #
        # sample minibatch
        b_state1, b_action, b_reward, b_state2 = rpb.sample_minibatch(BATCH_SIZE, False)
        b_action = torch.tensor(b_action)

        #
        # loop over all [agent, critic]-pairs
        if agents[0].shared_network_per_agent_class:
            #
            # compute y (i.e. the TD-target)
            #  Hint: s_{i+1} <- state2; s_i <- state1
            agents[0].model_actor.zero_grad()
            b_reward = b_reward.detach().expand(-1, len(agents) ).flatten()[:, np.newaxis]
            # wrong: b_reward = b_reward.detach().repeat(len(agents), 1)
            y = b_reward + DISCOUNT_FACTOR * agents[0].step_tensor(b_state2, use_actor = False).detach().max(dim=1).values[:, np.newaxis]
            # compute Q for state1
            q = agents[0].step_tensor(b_state1, use_actor = True).gather(1, b_action.flatten()[:, np.newaxis])
            # update agent by minimizing the loss L
            L = loss(q, y)
            L.backward()
            agents[0].optimizer_step()
        else:
          for agent_id, agent in enumerate(agents):
            #
            # compute y (i.e. the TD-target)
            #  Hint: s_{i+1} <- state2; s_i <- state1
            agent.model_actor.zero_grad()
            y = b_reward.detach() + DISCOUNT_FACTOR * agent.step_tensor(b_state2, use_actor = False).detach().max(dim=1).values[:, np.newaxis]
            # compute Q for state1
            q = agent.step_tensor(b_state1, use_actor = True).gather(1, b_action[:, agent_id][:, np.newaxis])
            # update agent by minimizing the loss L
            L = loss(q, y)
            L.backward()
            agent.optimizer_step()

        if timestep % 20 == 0:
            eval_ep_str     = "  " if evaluation_epoch   else "no"
            rand_pr_add_str = "  " if add_random_process else "no"
            if timestep % 200 == 0:
                print(f"ep. {episode_number:3}, ts. {timestep:5}: {state['time']}, {eval_ep_str} eval ep., {rand_pr_add_str} rand. p. add.")
            #else:
            #    print(f"ep. {episode_number:3}, ts. {timestep:5}: {state['time']}, {eval_ep_str} eval ep., {rand_pr_add_str} rand. p. add.", end="\r")



