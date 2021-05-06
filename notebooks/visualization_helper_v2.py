
#
# New visualization helper
# For training runs without eees, eeesea tables
#

import os
import re
import ast
import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tables = ["eels", "sees", "seesea", "sees_er"]

def convert_sqlite_to_df(path_to_db):
    db = sqlite3.connect(os.path.join(path_to_db, "ouputs.sqlite"))
    dfs = {tname: None for tname in tables}
    for table in tables:
        dfs[table] = pd.read_sql(f"SELECT * from {table};", db)
        print(f"Table {table} convertet to a pandas dataframe.")
    db.close()
    return dfs

def DoW_ItoS(day_of_week_int):
    day_of_week_dict = {
        0: "Mo.",
        1: "Di.",
        2: "Mi.",
        3: "Do.",
        4: "Fr.",
        5: "Sa.",
        6: "So."}
    return day_of_week_dict[ day_of_week_int ]

def select_week_and_episode(dfs, selected_episode, start, end = None):
    subdfs = {}
    subdfs["eels"]   = dfs["eels"].loc[    dfs["eels"].loc[:,"episode"]    == selected_episode ]
    subdfs["sees"]   = dfs["sees"].loc[    dfs["sees"].loc[:,"episode"]    == selected_episode ].set_index("step")
    subdfs["seesea"] = dfs["seesea"].loc[  dfs["seesea"].loc[:,"episode"]  == selected_episode ].set_index("step")
    subdfs["seeser"] = dfs["sees_er"].loc[ dfs["sees_er"].loc[:,"episode"] == selected_episode ].set_index("step")

    subdfs["sees"]["datetime"]   = pd.to_datetime(subdfs["sees"]["datetime"])
    subdfs["seesea"]["datetime"] = subdfs["sees"]["datetime"]
    subdfs["seeser"]["datetime"] = subdfs["sees"]["datetime"]
    if end is None:
        end = start + timedelta(days=7)
    subdfs["sees"]   = subdfs["sees"].loc[   subdfs["sees"]["datetime"]   >= start ]
    subdfs["seesea"] = subdfs["seesea"].loc[ subdfs["seesea"]["datetime"] >= start ]
    subdfs["seeser"] = subdfs["seeser"].loc[ subdfs["seeser"]["datetime"] >= start ]
    subdfs["sees"]   = subdfs["sees"].loc[   subdfs["sees"]["datetime"]   <= end ]
    subdfs["seesea"] = subdfs["seesea"].loc[ subdfs["seesea"]["datetime"] <= end ]
    subdfs["seeser"] = subdfs["seeser"].loc[ subdfs["seeser"]["datetime"] <= end ]

    subdfs["sees"].set_index(  "datetime", inplace=True)
    subdfs["seesea"].set_index("datetime", inplace=True)
    subdfs["seeser"].set_index("datetime", inplace=True)

    return subdfs

def select_week_and_episode_for_dfs_list(alldfs, selected_episodes, selected_weeks):
    subdfs = []
    for idx, dfs in enumerate(alldfs):
        subdfs1 = select_week_and_episode(dfs, selected_episodes[idx], selected_weeks[idx])
        subdfs.append( subdfs1 )
    return subdfs

def select_week_and_episode_with_end_for_dfs_list(alldfs, selected_episodes, selected_periods_start, selected_periods_end):
    subdfs = []
    for idx, dfs in enumerate(alldfs):
        subdfs1 = select_week_and_episode(dfs, selected_episodes[idx], selected_periods_start[idx], selected_periods_end[idx])
        subdfs.append( subdfs1 )
    return subdfs

def get_runtime_overview_df(dfs1, dfs2, colname1="", colname2=""):
    datadict = {}
    datadict["Number of training episodes"] = [dfs1['eels'].loc[:, "time_cons"].count(), dfs2['eels'].loc[:, "time_cons"].count()]
    datadict["Runtime in s"] = [dfs1['eels'].loc[:, "time_cons"].sum(), dfs2['eels'].loc[:, "time_cons"].sum()]
    datadict["Runtime in h"] = [dfs1['eels'].loc[:, "time_cons"].sum()/3600, dfs2['eels'].loc[:, "time_cons"].sum()/3600]
    datadict["Mean episode runtime in s"] = [dfs1['eels'].loc[:, "time_cons"].mean(), dfs2['eels'].loc[:, "time_cons"].mean()]

    datadict["Mean episode runtime during eval. episode in s"] = [dfs1['eels'].loc[dfs1['eels'].loc[:, "eval_epoch"] == "True", "time_cons"].mean(),
                                                                 dfs2['eels'].loc[dfs2['eels'].loc[:, "eval_epoch"]  == "True", "time_cons"].mean()]
    datadict["Mean episode runtime after eval. episode in s"] = [dfs1['eels'].loc[dfs1['eels'].loc[:, "eval_epoch"].shift(1) == "True", "time_cons"].mean(),
                                                                 dfs2['eels'].loc[dfs2['eels'].loc[:, "eval_epoch"].shift(1) == "True", "time_cons"].mean()]
    datadict["Mean episode runtime in no eval. episode in s"] = [dfs1['eels'].loc[dfs1['eels'].loc[:, "eval_epoch"]  == "False", "time_cons"].mean(),
                                                                 dfs2['eels'].loc[dfs2['eels'].loc[:, "eval_epoch"]  == "False", "time_cons"].mean()]

    return pd.DataFrame.from_dict(datadict, orient='index', columns=[colname1, colname2])

def compute_last_available_eval_episode(alldfs):
    last_available_eval_episodes = []
    for idx, dfs in enumerate(alldfs):
        last_eval_episode = dfs['sees'].loc[:, "episode"].unique()[-1]
        last_available_eval_episodes.append(last_eval_episode)
        print(f"Last available evaluation episode for dfs{idx}: {last_eval_episode:6}")
    return last_available_eval_episodes

def print_reward_informations(alldfs, selected_episodes):
    print("First complete week of evaluation episode")
    for idx, dfs in enumerate(alldfs):
        econs = dfs['eels'].iloc[0]['sum_energy_Wh']/1000
        nstpc = dfs['eels'].iloc[0]['sum_manual_stp_ch_n'] if 'sum_manual_stp_ch_n' in dfs['eels'].columns else "?"
        print(f"For episode {selected_episodes[idx]:5}: Energy consumption: {econs:10.2f} kWh; Numer of setpoint changes: {nstpc:8}")
    print()
    print("Mean values for episode ...")
    for idx, dfs in enumerate(alldfs):
        mrwd  = dfs['eels'].iloc[0]['mean_reward']
        nstpc = dfs['eels'].iloc[0]['mean_manual_stp_ch_n']
        econs = dfs['eels'].iloc[0]['mean_energy_Wh']
        print(f"... {selected_episodes[idx]:5}: Reward: {mrwd:8.5f} kWh; Setpoint change magnit.: {nstpc:8.5f}; Energy cons.: {econs:8.5f}")

def get_available_rooms_and_agents(alldfs):
    subdfs_rooms  = []
    subdfs_agents = []
    for idx, sdfs in enumerate(alldfs):
        sdfs_rooms  = sdfs["seeser"].loc[:, "room"].unique()
        sdfs_agents = sdfs["seesea"].loc[:, "agent_nr"].unique()
        subdfs_rooms.append(sdfs_rooms)
        subdfs_agents.append(sdfs_agents)
        print(f"Available Rooms     in (sub-)dfs{idx}: {sdfs_rooms}")
        print(f"Available Agent IDs in (sub-)dfs{idx}: {sdfs_agents}", "\n")
    return subdfs_rooms, subdfs_agents

def get_arguments_overview(colname1, colname2, checkpoint_dir1, checkpoint_dir2):
    def prepaire_output_dict(options_file):
        output_dict = {}
        for line in options_file:
            line2 = re.sub('\s+'," ",line)
            lsplit = line2.split(' ')
            default_changed = len(lsplit) > 2 and lsplit[2] == "[Default:"
            default = None if not default_changed else lsplit[3].replace(']','')
            output_dict[lsplit[0]] = [lsplit[1], default_changed, default]
        return output_dict
    def highlight_color(val):
        color_left = ""
        color_right= ""
        if val.iloc[1]:
            color_left = "background-color: orange"
        if val.iloc[3]:
            color_right = "background-color: orange"
        return [color_left, color_left, "", color_right, color_right]
    f1 = open(os.path.join(checkpoint_dir1, "options.txt"), "r")
    dict1 = prepaire_output_dict(f1.readlines())
    f1.close()
    f2 = open(os.path.join(checkpoint_dir2, "options.txt"), "r")
    dict2 = prepaire_output_dict(f2.readlines())
    f2.close()
    df1 = pd.DataFrame.from_dict(dict1, orient='index', columns=[colname1, colname1 + ' changed', 'default'])
    df2 = pd.DataFrame.from_dict(dict2, orient='index', columns=[colname2, colname2 + ' changed', 'default'])
    df2 = df2.loc[:, [colname2 + ' changed', colname2]]
    df  = df1.join(df2)
    return df.style.apply(highlight_color, axis=1)

def plot_lr_epsilon(dfs, ax):
    ax.plot( dfs['eels']["epsilon"], label="epsilon" )
    ax.plot( dfs['eels']["lr"],      label="learning rate" )
    ax.legend()

def plot_eels_reward(dfs, ax):
    evaluation_episodes = dfs['sees'].loc[:, "episode"].unique()

    dfs['eels'].loc[:,   "mean_reward"].plot(ax=ax[0], label="all episodes")
    ax[0].set_ylabel("Mean Reward")
    dfs['eels'].loc[:,   "sum_energy_Wh"].plot(ax=ax[1], label="all episodes")
    ax[1].set_ylabel("Energy consumption in Wh (for a complete episode)")
    dfs['eels'].loc[:,    "mean_manual_stp_ch_n"].plot(ax=ax[2], label="all episodes")
    ax[2].set_ylabel("Magnitude of manual setpoint changes")
    dfs['eels'].loc[evaluation_episodes, "mean_reward"].plot(ax=ax[0], label="evaluation episodes")
    dfs['eels'].loc[evaluation_episodes, "sum_energy_Wh"].plot(ax=ax[1], label="evaluation episodes")
    dfs['eels'].loc[ evaluation_episodes, "mean_manual_stp_ch_n"].plot(ax=ax[2], label="evaluation episodes")
    for i in range(3):
        ax[i].legend()
        ax[i].set_xlabel("Episode")

def plot_eels_agent_details(dfs, ax):
    dfs['eels'].loc[:, "loss_mean"].plot(ax=ax[0],   label="Mean critic loss per episode")
    dfs['eels'].loc[20:, "loss_mean"].plot(ax=ax[1], label="Mean critic loss per episode (from ep. 20 on)")
    
    if len(dfs['eels'].loc[:, "q_st2_mean"].unique()) <= 1:
        # only 0 values, nothing interesting
        dfs['eels'].loc[20:, "loss_mean"].plot(ax=ax[2], logy=True, label="Log Mean critic loss per episode (from ep. 20 on)")
        for i in range(3):
            ax[i].legend()
    else:
        dfs['eels'].loc[:, "q_st2_mean"].plot(ax=ax[2],  label="Mean Q-value per episode")
        dfs['eels'].loc[20:, "q_st2_mean"].plot(ax=ax[3],label="Mean Q-value per episode (from ep. 20 on)")
        for i in range(4):
            ax[i].legend()
    if len(dfs['eels'].loc[20:, "J_mean"].unique()) > 1:
        dfs['eels'].loc[20:, "J_mean"].plot(ax=ax[4],label="Mean J-value per episode (from ep. 20 on)")
        ax[4].legend()

def plot_eels_agent_details_frobnorm(dfs, ax, with_critics=True):
    idx_add = 0
    if with_critics:
        dfs['eels'].loc[:, "frobnorm_critic_matr_mean"].plot(ax=ax[0], title="mean Frobenius norm of critic FC-matricies")
        dfs['eels'].loc[:, "frobnorm_critic_bias_mean"].plot(ax=ax[1], title="mean Frobenius norm of critic FC-biases")
        idx_add = 2
    dfs['eels'].loc[:, "frobnorm_agent_matr_mean"].plot(ax=ax[0 + idx_add],  title="mean Frobenius norm of agent FC-matricies")
    dfs['eels'].loc[:, "frobnorm_agent_bias_mean"].plot(ax=ax[1 + idx_add],  title="mean Frobenius norm of agent FC-biases")

def plot_sees_only_mstpc(dfs, ax):
    dfs["sees"].loc[:, "manual_stp_ch_n"].plot(ax=ax)
    ax.legend()

def plot_sees_reward(dfs, ax):
    l1 = dfs['sees'].loc[:,   "reward"].plot(ax=ax[0], label="Reward", color="tab:purple")
    ax[0].set_ylabel("Reward")
    l2 = dfs['sees'].loc[:,   "energy_Wh"].plot(ax=ax[1], label="Energy consumption in Wh", color="tab:olive")
    ax[1].set_ylabel("Wh")
    l3 = dfs['sees'].loc[:,    "manual_stp_ch_n"].plot(ax=ax[2], label="Magnitude of manual setpoint changes", color="tab:cyan")
    for i in range(3):
        ax[i].set_xlabel("Epoche / Timestep")
        ax[i].legend()
    return l1, l2, l3

def plot_sees(dfs, ax):
    dfs["sees"]["outdoor_temp"].plot(ax=ax[0], label="outdoor_temp")
    dfs["sees"]["outdoor_humidity"].plot(ax=ax[1], label="outdoor_humidity")
    dfs["sees"]["outdoor_windspeed"].plot(ax=ax[2], label="outdoor_windspeed")
    dfs["sees"]["outdoor_winddir"].plot(ax=ax[3], label="outdoor_winddir")
    dfs["sees"]["outdoor_solar_radi_dir"].plot(ax=ax[4], label="outdoor_solar_radi_dir")
    dfs["sees"]["outdoor_solar_radi_indir"].plot(ax=ax[4], label="outdoor_solar_radi_indir")
    for i in range(1,5):
        ax[i].legend()

def plot_seeser(dfs, selected_room, ax):
    selroom = dfs["seeser"].loc[ dfs["seeser"].loc[:, "room"] == selected_room ]
    selroom.loc[:, "temp"].plot(ax=ax[0], label=f"Temp, R. {selected_room}")
    selroom.loc[:, "humidity"].plot(ax=ax[1], label=f"Humidity, R. {selected_room}")
    selroom.loc[:, "occupancy"].plot(ax=ax[2], label=f"Occupancy, R. {selected_room}")
    selroom.loc[:, "co2"].plot(ax=ax[3], label=f"CO2, R. {selected_room}")
    for i in range(4):
        ax[i].legend()

def plot_seeser_all_rooms(dfs, ax):
    for room in dfs["seeser"].loc[:, "room"].unique():
        plot_seeser(dfs, room, ax)

def plot_seesea(dfs, ax=None):
    if len(dfs["seesea"].loc[:, "agent_nr"].unique()) == 1:
        # single agent
        selected_agent = dfs["seesea"].loc[:, "agent_nr"].unique()[0]
        plot_seesea_single_agent(dfs, selected_agent, ax)
    else:
        # multi agent
        offset = 0
        for selected_agent in dfs["seesea"].loc[:, "agent_nr"].unique():
            offset += plot_seesea_single_agent(dfs, selected_agent, ax[offset:])

def plot_seesea_single_agent(dfs, selected_agent, ax=None):
    actions_dict_df = dfs["seesea"].loc[ dfs["seesea"].loc[:, "agent_nr"] == selected_agent ]
    actions_df = pd.DataFrame([ ast.literal_eval( action )
                                for action
                                in actions_dict_df.reset_index().loc[:, "agent_actions"]
                              ])
    actions_df.index = actions_dict_df.index
    sum_of_outputs = 0
    if ax is None:
        actions_df.plot(subplots=True, figsize=(7,15))
        sum_of_outputs = 1
    else:
        for idx, col in enumerate(actions_df.columns):
            actions_df.loc[:, [col]].plot(ax=ax[idx])
            sum_of_outputs += 1
    return sum_of_outputs

def plot_room_temp_agent_setpoint(dfs, room, agentid, ax, fill_between = False):
    dfs_room  = dfs["seeser"].loc[ dfs["seeser"].loc[:, "room"] == room ]
    dfs_agent_dict = dfs["seesea"].loc[ dfs["seesea"].loc[:, "agent_nr"] == agentid ]
    dfs_agnet = pd.DataFrame([ ast.literal_eval( action )
                                for action
                                in dfs_agent_dict.reset_index().loc[:, "agent_actions"]
                              ])
    dfs_agnet.index = dfs_agent_dict.index
    if f"{room} Zone Heating/Cooling-Mean Setpoint" in dfs_agnet.columns:
        dfs_agent_mean  = dfs_agnet.loc[:, f"{room} Zone Heating/Cooling-Mean Setpoint"]
        dfs_agent_delta = dfs_agnet.loc[:, f"{room} Zone Heating/Cooling-Delta Setpoint"]
    elif "Zone Heating/Cooling-Mean Setpoint" in dfs_agnet.columns:
        dfs_agent_mean  = dfs_agnet.loc[:, "Zone Heating/Cooling-Mean Setpoint"]
        dfs_agent_delta = dfs_agnet.loc[:, "Zone Heating/Cooling-Delta Setpoint"]
    else:
        dfs_agent_mean  = dfs_agnet.loc[:, "Zone Heating Setpoint"]
        dfs_agent_delta = 0

    if "dfs_agent_delta" in locals().keys():
        dfs_room["temp"].plot(ax=ax, label="Real room temperature")
        (dfs_agent_mean + dfs_agent_delta).plot(ax=ax, label="Agent upper setpoint bound")
    else:
        (dfs_agent_mean - dfs_agent_delta).plot(ax=ax, label="Agent heating setpoint")

    current_ylim = ax.get_ylim()
    dfs_room_ttemp = dfs_room["target_temp"]
    dfs_room_ttemp.plot(ax=ax, label="Target room temperature", color="tab:red")
    if fill_between:
        ax.fill_between(dfs_room_ttemp.index, dfs_room_ttemp-1, dfs_room_ttemp+1, alpha=0.5, color="red")
    ax.set_ylim(current_ylim)
    ax.set_ylabel(room)





#
# below here there are functions, that do not create individual plots,
# but they create a complete group of plots
#


def complete_plot_epsilon(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=1, ncols=len(alldfs), figsize=(fig_width,4), sharex=False)
    for idx, dfs in enumerate(alldfs):
        plot_lr_epsilon(dfs, axes[idx])
    return pl, axes


def complete_plot_reward_stpc_econs(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=3, ncols=len(alldfs), figsize=(fig_width,10), sharex=False)
    for idx, dfs in enumerate(alldfs):
        plot_eels_reward(dfs, axes[:, idx])
    return pl, axes


def complete_plot_losses(alldfs, fig_width, with_agents=True):
    if with_agents:
        pl, axes = plt.subplots(nrows=5, ncols=len(alldfs), figsize=(fig_width,8), sharex=True)
    else:
        pl, axes = plt.subplots(nrows=3, ncols=len(alldfs), figsize=(fig_width,8), sharex=True)
    for idx, dfs in enumerate(alldfs):
        plot_eels_agent_details(dfs, axes[:, idx])
    return pl, axes


def complete_plot_frobenius_norms(alldfs, fig_width, with_critics=True):
    if with_critics:
        pl, axes = plt.subplots(nrows=4, ncols=len(alldfs), figsize=(fig_width,6))
    else:
        pl, axes = plt.subplots(nrows=2, ncols=len(alldfs), figsize=(fig_width,5))
    for idx, dfs in enumerate(alldfs):
        plot_eels_agent_details_frobnorm(dfs, axes[:, idx], with_critics)
    return pl, axes


def complete_plot_weather_information(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=5, ncols=len(alldfs), figsize=(fig_width,6), sharex=True)
    for idx, dfs in enumerate(alldfs):
        plot_sees(dfs, axes[:, idx])
    return pl, axes


def complete_plot_number_of_stp_ch(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=1, ncols=len(alldfs), figsize=(fig_width,2))
    for idx, dfs in enumerate(alldfs):
        plot_sees_only_mstpc(dfs, axes[idx])
    return pl, axes


def complete_plot_room_status(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=4, ncols=len(alldfs), figsize=(fig_width,12), sharex=True)
    for idx, dfs in enumerate(alldfs):
        plot_seeser_all_rooms(dfs, axes[:, idx])
    # for individual rooms use plot_seeser(subdfs, room_id, ax)
    return pl, axes


def complete_plot_all_agent_outputs(alldfs, fig_width, subdfs_agents):
    max_n_agents = max(2, max([len(sdfs_agents) for sdfs_agents in subdfs_agents]))
    nrows = max_n_agents * max([sdfs['seesea'].iloc[0]["agent_actions"].count(":") for sdfs in alldfs])
    pl, axes = plt.subplots(nrows=nrows, ncols=len(alldfs), figsize=(fig_width,nrows), sharex=True)
    for a in axes.flatten():
        a.ticklabel_format(useOffset=False, style='plain')
    for idx, sdfs in enumerate(alldfs):
        plot_seesea(sdfs, axes[:, idx])
    return pl, axes


def complete_plot_total_overview(subdfs, fig_width, subdfs_rooms, subdfs_agents):
    if not type(subdfs) is list:
        subdfs       = [subdfs]
        subdfs_rooms = [subdfs_rooms]
        subdfs_agents= [subdfs_agents]

    max_n_agents = max(2, max([len(sdfs_agents) for sdfs_agents in subdfs_agents]))
    p, axes = plt.subplots(nrows=max_n_agents+3, ncols=len(subdfs), figsize=(fig_width,3*max_n_agents), sharex=True)
    legend_handles = []
    legend_labels  = []

    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]

    # plot rewards
    for idx, dfs in enumerate(subdfs):
        plot_sees_reward(subdfs[idx], axes[:, idx])

    # plot for every room
    for idx, sdfs in enumerate(subdfs):
        for idx2, room, agentid in zip(range(len(subdfs_rooms[idx])), subdfs_rooms[idx], subdfs_agents[idx]):
            idx2offset = idx2+3
            plot_room_temp_agent_setpoint(sdfs, room, agentid, axes[idx2offset, idx], True)
            handles, labels = axes[idx2offset, idx].get_legend_handles_labels()
            legend_handles.extend(handles)
            legend_labels.extend(labels)

    p.legend(handles, labels, loc='lower center', ncol=2)
    #p.subplots_adjust(right=0.7)
    return p, axes


def complete_plot_(alldfs, fig_width):
    return pl, axes


