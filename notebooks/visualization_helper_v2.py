
#
# New visualization helper
# For training runs without eees, eeesea tables
#

import os
import re
import ast
import pickle
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

def load_and_convert_q_values(dirnames):
    q_values = []
    for dirname in dirnames:
        fpath = os.path.join(dirname, "q_values.pickle")
        if os.path.exists(fpath):
            # load pickle file
            f = open(fpath, "rb")
            q_status = pickle.load(f)
            f.close()
            #
            q_vals = q_status["Q value list"]
            action_vals = q_status["Actions"][0]
            # convert to numpy array
            n_agents = len(q_vals)
            conv_q_vals = []
            for idx in range(n_agents):
                conv_q_vals.append(np.array(q_vals[idx]))
            q_values.append(conv_q_vals)
        else:
            print(f"No Q-value list found for {dirname}.")
            q_values.append([])
    return q_values

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

def get_runtime_overview_df(alldfs, dirnames):
    datadict = {}
    datadict["Number of training episodes"] = [dfs['eels'].loc[:, "time_cons"].count() for dfs in alldfs]
    datadict["Runtime in s"] = [dfs['eels'].loc[:, "time_cons"].sum() for dfs in alldfs]
    datadict["Runtime in h"] = [dfs['eels'].loc[:, "time_cons"].sum()/3600 for dfs in alldfs]
    datadict["Mean episode runtime in s"] = [dfs['eels'].loc[:, "time_cons"].mean() for dfs in alldfs]

    datadict["Mean episode runtime during eval. episode in s"] = [dfs['eels'].loc[dfs['eels'].loc[:, "eval_epoch"] == "True", "time_cons"].mean() for dfs in alldfs]
    datadict["Mean episode runtime after eval. episode in s"] = [dfs['eels'].loc[dfs['eels'].loc[:, "eval_epoch"].shift(1) == "True", "time_cons"].mean() for dfs in alldfs]
    datadict["Mean episode runtime in no eval. episode in s"] = [dfs['eels'].loc[dfs['eels'].loc[:, "eval_epoch"]  == "False", "time_cons"].mean() for dfs in alldfs]

    return pd.DataFrame.from_dict(datadict, orient='index', columns=[dname.split("/")[2] for dname in dirnames])

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

def get_arguments_overview(dirnames, only_with_different_vals = True):
    def prepaire_output_dict(options_file):
        output_dict = {}
        for line in options_file:
            line2 = re.sub('\s+'," ",line)
            lsplit = line2.split(' ')
            default_changed = len(lsplit) > 2 and lsplit[2] == "[Default:"
            default = None if not default_changed else lsplit[3].replace(']','')
            if lsplit[0] == "checkpoint_dir" or lsplit[0] == "idf_file" or lsplit[0] == "epw_file":
                continue
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

    argument_dfs = []
    for checkp_dirname in dirnames:
        fl = open(os.path.join(checkp_dirname, "options.txt"), "r")
        dict1 = prepaire_output_dict(fl.readlines())
        fl.close()
        scenario_name = checkp_dirname.split("/")[2]
        df1 = pd.DataFrame.from_dict(dict1, orient='index', columns=[scenario_name, scenario_name + ' changed', 'default'])
        df1 = df1.loc[:, [scenario_name]]
        argument_dfs.append(df1)

    fulldfs = argument_dfs[0]
    if len(argument_dfs) > 1:
        for idx, df2 in enumerate(argument_dfs[1:]):
            #df2_selected_cols = list(df2.columns)
            #df2_selected_cols.remove('default')
            #df2 = df2.loc[:, df2_selected_cols]
            fulldfs = fulldfs.join(df2, rsuffix="_"+str(idx+2))
    #fulldfs_cols = list(fulldfs.columns)
    #fulldfs_cols.remove('default')
    #return fulldfs.loc[:, fulldfs_cols]
    if only_with_different_vals:
        indexes_with_different_vals = []
        for idx, data in fulldfs.iterrows():
            if len(data.unique()) > 1:
                indexes_with_different_vals.append(idx)
        return fulldfs.loc[indexes_with_different_vals, :]
    return fulldfs
    #return df.style.apply(highlight_color, axis=1)

def plot_lr_epsilon(dfs, ax):
    ax.plot( dfs['eels']["epsilon"], label="epsilon" )
    ax.plot( dfs['eels']["lr"],      label="learning rate" )
    ax.legend()

def plot_eels_reward(dfs, ax):
    evaluation_episodes = dfs['sees'].loc[:, "episode"].unique()

    dfs['eels'].loc[:,   "mean_reward"].plot(ax=ax[0], label="all episodes")
    ax[0].set_ylabel("Mean Reward")
    dfs['eels'].loc[:,   "sum_energy_Wh"].plot(ax=ax[1], label="all episodes")
    ax[1].set_ylabel("Energy consumption in Wh\n(for a complete episode)")
    dfs['eels'].loc[:,    "mean_manual_stp_ch_n"].plot(ax=ax[2], label="all episodes")
    ax[2].set_ylabel("Magnitude of\nmanual setpoint changes")
    dfs['eels'].loc[evaluation_episodes, "mean_reward"].plot(ax=ax[0], label="evaluation episodes")
    dfs['eels'].loc[evaluation_episodes, "sum_energy_Wh"].plot(ax=ax[1], label="evaluation episodes")
    dfs['eels'].loc[ evaluation_episodes, "mean_manual_stp_ch_n"].plot(ax=ax[2], label="evaluation episodes")
    dfs['eels'].loc[evaluation_episodes, "mean_reward"].rolling(window=30, min_periods=0).mean().plot(ax=ax[0], label="rolling mean for evaluation episodes", color="tab:red")
    dfs['eels'].loc[evaluation_episodes, "sum_energy_Wh"].rolling(window=30, min_periods=0).mean().plot(ax=ax[1], label="rolling mean for evaluation episodes", color="tab:red")
    dfs['eels'].loc[evaluation_episodes, "mean_manual_stp_ch_n"].rolling(window=30, min_periods=0).mean().plot(ax=ax[2], label="rolling mean for evaluation episodes", color="tab:red")
    for i in range(3):
        #ax[i].legend()
        ax[i].set_xlabel("Episode")

def plot_eels_agent_details(dfs, ax):
    dfs['eels'].loc[:, "loss_mean"].plot(ax=ax[0],   label="Mean per episode", color="tab:orange")
    dfs['eels'].loc[20:, "loss_mean"].plot(ax=ax[1], label="Mean per episode (from ep. 20 on)", color="tab:olive")
    dfs['eels'].loc[:, "loss_mean"].rolling(window=30, min_periods=0).mean().plot(ax=ax[0],   label="Rolling mean per episode", color="tab:red")
    dfs['eels'].loc[20:, "loss_mean"].rolling(window=30, min_periods=0).mean().plot(ax=ax[1], label="Rolling mean per episode (from ep. 20 on)", color="tab:green")
    ax[0].set_ylabel("Mean critic loss")
    ax[1].set_ylabel("Mean critic loss")
    
    if len(dfs['eels'].loc[:, "q_st2_mean"].unique()) <= 1:
        # only 0 values, nothing interesting
        dfs['eels'].loc[20:, "loss_mean"].plot(ax=ax[2], logy=True, label="Mean per episode (from ep. 20 on)", color="tab:olive")
        dfs['eels'].loc[20:, "loss_mean"].rolling(window=30, min_periods=0).mean().plot(ax=ax[2], logy=True, label="Rolling mean per episode (from ep. 20 on)", color="tab:green")
        ax[2].set_ylabel("Log mean critic loss")
        #for i in range(3):
        #    ax[i].legend()
    else:
        dfs['eels'].loc[:, "q_st2_mean"].plot(ax=ax[2],  label="Mean Q-value per episode")
        dfs['eels'].loc[20:, "q_st2_mean"].plot(ax=ax[3],label="Mean Q-value per episode (from ep. 20 on)")
        #for i in range(4):
        #    ax[i].legend()
        ax[2].legend()
        ax[3].legend()
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
    selroom.loc[:, "temp"].plot(ax=ax[0], label=f"{selected_room}")
    selroom.loc[:, "humidity"].plot(ax=ax[1], label=f"{selected_room}")
    selroom.loc[:, "occupancy"].plot(ax=ax[2], label=f"{selected_room}")
    selroom.loc[:, "co2"].plot(ax=ax[3], label=f"{selected_room}")
    ax[0].set_ylabel("Temperature in C")
    ax[1].set_ylabel("Humidity in %")
    ax[2].set_ylabel("Occupancy")
    ax[3].set_ylabel("CO2 Concentration in ppm")

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
        #dfs_agent_delta = 0

    dfs_room["temp"].plot(ax=ax, label="Real room temperature")
    if "dfs_agent_delta" in locals().keys():
        (dfs_agent_mean + dfs_agent_delta).plot(ax=ax, label="Agent upper setpoint bound")
        (dfs_agent_mean - dfs_agent_delta).plot(ax=ax, label="Agent lower setpoint bound")
    else:
        dfs_agent_mean.plot(ax=ax, label="Agent heating setpoint")

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
    if not type(axes) is np.ndarray:
        axes = np.array([axes])
    for idx, dfs in enumerate(alldfs):
        plot_lr_epsilon(dfs, axes[idx])
    return pl, axes


def complete_plot_reward_stpc_econs(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=3, ncols=len(alldfs), figsize=(fig_width,10), sharex=False)
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
    for idx, dfs in enumerate(alldfs):
        plot_eels_reward(dfs, axes[:, idx])
        handles, labels = axes[0, idx].get_legend_handles_labels()
    pl.legend(handles, labels, loc='lower center', ncol=3)
    return pl, axes


def complete_plot_losses(alldfs, fig_width, with_agents=True):
    if with_agents:
        pl, axes = plt.subplots(nrows=5, ncols=len(alldfs), figsize=(fig_width,8), sharex=True)
    else:
        pl, axes = plt.subplots(nrows=3, ncols=len(alldfs), figsize=(fig_width,8), sharex=True)
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
    for idx, dfs in enumerate(alldfs):
        plot_eels_agent_details(dfs, axes[:, idx])
        handles, labels = axes[0, idx].get_legend_handles_labels()
        handles2,labels2= axes[1, idx].get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)
    pl.legend(handles, labels, loc='lower center', ncol=2)
    return pl, axes


def complete_plot_frobenius_norms(alldfs, fig_width, with_critics=True):
    if with_critics:
        pl, axes = plt.subplots(nrows=4, ncols=len(alldfs), figsize=(fig_width,6))
    else:
        pl, axes = plt.subplots(nrows=2, ncols=len(alldfs), figsize=(fig_width,5))
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
    for idx, dfs in enumerate(alldfs):
        plot_eels_agent_details_frobnorm(dfs, axes[:, idx], with_critics)
    return pl, axes


def complete_plot_weather_information(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=5, ncols=len(alldfs), figsize=(fig_width,6), sharex=True)
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
    for idx, dfs in enumerate(alldfs):
        plot_sees(dfs, axes[:, idx])
    return pl, axes


def complete_plot_number_of_stp_ch(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=1, ncols=len(alldfs), figsize=(fig_width,2))
    if not type(axes) is np.ndarray:
        axes = np.array([axes])
    for idx, dfs in enumerate(alldfs):
        plot_sees_only_mstpc(dfs, axes[idx])
    return pl, axes


def complete_plot_room_status(alldfs, fig_width):
    pl, axes = plt.subplots(nrows=4, ncols=len(alldfs), figsize=(fig_width,12), sharex=True)
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
    for idx, dfs in enumerate(alldfs):
        plot_seeser_all_rooms(dfs, axes[:, idx])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    pl.legend(handles, labels, loc='lower center', ncol=3)
    # for individual rooms use plot_seeser(subdfs, room_id, ax)
    return pl, axes


def complete_plot_all_agent_outputs(alldfs, fig_width, subdfs_agents):
    max_n_agents = max(2, max([len(sdfs_agents) for sdfs_agents in subdfs_agents]))
    nrows = max_n_agents * max([sdfs['seesea'].iloc[0]["agent_actions"].count(":") for sdfs in alldfs])
    pl, axes = plt.subplots(nrows=nrows, ncols=len(alldfs), figsize=(fig_width,nrows), sharex=True)
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
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


def plot_stpch_and_econs_distrib(subdfs, fig_width):
    if not type(subdfs) is list:
        subdfs       = [subdfs]
    p, axes = plt.subplots(nrows=3, ncols=len(subdfs), figsize=(fig_width,9))
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
    for idx, sdfs in enumerate(subdfs):
        sdfs["sees"].loc[:, "reward"].hist(bins=30, log=True, ax=axes[0,idx])
        sdfs["sees"].loc[:, "manual_stp_ch_n"].hist(bins=30, log=True, ax=axes[1,idx])
        sdfs["sees"].loc[:, "energy_Wh"].hist(bins=30, ax=axes[2,idx])
        econs_mean  = sdfs["sees"].loc[:, "energy_Wh"].mean()
        reward_mean = sdfs["sees"].loc[:, "reward"].mean()
        axes[2,idx].axvline(econs_mean, color='b')
        axes[0,idx].axvline(reward_mean, color='b')
        print(f"For plot number {idx+1}, mean energy consumption = {econs_mean:8.1f} Wh, mean reward = {reward_mean:7.4f}")
        axes[0,idx].set_ylabel("Log Count")
        axes[1,idx].set_ylabel("Log Count")
        axes[2,idx].set_ylabel("Count")
        axes[0,idx].set_xlabel("Reward for a single timestep")
        axes[1,idx].set_xlabel("Manual setpoint change magnitude\nfor a single timestep")
        axes[2,idx].set_xlabel("Energy consumption\nfor a single timestep")


def plot_q_values(q_values, fig_width):
    n_scenarios = len(q_values)
    n_rows = sum([len(q_agent_vals) for q_agent_vals in q_values])
    p, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(fig_width,2*n_rows), sharex=True)
    plot_idx = 0
    for scenario in range(n_scenarios):
        for agent_id in range( len(q_values[scenario]) ):
            im = axes[plot_idx].imshow( q_values[scenario][agent_id][:,0,:].T, aspect="auto" )
            axes[plot_idx].set_ylabel(f"Scenario {scenario}\nAgent {agent_id}")
            cbar = p.colorbar(im, extend='both', shrink=0.95, ax=axes[plot_idx])
            plot_idx += 1


