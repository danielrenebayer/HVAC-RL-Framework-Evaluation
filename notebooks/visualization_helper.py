
import os
import ast
import sqlite3
from datetime import timedelta

import pandas as pd

tables = ["eees", "eeesea", "eels", "sees", "seesea", "sees_er"]

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
    subdfs["eees"]   = dfs["eees"].loc[    dfs["eees"].loc[:,"episode"]    == selected_episode ].set_index("step")
    subdfs["eeesea"] = dfs["seesea"].loc[  dfs["seesea"].loc[:,"episode"]  == selected_episode ].set_index("step")
    subdfs["eels"]   = dfs["eels"].loc[    dfs["eels"].loc[:,"episode"]    == selected_episode ]
    subdfs["sees"]   = dfs["sees"].loc[    dfs["sees"].loc[:,"episode"]    == selected_episode ].set_index("step")
    subdfs["seesea"] = dfs["seesea"].loc[  dfs["seesea"].loc[:,"episode"]  == selected_episode ].set_index("step")
    subdfs["seeser"] = dfs["sees_er"].loc[ dfs["sees_er"].loc[:,"episode"] == selected_episode ].set_index("step")

    subdfs["sees"]["datetime"]   = pd.to_datetime(subdfs["sees"]["datetime"])
    subdfs["eees"]["datetime"]   = subdfs["sees"]["datetime"]
    subdfs["seesea"]["datetime"] = subdfs["sees"]["datetime"]
    subdfs["seeser"]["datetime"] = subdfs["sees"]["datetime"]
    if end is None:
        end = start + timedelta(days=7)
    subdfs["eees"]   = subdfs["eees"].loc[   subdfs["eees"]["datetime"]   >= start ]
    subdfs["sees"]   = subdfs["sees"].loc[   subdfs["sees"]["datetime"]   >= start ]
    subdfs["seesea"] = subdfs["seesea"].loc[ subdfs["seesea"]["datetime"] >= start ]
    subdfs["seeser"] = subdfs["seeser"].loc[ subdfs["seeser"]["datetime"] >= start ]
    subdfs["eees"]   = subdfs["eees"].loc[   subdfs["eees"]["datetime"]   <= end ]
    subdfs["sees"]   = subdfs["sees"].loc[   subdfs["sees"]["datetime"]   <= end ]
    subdfs["seesea"] = subdfs["seesea"].loc[ subdfs["seesea"]["datetime"] <= end ]
    subdfs["seeser"] = subdfs["seeser"].loc[ subdfs["seeser"]["datetime"] <= end ]

    subdfs["eees"].set_index(  "datetime", inplace=True)
    subdfs["sees"].set_index(  "datetime", inplace=True)
    subdfs["seesea"].set_index("datetime", inplace=True)
    subdfs["seeser"].set_index("datetime", inplace=True)

    return subdfs

def plot_eees(dfs, ax):
    eees_mean = dfs["eees"].groupby('episode').mean()
    eees_sum  = dfs["eees"].groupby('episode').sum()

    dfs["eees"].loc[:, "reward"].plot(ax=ax[0], label="Reward, for each step")
    eees_mean.loc[:,   "reward"].plot(ax=ax[1], label="Reward, mean for each episode")
    dfs["eees"].loc[:, "energy_Wh"].plot(ax=ax[2], label="Energy consumption in Wh, for each step")
    eees_mean.loc[:,   "energy_Wh"].plot(ax=ax[3], label="Energy consumption in Wh, mean for each episode")
    dfs["eees"].loc[:, "manual_stp_ch_n"].plot(ax=ax[4], label="Number of manual setpoint changes, for each step")
    eees_sum.loc[:,    "manual_stp_ch_n"].plot(ax=ax[5], label="Number of manual setpoint changes, sum for each episode")
    for i in range(6):
        ax[i].legend()

def plot_eeesea(dfs, ax):
    if len(dfs["eeesea"].index) == 0:
        print("No data to plot")
        return

    eeesea_mean = dfs["eeesea"].groupby('episode').mean()

    eeesea_mean.loc[:, "loss"].plot(ax=ax[0],   label="Mean critic loss per episode")
    eeesea_mean.loc[20:, "loss"].plot(ax=ax[1], label="Mean critic loss per episode (from ep. 20 on)")
    eeesea_mean.loc[:, "q_st2"].plot(ax=ax[2],  label="Mean Q-value per episode")
    eeesea_mean.loc[20:, "q_st2"].plot(ax=ax[3],label="Mean Q-value per episode (from ep. 20 on)")
    for i in range(4):
        ax[i].legend()
    if "J" in eeesea_mean.columns:
        eeesea_mean.loc[20:, "J"].plot(ax=ax[4],label="Mean J-value per episode (from ep. 20 on)")
        ax[4].legend()

def plot_eeesea_frobnorm(dfs, ax):
    if len(dfs["eeesea"].index) == 0:
        print("No data to plot")
        return

    eeesea_mean = dfs["eeesea"].groupby('episode').mean()

    eeesea_mean.loc[:, "frobnorm_critic_matr"].plot(ax=ax[0], title="mean Frobenius norm of critic FC-matricies")
    eeesea_mean.loc[:, "frobnorm_critic_bias"].plot(ax=ax[1], title="mean Frobenius norm of critic FC-biases")
    eeesea_mean.loc[:, "frobnorm_agent_matr"].plot(ax=ax[2],  title="mean Frobenius norm of agent FC-matricies")
    eeesea_mean.loc[:, "frobnorm_agent_bias"].plot(ax=ax[3],  title="mean Frobenius norm of agent FC-biases")

def plot_eees_only_mstpc(dfs, ax):
    dfs["eees"].loc[:, "manual_stp_ch_n"].plot(ax=ax)
    ax.legend()

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

def plot_room_temp_agent_setpoint(dfs, roomid, roomnumber, agentid, ax):
    dfs_room  = dfs["seeser"].loc[ dfs["seeser"].loc[:, "room"] == roomid ]
    dfs_agent_dict = dfs["seesea"].loc[ dfs["seesea"].loc[:, "agent_nr"] == agentid ]
    dfs_agnet = pd.DataFrame([ ast.literal_eval( action )
                                for action
                                in dfs_agent_dict.reset_index().loc[:, "agent_actions"]
                              ])
    dfs_agnet.index = dfs_agent_dict.index
    if f"SPACE{roomnumber}-1 Zone Heating/Cooling-Mean Setpoint" in dfs_agnet.columns:
        dfs_agent_mean  = dfs_agnet.loc[:, f"SPACE{roomnumber}-1 Zone Heating/Cooling-Mean Setpoint"]
        dfs_agent_delta = dfs_agnet.loc[:, f"SPACE{roomnumber}-1 Zone Heating/Cooling-Delta Setpoint"]
    else:
        dfs_agent_mean  = dfs_agnet.loc[:, "Zone Heating/Cooling-Mean Setpoint"]
        dfs_agent_delta = dfs_agnet.loc[:, "Zone Heating/Cooling-Delta Setpoint"]

    dfs_room["temp"].plot(ax=ax, label=f"SPACE{roomnumber}-1 Real temperature")
    (dfs_agent_mean + dfs_agent_delta).plot(ax=ax, label="Upper setpoint bound")
    (dfs_agent_mean - dfs_agent_delta).plot(ax=ax, label="Lower setpoint bound")
    if not ax is None: ax.legend()



