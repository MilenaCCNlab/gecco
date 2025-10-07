import pandas as pd
import numpy as np

data = pd.read_csv('feher2020_exp1.csv')


dfs = []


for a in data.participant.unique():



    data_agent = data[data.participant == a].reset_index()
    data_agent["trial"] = data_agent["trial"] // 2  # always two consecutive steps make one trial
    data_agent["reward"] = data_agent["reward"].astype(int)

    carpet_choice = np.array(
        [data_agent[data_agent.trial == t].reset_index().choice.values[0] for t in data_agent.trial.unique()])
    mountain = np.array(
        [data_agent[data_agent.trial == t].reset_index().current_state.values[1] for t in data_agent.trial.unique()])
    genie_choice = np.array(
        [data_agent[data_agent.trial == t].reset_index().choice.values[1] for t in data_agent.trial.unique()])
    rewards = np.array(
        [data_agent[data_agent.trial == t].reset_index().reward.values[1] for t in data_agent.trial.unique()])
    trials = list(data_agent.trial.unique())

    reward0_0p = np.array(
        [data_agent[data_agent.trial == t].reset_index()["reward.0.0"].values[1] for t in data_agent.trial.unique()])
    reward0_1p = np.array(
        [data_agent[data_agent.trial == t].reset_index()["reward.0.1"].values[1] for t in data_agent.trial.unique()])
    reward1_0p = np.array(
        [data_agent[data_agent.trial == t].reset_index()["reward.1.0"].values[1] for t in data_agent.trial.unique()])
    reward1_1p = np.array(
        [data_agent[data_agent.trial == t].reset_index()["reward.1.1"].values[1] for t in data_agent.trial.unique()])


    df = pd.DataFrame({'participant':[a]*len(trials),
                       'trial': trials,
                       'choice_1':carpet_choice,
                       'state':mountain,
                       'choice_2':genie_choice,
                       'reward':rewards,
                       'reward_p_s0_0':reward0_0p,
                       'reward_p_s0_1':reward0_1p,
                       'reward_p_s1_0':reward1_0p,
                       'reward_p_s1_1':reward1_1p})

    dfs.append(df)


large_df = pd.concat(dfs).reset_index()
large_df = large_df.drop(columns='index')
large_df.to_csv('two_step_data.csv')