import os
import glob
from world import World
from names_dataset import NameDataset
import numpy as np
import pandas as pd

# ————— Set up checkpoint file list —————
checkpoint_dir = "checkpoint/run-1"
pattern = os.path.join(checkpoint_dir, "GABM_R03_*.pkl")
checkpoint_files = sorted(glob.glob(pattern))

# ————— Prepare NameDataset once for gender + rank mapping —————
nd = NameDataset()
# top s names (s must be even)
s = 2000
if s % 2 == 1:
    s += 1
male_top   = nd.get_top_names(s//2, 'Male',   'US')['US']['M']
female_top = nd.get_top_names(s//2, 'Female', 'US')['US']['F']
male_rank  = {n: r+1 for r, n in enumerate(male_top)}
female_rank= {n: r+1 for r, n in enumerate(female_top)}

# ————— Container for all runs —————
df_all = []

for run_number, ckpt in enumerate(checkpoint_files, start=1):
    print(f"Processing run {run_number} → {ckpt}")
    model = World.load_checkpoint(ckpt)

    # ———— 1) responses_over_time.csv for this run ————
    mems = [agent.mems for agent in model.schedule]
    df_resp = pd.DataFrame(mems)
    df_resp.to_csv(f"responses_over_time_run_{run_number}.csv", index=False)

    # ———— 2) stats_for_agents.csv for this run ————
    try:
        df_stats = model.datacollector.get_model_vars_dataframe()
    except AttributeError:
        df_stats = pd.DataFrame(model.datacollector.model_vars)

    # compute "Daily New Cases Day 4" as a PROPORTION of pop
    day4 = np.array(model.day_infected_is_4[:len(df_stats)])
    df_stats['Daily New Cases Day 4'] = day4 #/ model.population

    df_stats.to_csv(f"stats_for_agents_run_{run_number}.csv", index=False)

    # ———— 3) build the long form per-agent/per-timestep table ————
    ts_cols = sorted(c for c in df_resp.columns if isinstance(c, int))
    N_agents = len(df_resp)
    N_steps  = len(ts_cols)

    responses   = []
    health_strs = []
    tsteps      = []
    runs        = []
    for ts in ts_cols:
        step_df = pd.json_normalize(df_resp[ts])
        resp_bin = step_df['response'].str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int).tolist()
        responses.extend(resp_bin)
        health_strs.extend(step_df['health string'].tolist())
        tsteps.extend([ts] * N_agents)
        runs.extend([run_number] * N_agents)

    # static bio: name, age, gender, normalized name-rank
    names = df_resp['name'].tolist()
    ages  = df_resp['age'].tolist()
    genders = []
    name_ranks = []
    for nm in names:
        if nm in male_rank:
            genders.append(1)
            name_ranks.append(male_rank[nm])
        elif nm in female_rank:
            genders.append(0)
            name_ranks.append(female_rank[nm])
        else:
            genders.append(None)
            name_ranks.append(None)
    name_ranks = np.array(name_ranks, dtype=float)
    name_ranks = 1 + 1/(s/2) - (name_ranks/(s/2))

    trait_lists = df_resp['traits'].tolist()
    df_traits = pd.DataFrame(trait_lists,
                             columns=[
                                'extraversion',
                                'agreeableness',
                                'conscientiousness',
                                'emotional_stability',
                                'open_to_experience',
                             ])
    df_traits_bin = pd.DataFrame({
        'Extraversion':        df_traits['extraversion']       .eq('extroverted')         .astype(int),
        'Agreeableness':       df_traits['agreeableness']      .eq('agreeable')           .astype(int),
        'Conscientiousness':   df_traits['conscientiousness']  .eq('conscientious')       .astype(int),
        'Emotional Stability': df_traits['emotional_stability'].eq('emotionally stable') .astype(int),
        'Openness to Experience': df_traits['open_to_experience'].eq('open to experience').astype(int)
    })

    df_static = pd.DataFrame({
        'name': names,
        'age': ages,
        'gender': genders,
        'Name Rank': name_ranks
    })
    df_static = pd.concat([df_static, df_traits_bin], axis=1)
    df_static_rep = pd.concat([df_static] * N_steps, ignore_index=True)

    df_dyn = pd.DataFrame({
        'Run Number': runs,
        'Time Step':  tsteps,
        'Response':   responses,
        'Health String': health_strs
    })
    df_dyn['slight_cough']     = df_dyn['Health String'].str.contains('slight cough').astype(int)
    df_dyn['cough_and_fever']  = df_dyn['Health String'].str.contains('cough and a fever').astype(int)
    df_dyn.drop(columns=['Health String'], inplace=True)

    cases4 = df_stats['Daily New Cases Day 4'].values
    df_cases_rep = pd.DataFrame({
        'Daily New Cases Day 4': np.repeat(cases4, N_agents)
    })

    df_run = pd.concat([
        df_static_rep.reset_index(drop=True),
        df_dyn.reset_index(drop=True),
        df_cases_rep.reset_index(drop=True)
    ], axis=1)

    df_all.append(df_run)

# concat all runs and write
(df_full_all := pd.concat(df_all, ignore_index=True))
df_full_all.to_csv("AllRuns_Indiv_Data_for_logistic_regression.csv", index=False)