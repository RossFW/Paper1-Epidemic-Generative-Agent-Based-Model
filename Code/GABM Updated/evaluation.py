import os
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_simulation(model, args, run_number, output_path):
    """
    Evaluate simulation outputs from 'model' after the run is complete.

    This function:
      1. Exports agent-level data in two formats:
         - A "compact" CSV (one row per agent, using agent.mems)
         - An "expanded" CSV (one row per agent per timestep from agent.mems)
      2. Exports population-level data by:
         - Reading the DataCollector’s DataFrame (which includes columns such as DailyNewCases,
           TotalContacts, and Day4Infected)
         - Adjusting "New Infections" so that Day 0 includes initial infections and Day 1 subtracts them
         - Computing "Cumulative Infections" as the cumsum of new infections
         - Adding columns for "Daily New Cases Day 4", "Total Contact", and "Max Potential Contact Rate"
      3. Generates example plots (SIR and Home vs. Outside)
    """

    # Ensure the output folder exists.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # -------------------------------------------------
    # 1. Agent-level data (COMPACT)
    #    Each agent’s mems dictionary is stored in one row.
    # -------------------------------------------------
    mems_list = [agent.mems for agent in model.schedule]
    compact_df = pd.DataFrame(mems_list)
    column_map = {}
    for col in compact_df.columns:
    # If it's an integer, shift it by +1
        if isinstance(col, int):
            column_map[col] = col + 1
    compact_df.rename(columns=column_map, inplace=True)
    compact_csv = os.path.join(output_path, f"{args.name}-individual_data_run{run_number}_compact.csv")
    
    compact_df.to_csv(compact_csv, index=False)

    # -------------------------------------------------
    # 2. Agent-level data (EXPANDED)
    #    One row per agent per timestep stored in agent.mems.
    # -------------------------------------------------
    expanded_rows = []
    for agent in model.schedule:
        agent_id = agent.unique_id
        name = agent.name
        age = agent.age
        # Get traits either from mems or from the agent attribute
        traits = agent.mems.get("traits", agent.traits)
        # Create binary flags for the Big Five traits (example logic)
        introverted = 1 if "introverted" in traits[0].lower() else 0
        agreeable = 1 if "agreeable" in traits[1].lower() else 0
        conscientious = 0 if "unconscientious" in traits[2].lower() else 1
        emotionally_stable = 1 if "emotionally stable" in traits[3].lower() else 0
        open_to_experience = 1 if "open to experience" in traits[4].lower() else 0

        # Loop over each timestep (numeric keys) stored in mems.
        for k, daily_info in agent.mems.items():
            if not isinstance(k, int):
                continue  # skip non-numeric keys

            reasoning = daily_info.get("reasoning", "")
            response_str = daily_info.get("response", "").lower().strip()
            location_str = daily_info.get("location", "").lower().strip()
            health_str = daily_info.get("health condition", "")

            # Create binary flags for health conditions
            susceptible_flag = 1 if health_str == "Susceptible" else 0
            infected_flag = 1 if health_str == "Infected" else 0
            recovered_flag = 1 if health_str == "Recovered" else 0

            response_flag = 1 if response_str == "yes" else 0
            location_flag = 1 if location_str == "home" else 0

            row = {
                "Time_step": k+1,
                "agent_id": agent_id,
                "name": name,
                "age": age,
                "introverted": introverted,
                "agreeable": agreeable,
                "conscientious": conscientious,
                "emotionally_stable": emotionally_stable,
                "open_to_experience": open_to_experience,
                "Susceptible": susceptible_flag,
                "Infected": infected_flag,
                "Recovered": recovered_flag,
                "reasoning": reasoning,
                "response_str": response_str,
                "response_flag": response_flag,
                "location_str": location_str,
                "location_flag": location_flag
            }
            expanded_rows.append(row)
    expanded_df = pd.DataFrame(expanded_rows)
    expanded_csv = os.path.join(output_path, f"{args.name}-individual_data_run{run_number}_expansive.csv")
    expanded_df.to_csv(expanded_csv, index=False)

    ############################################
    # 3. Population-level Data (Simplified)
    ############################################

    # 1) Get the DataCollector DataFrame and rename the index to "Step"
    pop_df = model.datacollector.get_model_vars_dataframe().reset_index()

    pop_df.rename(columns={"index": "Step"}, inplace=True)

    # 2) If "DailyNewCases" is in our data, compute a simple cumulative sum
    #    that includes the initially infected
    if "DailyNewCases" in pop_df.columns:
        pop_df["Cumulative Infections"] = pop_df["DailyNewCases"].cumsum() + model.initial_infected

        # Optionally, define a "New Infections" column the same as "DailyNewCases"
        # and adjust day 0 if desired.
        new_infections = pop_df["DailyNewCases"].copy()
        if len(new_infections) > 0:
            new_infections.iloc[0] += model.initial_infected
        pop_df["New Infections"] = new_infections

    # 3) If "Day4Infected" is there, rename or copy it to something more readable
    if "Day4Infected" in pop_df.columns:
        # Create a new column that is shifted by +1 row
        pop_df["# Day 4 New Cases"] = pop_df["Day4Infected"].shift(1, fill_value=0)
        # Then drop or ignore the original 'Day4Infected' column if you want
        pop_df.drop(columns=["Day4Infected"], inplace=True)
        

    # 4) Create a friendlier "# Contacts" column from "TotalContacts"
    if "TotalContacts" in pop_df.columns:
        pop_df["# Contacts"] = pop_df["TotalContacts"]

    # 5) Create "Max # of Potential Contact" if you want
    pop_df["Max # of Potential Contact"] = model.population * args.contact_rate

    # 6) Reorder columns (only keep those that exist)
    desired_order = [
        "Step", 
        "Susceptible", 
        "Infected", 
        "Recovered",
        "# Home", 
        "# Outside", 
        "New Infections", 
        "Cumulative Infections",
        "# Day 4 New Cases", 
        "# Contacts", 
        "Max # of Potential Contact"
    ]
    existing_order = [col for col in desired_order if col in pop_df.columns]
    pop_df = pop_df[existing_order]

    # 7) Write CSV
    pop_csv = os.path.join(output_path, f"{args.name}-population_data_run{run_number}.csv")
    pop_df.to_csv(pop_csv, index=False)


    # -------------------------------------------------
    # 4. Generate Example Plots
    # -------------------------------------------------
    # SIR Plot
    plt.figure(figsize=(10, 6))
    plt.plot(pop_df["Step"], pop_df["Susceptible"], label="Susceptible")
    plt.plot(pop_df["Step"], pop_df["Infected"], label="Infected")
    plt.plot(pop_df["Step"], pop_df["Recovered"], label="Recovered")
    plt.xlabel("Step")
    plt.ylabel("Number of Agents")
    plt.title(f"SIR - {args.name} Run {run_number}")
    plt.legend()
    plt.tight_layout()
    sir_plot = os.path.join(output_path, f"{args.name}-SIR_run{run_number}.png")
    plt.savefig(sir_plot, bbox_inches="tight")
    plt.close()

    # Home vs. Outside Plot
    plt.figure(figsize=(10, 6))
    # If these columns exist in pop_df, plot them.
    if "# Outside" in pop_df.columns:
        plt.plot(pop_df["Step"], pop_df["# Outside"], label="Citizens Outside")
    if "# Home" in pop_df.columns:
        plt.plot(pop_df["Step"], pop_df["# Home"], label="Citizens at Home")
    plt.xlabel("Step")
    plt.ylabel("Number of Agents")
    plt.title(f"Home vs. Outside - {args.name} Run {run_number}")
    plt.legend()
    plt.tight_layout()
    home_plot = os.path.join(output_path, f"{args.name}-Home_vs_Outside_run{run_number}.png")
    plt.savefig(home_plot, bbox_inches="tight")
    plt.close()

    print(f"Evaluation for run {run_number} complete.")
    print(f"Files saved in {output_path}:\n"
          f"  • {os.path.basename(compact_csv)} (Agent compact)\n"
          f"  • {os.path.basename(expanded_csv)} (Agent expanded)\n"
          f"  • {os.path.basename(pop_csv)} (Population-level)\n"
          f"  • {os.path.basename(sir_plot)}\n"
          f"  • {os.path.basename(home_plot)}\n")
