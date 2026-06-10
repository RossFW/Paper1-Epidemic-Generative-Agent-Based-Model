import math
import time
import pickle
import numpy as np
import concurrent.futures
from tqdm import tqdm

from agent import Agent
from datacollector import DataCollector
from utils import (
    generate_age, generate_names, generate_big5_traits,
    probability_threshold, update_day, clear_cache
)

# DataCollector helper functions
def compute_num_susceptible(model):
    return sum(a.health_condition == "Susceptible" for a in model.schedule)

def compute_num_infected(model):
    return sum(a.health_condition == "Infected" for a in model.schedule)

def compute_num_recovered(model):
    return sum(a.health_condition == "Recovered" for a in model.schedule)

def compute_num_outside(model):
    return sum(a.location == "outside" for a in model.schedule)

def compute_num_at_home(model):
    return sum(a.location == "home" for a in model.schedule)

def get_daily_new_cases(model):
    return model.daily_new_cases

def get_total_contacts(model):
    return model.total_contacts_today

def get_day4_infected(model):
    return model.day_4_infected_today


class World:
    """
    Example 'World' class that:
     - Collects a 'Day 0' row of initial conditions
     - Resets daily counters in each step
     - Collects daily data immediately after step
     - Uses concurrency for agent location decisions
     - Implements an offset for checkpoint loading
     - Implements early stopping if no infected remain
    """

    def __init__(self, args):
        """
        Initialize the World with the specified arguments.
        """

        # Basic parameters from command-line or defaults
        self.name = args.name
        self.step_count = args.no_days
        self.offset = args.offset

        #Basic infection spread parameters
        self.contact_rate = args.contact_rate
        self.infection_rate = args.infection_rate

        # Population setup
        self.initial_healthy = args.no_init_healthy
        self.initial_infected = args.no_init_infect
        self.population = self.initial_healthy + self.initial_infected

        # We'll keep agents in a list or array
        self.schedule = []
        # Keep track of who is outside
        self.agents_outside = []
        # Current time step (days since start)
        self.time_step = 0

        # ----- DAILY COUNTERS -----
        # These are reset at the start of each step:
        self.daily_new_cases = 0
        self.total_contacts_today = 0
        self.day_4_infected_today = 0
        self.yesterday_day_4_infected = 0

        # For early stopping: track how many are infected
        self.currently_infected = self.initial_infected  # will be updated in step()

        # Create the DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": compute_num_susceptible,
                "Infected": compute_num_infected,
                "Recovered": compute_num_recovered,
                "# Home": compute_num_at_home,
                "# Outside": compute_num_outside,
                "DailyNewCases": get_daily_new_cases,
                "TotalContacts": get_total_contacts,
                "Day4Infected": get_day4_infected,
            }
        )

        # ----- Create Agents -----
        names = generate_names(self.population, self.population * 2)
        agent_id = 0
        for i in range(self.population):
            # Everyone starts outside by default
            location = "outside"

            # Decide if healthy or infected
            if i < self.initial_healthy:
                health_condition = "Susceptible"
                day_infected = None
            else:
                health_condition = "Infected"
                day_infected = 1

            # Create the agent object
            citizen = Agent(
                model=self,
                unique_id=agent_id,
                name=names[i],
                age=generate_age(),
                traits=generate_big5_traits(),
                location=location,
                health_condition=health_condition,
                day_infected=day_infected
            )
            self.schedule.append(citizen)
            agent_id += 1

        # Convert to numpy array for convenience (optional)
        self.schedule = np.array(self.schedule, dtype=object)
        # Also build self.agents_outside for the first day
        self.agents_outside = [a for a in self.schedule if a.location == "outside"]


    def decide_agent_interactions(self):
        """
        Decide who interacts with whom among the agents outside.
        The final contact assignments are stored in each agent's 'agent_interaction' list.
        """
        # If only some fraction of the population is outside, scale contact rate
        fraction_outside = len(self.agents_outside) / self.population
        effective_rate = fraction_outside * self.contact_rate
        prob, base_int = math.modf(effective_rate)

        # Shuffle for randomness
        np.random.shuffle(self.agents_outside)

        # Assign each agent's contact rate
        for agent in self.agents_outside:
            # e.g. if effective_rate=3.6 => base_int=3, prob=0.6 => 60% chance for an extra contact
            agent.indiv_contact_rate = base_int + (1 if probability_threshold(prob) else 0)

        # Actually pair them up
        for agent in self.agents_outside:
            # Build a candidate list of outside agents not yet in agent.agent_interaction
            potential_list = [
                a for a in self.agents_outside
                if a is not agent and (a not in agent.agent_interaction)
            ]
            # Now pick from that list until agent.indiv_contact_rate is reached or we run out
            while len(agent.agent_interaction) < agent.indiv_contact_rate and potential_list:
                other_agent = np.random.choice(potential_list)
                agent.add_agent_interaction(other_agent)
                potential_list.remove(other_agent)


    def step(self):
        """
        Run one "day" of the model:
          1. Reset daily counters
          2. Agents decide location concurrently
          3. Build interaction pairs
          4. Infect
          5. Update day_infected
          6. Count day-4 infected
          7. time_step++
        """

        # 1. Reset daily counters
        self.daily_new_cases = 0
        self.total_contacts_today = 0
        self.day_4_infected_today = 0

        # 2. Agents decide location (parallel)
        max_workers = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(agent.decide_location) for agent in self.schedule]
            concurrent.futures.wait(futures)
        
        #Code for non_multi-processing.
        # for agent in self.schedule:
        #     agent.decide_location()

        # Rebuild the self.agents_outside list
        self.agents_outside = [a for a in self.schedule if a.location == "outside"]

        # 3. Decide interactions
        self.decide_agent_interactions()

        # 4. Interact => Infect (also track total contacts)
        np.random.shuffle(self.schedule)
        for agent in self.schedule:
            # Tally how many interactions occur
            self.total_contacts_today += len(agent.agent_interaction)
            # Infect
            agent.interact()
            # agent.interact() resets agent.agent_interaction afterwards

        for agent in self.schedule:
            update_day(agent)  # This increments agent.day_infected, sets 'To_Be_Infected' to 'Infected', etc.

        # Recompute how many are infected after the day ends
        new_infected_count = compute_num_infected(self)
        self.currently_infected = new_infected_count

        # self.daily_new_cases is incremented inside agent.infect() (see agent code),
        # or you can do it here if you prefer. 
        # Just be sure you do it consistently.

        # 6. Count day-4 infected
        for agent in self.schedule:
            if agent.health_condition == "Infected" and agent.day_infected == 4:
                self.day_4_infected_today += 1

        # 7. This day is over, increment time_step
        self.time_step += 1
        self.yesterday_day_4_infected = self.day_4_infected_today

    def run_model(self, checkpoint_path, offset):
        """
        Execute the entire simulation. 
         - Collect a "Day 0" row for initial conditions
         - For each day: step() then collect
         - Possibly stop early if no infected remain
         - Save checkpoint each day
        """

        self.offset = offset
        end_program = 0
        start = time.time()

        # --- 0. Collect initial conditions for "Day 0" ---
        self.datacollector.collect(self)

        # --- 1. Main Loop ---
        for i in tqdm(range(self.offset, self.step_count)):

            # A) One day of simulation
            self.step()

            # B) Collect the "end of day" data
            self.datacollector.collect(self)

            # Print debug info
            print(f"End of Day {self.time_step}: daily_new_cases = {self.daily_new_cases}, infected = {self.currently_infected}, day4inf = {self.day_4_infected_today}")

            # C) Early stopping check
            if self.currently_infected == 0:
                end_program += 1
            if end_program == 2:
                # Save final checkpoint and break
                final_path = f"{checkpoint_path}/{self.name}-final_early.pkl"
                self.save_checkpoint(final_path)
                break

            # D) Save a checkpoint at the end of each day
            path = f"{checkpoint_path}/{self.name}-{i+1}.pkl"
            self.save_checkpoint(path)
            clear_cache()

        end = time.time()
        print(f"Time taken for {self.population} agents and {self.time_step} days: {end - start} seconds.")


    def save_checkpoint(self, file_path):
        """
        Save a pickle checkpoint of the current model state.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_checkpoint(file_path):
        """
        Load a model state from a pickle checkpoint.
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)
