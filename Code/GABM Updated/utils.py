from dotenv import load_dotenv
load_dotenv() 

from names_dataset import NameDataset
import numpy as np
import openai
import os
import shutil
import random
import logging
import backoff

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

backoff_logger = logging.getLogger('backoff')
backoff_logger.setLevel(logging.INFO)
file_handler_backoff = logging.FileHandler('backoff_log.txt')
file_handler_backoff.setFormatter(formatter)
backoff_logger.addHandler(file_handler_backoff)

error_logger = logging.getLogger('errors')
error_logger.setLevel(logging.ERROR)  
file_handler_errors = logging.FileHandler('errors.txt')
file_handler_errors.setFormatter(formatter)
error_logger.addHandler(file_handler_errors)

def probability_threshold(threshold):
    '''
    Generates random number from 0 to 1
    '''
    
    return (np.random.rand()<threshold)

def generate_names(n: int, s: int, country_alpha2='US'):
    '''
    Returns random names as names for agents from top names in the USA
    Used in World.init to initialize agents
    '''

    # This function will randomly selct n names (n/2 male and n/2 female) without
    # replacement from the s most popular names in the country defined by country_alpha2
    if n % 2 == 1:
        n += 1
    if s % 2 == 1:
        s += 1

    nd = NameDataset()
    male_names = nd.get_top_names(s//2, 'Male', country_alpha2)[country_alpha2]['M']
    female_names = nd.get_top_names(s//2, 'Female', country_alpha2)[country_alpha2]['F']
    if s < n:
        raise ValueError(f"Cannot generate {n} unique names from a list of {s} names.")
    # generate names without repetition
    names = random.sample(male_names, k=n//2) + random.sample(female_names, k=n//2)
    del male_names
    del female_names
    np.random.shuffle(names)
    return names


def generate_big5_traits():
    '''
    Return big 5 traits for each agent
    Used in World.init to initialize agents
    '''

    trait_list = [["extroverted", "introverted"],["agreeable","antagonistic"],["conscientious","unconscientious"],["neurotic","emotionally stable"],["open to experience","closed to experience"]]
    traits = [np.random.choice(a, replace= False) for a in trait_list]
    return traits

def generate_age():

    #list of percentage of population by age (18-65) from 2023
    likelihoods = [
    2.0752895752895800,
    2.0752895752895800,
    2.1396396396396400,
    2.2844272844272800,
    2.2522522522522500,
    2.1718146718146700,
    2.1235521235521200,
    2.1074646074646100,
    2.1074646074646100,
    2.1396396396396400,
    2.1718146718146700,
    2.2039897039897000,
    2.2522522522522500,
    2.2844272844272800,
    2.2844272844272800,
    2.2200772200772200,
    2.1879021879021900,
    2.1557271557271600,
    2.1557271557271600,
    2.1557271557271600,
    2.1235521235521200,
    2.1557271557271600,
    2.1396396396396400,
    2.1074646074646100,
    2.1074646074646100,
    2.0109395109395100,
    1.9787644787644800,
    1.9465894465894500,
    1.8822393822393800,
    1.8983268983269000,
    1.8661518661518700,
    1.8983268983269000,
    1.9787644787644800,
    2.0913770913770900,
    2.0592020592020600,
    1.9787644787644800,
    1.9305019305019300,
    1.9465894465894500,
    1.9787644787644800,
    2.0431145431145400,
    2.0913770913770900,
    2.0913770913770900,
    2.0913770913770900,
    2.0752895752895800,
    2.0431145431145400,
    1.9948519948519900,
    1.9948519948519900,
    1.9465894465894500
    ]
    likelihoods = [l/100 for l in likelihoods] #ensure that percentages are now probabilities
    assert int(sum(likelihoods)) == 1, f"Sum of likelihoods is not 1! Sum is: {sum(likelihoods)}"
    age_range = np.arange(18,66) #list of integers from 18-65
    return int(np.random.choice(age_range,size=1, p = likelihoods)) #specifying probability distribution for choosing age

def update_day(agent):
    """
    Update the agent's day_infected counter.
    Increment counters for new infections and update the world's currently_infected count.
    """
    agent.indiv_contact_rate = 0
    Time_to_heal = 6

    if agent.health_condition in ["Susceptible", "Recovered"]:
        return

    if agent.health_condition == "To_Be_Infected":
        agent.health_condition = "Infected"
        agent.day_infected = 0
        agent.model.daily_new_cases += 1
        agent.model.currently_infected += 1

    agent.day_infected += 1

    if agent.day_infected > Time_to_heal:
        agent.day_infected = None
        agent.health_condition = "Recovered"
        agent.model.currently_infected -= 1



api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)
@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError, openai.Timeout), max_tries=20, logger=backoff_logger)


def get_completion_from_messages(messages, model="gpt-4o-mini", temperature=0):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content
    except Exception as e:
        error_logger.error(f"Something unexpected happened YEET. Error: {e}")
        raise
        

def clear_cache(): #clear cache for memory efficiency
    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
