import time
import openai
import numpy as np
import pandas as pd
import concurrent.futures
import random
from names_dataset import NameDataset
import datetime
import os


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
    random.shuffle(names)
    return names

# def generate_big5_traits():
#     '''
#     Return big 5 traits for each agent
#     Used in World.init to initialize agents
#     '''

#     #Trait generation
#     agreeableness_pos=['Cooperation','Amiability','Empathy','Leniency','Courtesy','Generosity','Flexibility',
#                         'Modesty','Morality','Warmth','Earthiness','Naturalness']
#     agreeableness_neg=['Belligerence','Overcriticalness','Bossiness','Rudeness','Cruelty','Pomposity','Irritability',
#                         'Conceit','Stubbornness','Distrust','Selfishness','Callousness']
#     #Did not use Surliness, Cunning, Predjudice,Unfriendliness,Volatility, Stinginess

#     conscientiousness_pos=['Organization','Efficiency','Dependability','Precision','Persistence','Caution','Punctuality',
#                             'Punctuality','Decisiveness','Dignity']
#     #Did not use Predictability, Thrift, Conventionality, Logic
#     conscientiousness_neg=['Disorganization','Negligence','Inconsistency','Forgetfulness','Recklessness','Aimlessness',
#                             'Sloth','Indecisiveness','Frivolity','Nonconformity']

#     surgency_pos=['Spirit','Gregariousness','Playfulness','Expressiveness','Spontaneity','Optimism','Candor'] 
#     #Did not use Humor, Self-esteem, Courage, Animation, Assertion, Talkativeness, Energy level, Unrestraint
#     surgency_neg=['Pessimism','Lethargy','Passivity','Unaggressiveness','Inhibition','Reserve','Aloofness'] 
#     #Did not use Shyness, Silenece

#     emotional_stability_pos=['Placidity','Independence']
#     emotional_stability_neg=['Insecurity','Emotionality'] 
#     #Did not use Fear, Instability, Envy, Gullibility, Intrusiveness
    
#     intellect_pos=['Intellectuality','Depth','Insight','Intelligence'] 
#     #Did not use Creativity, Curiousity, Sophistication
#     intellect_neg=['Shallowness','Unimaginativeness','Imperceptiveness','Stupidity']


#     #Combine each trait
#     agreeableness_tot = agreeableness_pos + agreeableness_neg
#     conscientiousness_tot = conscientiousness_pos + conscientiousness_neg
#     surgency_tot = surgency_pos + surgency_neg
#     emotional_stability_tot = emotional_stability_pos + emotional_stability_neg
#     intellect_tot = intellect_pos + intellect_neg

#     #create traits list to be returned
#     traits_list = []

#     # for _ in range(n):
#     agreeableness_rand = random.choice(agreeableness_tot)
#     conscientiousness_rand = random.choice(conscientiousness_tot)
#     surgency_rand = random.choice(surgency_tot)
#     emotional_stability_rand = random.choice(emotional_stability_tot)
#     intellect_rand = random.choice(intellect_tot)

#     selected_traits=[agreeableness_rand,conscientiousness_rand,surgency_rand,
#                             emotional_stability_rand,intellect_rand]

#     traits_chosen = (', '.join(selected_traits))
#         #traits_list.append(selected_traits)
#         # traits_list.append(traits_chosen)
#     # del agreeableness_rand
#     # del conscientiousness_rand
#     # del surgency_rand
#     # del emotional_stability_rand
#     # del intellect_rand
#     # del selected_traits
#     # del traits_chosen
#     return selected_traits

def generate_big5_traits():
    '''
    Return big 5 traits for each agent
    Used in World.init to initialize agents
    '''

    trait_list = [["extroverted", "introverted"],["agreeable","antagonistic"],["conscientious","unconscientious"],["neurotic","emotionally stable"],["open to experience","closed to experience"]]
    traits = [np.random.choice(a) for a in trait_list]
    # traits_chosen = (', '.join(traits))
    return traits

def generate_age(): 

    #list of percentage of population by age (18-65) sourced from https://www.census.gov/popclock/
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
    age_range = np.arange(18,66) #list of integers from 18-65

    return int(np.random.choice(age_range,size=1,p=likelihoods)[0]) #specifying probability distribution for choosing age


def get_completion_from_messages(messages, model="gpt-3.5-turbo-0301", temperature=0):
    success = False
    retry = 0
    max_retries = 3
    while retry< max_retries and not success:
      try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, # this is the degree of randomness of the model's output
            )
        success = True
      except Exception as e:
        print(f"Error: {e}\nRetrying...")
        retry+=1
        
        time.sleep(0.5)

    return response.choices[0].message["content"]


class Agent:
    def __init__(self, name, age, traits, health):
        self.name = name
        self.age = age
        self.traits = traits
        self.health = health
        self.error = 0

    def decide_location(self, infection_perc):
        openai.api_key = os.getenv("OPENAI_API_KEY")  # set via environment / .env
        
        # aritras_prompt = f"""
        # You are {self.name}. You are {self.age} years old. You are a person who is {self.traits[0]}, {self.traits[1]}, {self.traits[2]}, {self.traits[3]}, and {self.traits[4]}.
        # You live in the town of Dewberry Hollow. You have a job and go to the office for work everyday.
        # {self.health}
        # You go to work to earn money to support yourself.
        # You know about the Catasat virus spreading across the country. It is an airborne virus causing an infectious disease that spreads from human to human. The deadliness of the virus is unknown. 
        # You check the newspaper and find that {infection_perc: .2f}% out of Dewberry Hollow's population was diagnosed with new infections of the Catasat virus yesterday.
        # Based on the information above, should you stay at home for the entire day? Please provide your reasoning.

        # If the answer is "Yes", please state your reasoning as "Reasoning: [explanation]." 
        # If the answer is "No", please state your reasoning as "Reasoning: [explanation]."

        # The format should be as follow:
        # Reasoning:
        # Response:

        # Example of response format:
        # Reasoning: You are tired.
        # Response: Yes

        # You have to provide Response in a single word; it must be either "Yes" or "No".
        # """

        new_question_prompt =f"""
        You are {self.name}. You are {self.age} years old. You are a person who is {self.traits[0]}, {self.traits[1]}, {self.traits[2]}, {self.traits[3]}, and {self.traits[4]}.
        You live in the town of Dewberry Hollow. You have a job and go to the office for work everyday.
        {self.health}
        You go to work to earn money to support yourself.
        You know about the Catasat virus spreading across the country. It is an airborne virus causing an infectious disease that spreads from human to human. The deadliness of the virus is unknown. 
        You check the newspaper and find that {infection_perc: .2f}% of Dewberry Hollow's population were diagnosed with new infections of the Catasat virus yesterday.
        Should you stay at home for the entire day? Please provide your reasoning.

        If the answer is "Yes", please state your reasoning as "Reasoning: [explanation]." 
        If the answer is "No", please state your reasoning as "Reasoning: [explanation]."

        The format should be as follow:
        Reasoning:
        Response:

        Example of response format:
        Reasoning: You are tired.
        Response: Yes

        You have to provide Response in a single word; it must be either "Yes" or "No".
        """
        # old_question_prompt = f"""
        # You are {self.name}. You are {self.age} years old.

        # Your traits are given below:
        # {self.traits[0]}, {self.traits[1]}, {self.traits[2]}, {self.traits[3]}, and {self.traits[4]}

        # Your basic bio is below:
        # {self.name} lives in the town of Dewberry Hollow. {self.name} likes the town and has friends who also live there. {self.name} has a job and goes to the office for work everyday.

        # I will provide {self.name}'s relevant memories here:
        # {self.health}
        # {self.name} knows about the Catasat virus spreading across the country. It is an infectious disease that spreads from human to human contact via an airborne virus.
        # The deadliness of the virus is unknown.
        # {self.name} checks the newspaper and finds that {infection_perc: .2f}% of Dewberry Hollow's population caught new infections of the Catasat virus yesterday.
        # Based on the provided context, would {self.name} stay at home for the entire day? Please provide your reasoning.

        # If the answer is "Yes," please state your reasoning as "Reasoning: [explanation]."
        # If the answer is "No," please state your reasoning as "Reasoning: [explanation]."

        # The format should be as follow:
        # Reasoning:
        # Response:

        # Example response format:

        # Reasoning: {self.name} is tired.
        # Response: Yes

        # It is important to provide Response in a single word.
        # """
        # actual_old_prompt = f"""
        # You are {self.name}. You are {self.age} years old. 
       
        # Your traits are given below:
        # {self.traits[0]}, {self.traits[1]}, {self.traits[2]}, {self.traits[3]}, {self.traits[4]}
        
        # Your basic bio is below:
        # {self.name} lives in the town of Dewberry Hollow. {self.name} likes the town and has friends who also live there. {self.name} has a job and goes to the office for work everyday.
        
        # I will provide {self.name}'s relevant memories here:
        # {self.health}
        # {self.name} knows about the Catasat virus spreading across the country. It is an infectious disease that spreads from human to human contact via an airborne virus. The deadliness of the virus is unknown. Scientists are warning about a potential epidemic.
        # {self.name} checks the newspaper and finds that {infection_perc: .2f}% of Dewberry Hollow's population caught new infections of the Catasat virus yesterday.
        # {self.name} goes to work to earn money to support {self.name}'s self.
       
        # Based on the provided memories, should {self.name} stay at home for the entire day? Please provide your reasoning.

        # If the answer is "Yes," please state your reasoning as "Reasoning: [explanation]." 
        # If the answer is "No," please state your reasoning as "Reasoning: [explanation]."
        
        # The format should be as follow:
        # Reasoning:
        # Response:

        # Example response format:

        # Reasoning: {self.name} is tired.
        # Response: Yes

        # It is important to provide Response in a single word.
        # """
        # print(question_prompt)

        messages =  [{'role':'system', 'content': new_question_prompt}]
        try:
            output = get_completion_from_messages(messages, temperature=0)
        except Exception as e:
            print(f"{e}\nProgram paused. Retrying after 10s...")
            time.sleep(10)
            output = get_completion_from_messages(messages, temperature=0)
        reasoning = ""
        response = ""
        try:
            intermediate  = output.split("Reasoning:",1)[1]
            reasoning, response = intermediate.split("Response:")
            response = response.strip().split(".",1)[0]
            reasoning = reasoning.strip()
        except:
            print("Reasoning or response were not parsed correctly.")
            response = "No"
            reasoning = None
        # self.traits = " ,".join(self.traits)
        if response.lower() in ["yes","no"]:
            self.error = 0
        else:
            self.error = 1
            response = np.random.choice(["yes","no"])

        return self.name, self.age, ", ".join(self.traits), self.health, response.lower(), reasoning, infection_perc, self.error

if __name__ == "__main__":

    np.random.seed(42)
    no_of_agents = 30
    print("Generating list of names...")
    name_list = generate_names(n=30,s=200)
    # trait_list = [["extroverted", "introverted"],["agreeable","antagonistic"],["conscientious","unconscientious"],["neurotic","emotionally stable"],["open to experience","closed to experience"]]
    # Health_strings=["{self.name} feels normal.", "You have a slight cough.", "You have a cough and a fever."]
    infection_perc = np.arange(0,3.25,0.25) #np.append(np.arange(0,2,0.1) ,np.arange(2,5.5,0.5))
    # infection_perc = [int(10000000*i/100) for i in infection_perc]
    name_data = np.array([],dtype="S")
    response_data = np.array([],dtype="S")
    reasoning_data = np.array([],dtype="S")
    age_data = np.array([],dtype = "int")
    traits_data = np.array([],dtype="S")
    health_data = np.array([], dtype = "S")
    infect_rate_data = np.array([],dtype= "float")
    error_handling = np.array([], dtype= "int")
    #trait_list = generate_big5_traits(no_of_agents)
    agent_list = []
    for i in range(no_of_agents):
        name = np.random.choice(name_list, replace= False)
        age = generate_age()
        traits = generate_big5_traits()
        # health = np.random.choice(Health_strings)
        health = "You feel normal."
        # health = f"{name} feels normal."
        agent_list.append(Agent(name, age, traits, health))
    
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%d-%m-%Y-%H-%M-%S")
    filedir =  f"./{formatted_date_time}-30-agents"
    os.mkdir(filedir)
    start = time.time()
    for inf in infection_perc:
        print(f"Running infection percentage {inf: .2f}...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(agent.decide_location, inf) for agent in agent_list]
            completed_futures, remaining_futures = concurrent.futures.wait(futures)
            for future in concurrent.futures.as_completed(completed_futures):
                name, age, traits, health, response, reasoning, infect_rate, error = future.result()
                name_data = np.append(name_data, name)
                age_data = np.append(age_data, age)
                traits_data = np.append(traits_data, traits)
                health_data = np.append(health_data, health)
                response_data = np.append(response_data, response)
                reasoning_data = np.append(reasoning_data, reasoning)
                infect_rate_data = np.append(infect_rate_data, infect_rate)
                error_handling = np.append(error_handling, error)
            executor.shutdown()

    df = pd.DataFrame({
        "Names": name_data,
        "Age": age_data,
        "Traits": traits_data,
        "Health": health_data,
        "Response": response_data,
        "Reasoning": reasoning_data,
        "Error Handling": error_handling,
        "Infection Rate": infect_rate_data
    })
    df.to_csv(f"{filedir}/exp_30_agents.csv", index=False)

    end = time.time()
    print("Time taken: ", end-start)
    df['Response Boolean'] = df['Response'].apply(lambda x: True if x == 'no' else False)

    # Group by 'infect_rate' and sum up the 'response' values
    result = df.groupby('Infection Rate')['Response Boolean'].sum()
    result/=no_of_agents
    
    result.to_csv(f"{filedir}/mobility_30_agents.csv")
    # df2 = pd.read_csv("./group_final_prompt_v1.csv")

