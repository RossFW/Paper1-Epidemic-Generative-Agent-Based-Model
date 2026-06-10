from utils import get_completion_from_messages, probability_threshold
import numpy as np
import sys


class Agent:
    '''
    Define who an agent is:
    unique_id: assigns ID to agent
    name: name of the agent
    age: age of the agent
    traits: Big Five personality traits of the agent
    location: flag for staying at home or going outside
    health_condition: flag to say if Susceptible, To_Be_Infected, Infected, or Recovered
    day_infected: agent attribute to count the number of days agent has spent infected
    '''

    def __init__(self,model,unique_id, name, age, traits,location,health_condition, day_infected):
        
        #Reference to world class
        self.model = model

        #Agents unique_id is for data tracking (not used in our code, but for quality of life purposes)
        self.unique_id = unique_id
        
        #Persona
        self.name = name
        self.age = age
        self.location=location
        self.traits=traits

        #Health initialization of agent
        self.health_condition=health_condition
        self.day_infected=day_infected

        #Creating list for agent interactions
        self.agent_interaction=[]
        self.indiv_contact_rate = 0

        #Used to save individual attributes for individual-level data analysis
        self.mems = {"name":name,"age":age,"traits":traits}

    #########################################
    #             Health Feedback           #
    #########################################  
    def get_health_string(self):
        '''
        This function is to get the relevant health string for the agent. 
        There are 3 types of health feedback an agent has: it "feels normal", "has a cough", and "has a cough and a fever."
        On day 0 to 2, the agent feels normal. On day 3, the agent has a cough. On day 4 and 5, the agent has a fever and a cough. On day 6, the fever subsides and the agent has a cough.
        '''
        # List of health status messages for the agent
        Health_strings=[f"You feel normal.", # Normal health status
                        f"You have a slight cough.", # Mild symptom: cough
                        f"You have a cough and a fever." # Severe symptoms: cough and fever
                        ]

        # If the agent is susceptible, recovered, or within first 2 days of infection, they feel normal
        if self.health_condition=="Susceptible" or self.health_condition=="Recovered" or self.day_infected<=2:
            return Health_strings[0]
        
        # On the 3rd day of infection, agent has a slight cough
        if self.day_infected==3:
            return Health_strings[1]
        
        # On 4th and 5th days of infection, agent has both cough and fever
        if self.day_infected==4 or self.day_infected==5:
            return Health_strings[2]

        # On the 6th day of infection, the fever subsides and the agent is left with a slight cough
        if self.day_infected==6:
            return Health_strings[1]


    ########################################
    #      Decision-making functions       #
    ########################################

    def get_decision(self):
        """
        1) Prompt ChatGPT for reasoning & response
        2) Parse them, fallback to yes/no if needed
        3) Return final (reasoning, response)
        """

        #Prompt Asked to ChatGPT
        question_prompt = f"""
        You are {self.name}. You are {self.age} years old. You are a person who is {self.traits[0]}, {self.traits[1]}, {self.traits[2]}, {self.traits[3]}, and {self.traits[4]}.
        You live in the town of Dewberry Hollow. You have a job and go to the office for work everyday.
        {self.get_health_string()}
        You go to work to earn money to support yourself.
        You know about the Catasat virus spreading across the country. It is an airborne virus causing an infectious disease that spreads from human to human. The deadliness of the virus is unknown. 
        You check the newspaper and find that {(self.model.yesterday_day_4_infected*100)/self.model.population: .1f}% of Dewberry Hollow's population were diagnosed with new infections of the Catasat virus yesterday.
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
        # Initialize a list containing a dictionary with a system role and the question prompt content.
        print(question_prompt)
        messages = [{"role": "developer", "content": question_prompt}]
        try:
            output = get_completion_from_messages(messages, temperature=0)
        except Exception as e:
            print(f"Something unexpected happened: {e}")
            sys.exit(e)

        reasoning = ""
        response  = ""
        try:
            intermediate = output.split("Reasoning:", 1)[1]
            reasoning, response = intermediate.split("Response:")
            reasoning = reasoning.strip()
            # Take everything up to the first period (or the entire string if no period)
            response = response.strip().split(".", 1)[0]
        except:
            print("Reasoning or response were not parsed correctly.")
            reasoning = None
            response  = None

        # If response is None (failed parsing), or it's not 'yes'/'no', fallback to random:
        if not response:
            response = np.random.choice(["yes", "no"], replace=False)
            print(f"No valid response found. Defaulting to: {response}")
        else:
            response = response.lower()
            if response not in ["yes", "no"]:
                response = np.random.choice(["yes", "no"], replace=False)
                print(f"Response was unexpected. Defaulting to: {response}")

        
        # Debug printing
        print(f"\n{self.name}'s Reasoning: {reasoning}\n{self.name}'s response: {response}")

        del question_prompt
        return reasoning, response


    ########################################
    #      Decide Location functions       #
    ########################################

    def decide_location(self):
        """
        Agents decide whether they want to stay home or go outside.
        We set location accordingly, then store the final location
        (and reasoning) in mems.
        """
        reasoning, response = self.get_decision()

        # If agent wants to stay home
        if response == "yes":
            self.location = "home"
            if self in self.model.agents_outside:
                self.model.agents_outside.remove(self)
        else:
            self.location = "outside"
            if self not in self.model.agents_outside:
                self.model.agents_outside.append(self)

        # Store post-decision data in mems
        self.mems[self.model.time_step] = {
            "health condition": self.health_condition,
            "reasoning": reasoning,
            "response": response,
            "health string": self.get_health_string(),
            "location": self.location
        }
        del reasoning, response




    ################################################################################
    #                       Meet_interact_infect functions                         #
    ################################################################################ 
    def add_agent_interaction(self, agent):
        '''
        Called in self.model.decide_agent_interactions()
        '''
        if len(self.agent_interaction) >= self.indiv_contact_rate or len(agent.agent_interaction) >= agent.indiv_contact_rate:
            return

        self.agent_interaction.append(agent)
        agent.agent_interaction.append(self)


    ########################################
    #                 Interact             #
    ########################################

    def interact(self):
        '''
        Step 1. Run infection for each agent_interaction
        Step 2. Reset agent_interaction for next day
        Used in self.step()
        '''

        for agent in self.agent_interaction:
            self.infect(agent)
        #Reset Agent Interaction list
        self.agent_interaction=[]

    ########################################
    #               Infect                 #
    ########################################
    def infect(self, other):
        '''
        Step 1. See health status of both members.
        Step 2. If one is infected, roll for threshold

        Used in self.interact()
        '''
        
        #sets infection threshold
        infection_rate = self.model.infection_rate


        #if self is sick and other is not
        if self.health_condition=="Infected":

            #See if there is a chance they get infected
            if probability_threshold(infection_rate) and other.health_condition=="Susceptible":

                #Other is infected
                other.health_condition="To_Be_Infected"

        #if other is sick and self is not
        elif other.health_condition=="Infected":

            #See if there is a chance they get infected
            if probability_threshold(infection_rate) and self.health_condition == "Susceptible":

                #Self is infected
                self.health_condition="To_Be_Infected"
