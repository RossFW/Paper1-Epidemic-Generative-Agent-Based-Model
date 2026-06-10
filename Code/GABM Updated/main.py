import os
# OPENAI_API_KEY is read from the environment (set it in your shell or a .env file).
from world import World
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import sys
import evaluation



if __name__ == "__main__":
    #Arguements for our code.
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GABM", help = "Name of the run to save outputs.")
    parser.add_argument("--contact_rate", default=5, type=int, help="Contact Rate")
    parser.add_argument("--infection_rate", default=0.1, type=float, 
                        help="Infection Rate")
    parser.add_argument("--no_init_healthy", default=18, type=int, 
                        help="Number of initial healthy people in the world.")
    parser.add_argument("--no_init_infect", default= 2, type=int,
                        help="Number of initial infected people in the world.")
    parser.add_argument("--no_days", default=999, type=int,
                        help="Total maximum number of days the world would run.")
    parser.add_argument("--time_to_heal", default=6,type=int, help="Time taken to heal from infection.")
    parser.add_argument("--no_of_runs", default = 1, type = int, help = "Total number of times you want to run this code.")
    parser.add_argument("--offset", default=0,type=int, help="offset is equal to number of days if you need to load a checkpoint")
    parser.add_argument("--load_from_run", default=0,type=int, help="equal to (run # - 1) if you need to load a checkpoint (e.g. if you want to load run 2 checkpoint 8, then offset = 8, load_from_run = 1)")

    args = parser.parse_args()
    print(f"Parameters: {args}")

    #Creating output and checkpoint folders as needed
    if os.path.exists("output") is not True:
        os.mkdir("output")
    if os.path.exists("checkpoint") is not True:
        os.mkdir("checkpoint")


    for i in range(args.load_from_run, args.no_of_runs):
        print(f"--------Run - {i+1}---------")
        #creates more folders for organization purposes
        checkpoint_path = f"checkpoint/run-{i+1}"
        output_path = f"output/run-{i+1}"
        if os.path.exists(checkpoint_path) is not True:
            os.mkdir(checkpoint_path)
        if os.path.exists(output_path) is not True:
            os.mkdir(output_path)

        if args.load_from_run != 0:  # Load specific checkpoint only from the specified run
            checkpoint_file = f"checkpoint/run-{args.load_from_run+1}/{args.name}-{args.offset}.pkl"
            if os.path.exists(checkpoint_file):
                model = World.load_checkpoint(checkpoint_file)
            else:
                #Try again if issue prevailed in args
                print(f"Warning! Checkpoint not found. Initializing new world for run {args.load_from_checkpoint+1}. This is normal if you want to continue from run {args.load_from_checkpoint+1} from scratch")
                model = World(args)
        
        else:
            if args.offset !=0:
                try:
                    model = World.load_checkpoint(f"checkpoint/run-1/{args.name}-{args.offset}.pkl")
                except Exception as e:
                    sys.exit(e)
            else:
                model = World(args)

        #Run model
        model.run_model(checkpoint_path, args.offset)
        evaluation.evaluate_simulation(model, args, run_number=i+1, output_path=output_path)
        model.save_checkpoint(file_path = checkpoint_path + f"/{args.name}-completed.pkl")
