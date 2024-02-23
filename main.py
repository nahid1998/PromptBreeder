import os
import pandas as pd
import random
import re

import LLMs
import Evolution
import Mutation

# parameters
llm_model = 'gpt-3.5'
dataset_name = 'gsm8k'
train_test_size = 10 # it was 100 in Prompt Breeder
population_size = 50 
problem_description = "Solve the math word problem, giving your answer as an arabic numeral."

# load DATASET
train_path = llm_model + "_on_" + dataset_name + "/train.csv"
if os.path.exists(train_path) :
    df_train = pd.read_csv(train_path)
    train_list = df_train.to_numpy()
    df_test = pd.read_csv(llm_model + "_on_" + dataset_name + "/test.csv")
else :
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, 'main')
    train_data = dataset['train']
    
    r = random.randrange(len(train_data) - (2 * train_test_size))
    train_list = []
    for i in range(r, (r + train_test_size)) :
        matches = re.findall(r'\b\d+\b', train_data['answer'][i])
        t_answer = int(matches[-1]) if matches else None
        train_list.append([train_data['question'][i], t_answer])
        df_train = pd.DataFrame(train_list)
        df_train.columns = ['question','answer']
        df_train.to_csv(train_path, index= False)
    test_list = []
    for i in range((r + train_test_size), (r + (2 * train_test_size))) :
        matches = re.findall(r'\b\d+\b', train_data['answer'][i])
        t_answer = int(matches[-1]) if matches else None
        test_list.append([train_data['question'][i], t_answer])
        df_test = pd.DataFrame(test_list)
        df_test.columns = ['question','answer']
        df_test.to_csv(llm_model + "_on_" + dataset_name + "/test.csv", index= False)

# initialize population
file_path = llm_model + "_on_" + dataset_name + "/population_initialize.csv"
if os.path.exists(file_path) :
    df_population = pd.read_csv(file_path)
    population = df_population.to_numpy()
else :
    population = Evolution.Evolution.Initialization(problem_description, population_size, llm_model,dataset_name)

print(len(population))

# Evaluate fitness
population_fitness = Evolution.Evolution.Evaluate_fitness(population[:20], train_list, llm_model,dataset_name, 0)


