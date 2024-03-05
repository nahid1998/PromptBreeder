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
problem_description = "Solve the math word problem, giving your answer as an arabic numeral."
train_test_size = 10 # it was 100 in Prompt Breeder
population_size = 50 
epochs = 0


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

# DO Mutation
for e in range(epochs) :
    mutated_population = []
    switch = random.randint(0,9)
    if switch == 0 : # ZERO ORDER PROMPT GENERATION
        Mutation.Direct_mutation.Zero_order_Prompt_Generation(population,llm_model,problem_description) #AMBIGUOUS

    if switch == 1 : # FIRST ORDER PROMPT GENERATION 
        for i in range(population_size/2):
            if population_fitness[i][4] > population_fitness[i + (population_size/2)][4] : # compare fitness and choose the one with higher fitness
                p = population_fitness[i]
            else :
                p = population_fitness[i + (population_size/2)]
            m_1 = Mutation.Direct_mutation.First_order_Prompt_Generation(llm_model, p[1], p[0])
            m_2 = Mutation.Direct_mutation.First_order_Prompt_Generation(llm_model, p[3], p[2])
            # Append the choosen task-prompt and its mutated-task-prompt 
            mutated_population.append([p[0],p[1],p[2],p[3]], [m_1,p[1],m_2,p[3]])  
        
    if switch == 2 : # ESTIMATION OF DISTRIBUTION (EDA) MUTATION
        filtered_list, len_f_p = Mutation.Estimation_of_Distribution_mutation.Filtered_List(population)
        for i in range(population_size/2): 
            # compare fitness and choose the one with higher fitness
            if population_fitness[i][4] > population_fitness[i + (population_size/2)][4] :
                p = population_fitness[i]
            else :
                p = population_fitness[i + (population_size/2)]
            m_1 = Mutation.Estimation_of_Distribution_mutation.EDA_Mutation(llm_model, filtered_list, p[1], e)
            m_2 = Mutation.Estimation_of_Distribution_mutation.EDA_Mutation(llm_model, filtered_list, p[3], e)
            # Append the choosen task-prompt and its mutated-task-prompt 
            mutated_population.append([p[0],p[1],p[2],p[3]], [m_1,p[1],m_2,p[3]])  

    if switch == 3 : # EDA RANK AND INDEX MUTATION
        filtered_list, len_f_p = Mutation.Estimation_of_Distribution_mutation.Filtered_List(population, True)
        for i in range(population_size/2): 
            # compare fitness and choose the one with higher fitness
            if population_fitness[i][4] > population_fitness[i + (population_size/2)][4] :
                p = population_fitness[i]
            else :
                p = population_fitness[i + (population_size/2)]
            m_1 = Mutation.Estimation_of_Distribution_mutation.EDA_Rank_Index_Mutation(llm_model, filtered_list, len_f_p, p[1], e)
            m_2 = Mutation.Estimation_of_Distribution_mutation.EDA_Rank_Index_Mutation(llm_model, filtered_list, len_f_p, p[3], e)
            # Append the choosen task-prompt and its mutated-task-prompt 
            mutated_population.append([p[0],p[1],p[2],p[3]], [m_1,p[1],m_2,p[3]])  
        
    if switch == 4 : # LINEAGE BASED MUTATION
        list_best_so_far = Mutation.Estimation_of_Distribution_mutation.List_Best_So_Far(best_so_fars)
        for i in range(population_size/2): 
            # compare fitness and choose the one with higher fitness
            if population_fitness[i][4] > population_fitness[i + (population_size/2)][4] :
                p = population_fitness[i]
            else :
                p = population_fitness[i + (population_size/2)]
            m_1 = Mutation.Estimation_of_Distribution_mutation.Lineage_Based_Mutation(llm_model, list_best_so_far, p[1], e)
            m_2 = Mutation.Estimation_of_Distribution_mutation.Lineage_Based_Mutation(llm_model, list_best_so_far, p[3], e)
            # Append the choosen task-prompt and its mutated-task-prompt 
            mutated_population.append([p[0],p[1],p[2],p[3]], [m_1,p[1],m_2,p[3]])  

    if switch == 5 : # ZERO-ORDER HYPER-MUTATION
        df_t = pd.read_csv('thinking_styles.csv')
        thinking_styles = df_t.to_numpy().flatten()
        for i in range(population_size/2): 
            # compare fitness and choose the one with higher fitness
            if population_fitness[i][4] > population_fitness[i + (population_size/2)][4] :
                p = population_fitness[i]
            else :
                p = population_fitness[i + (population_size/2)]
            t_1, m_1= Mutation.HyperMutation.Zero_Order_Hyper_Mutation(
                llm_model=llm_model, 
                problem_description=problem_description,
                thinking_style=thinking_styles[random.randint(len(thinking_styles))],
                task_prompt=p[0],
                epoch=e)
            t_2, m_2 = Mutation.HyperMutation.Zero_Order_Hyper_Mutation(
                llm_model=llm_model, 
                problem_description=problem_description,
                thinking_style=thinking_styles[random.randint(len(thinking_styles))],
                task_prompt=p[2],
                epoch=e)
            # Append the choosen task-prompt and its mutated-task-prompt 
            mutated_population.append([p[0],p[1],p[2],p[3]], [t_1,m_1,t_2,m_2])  

    if switch == 6 : # FIRST-ORDER HYPER-MUTATION
        for i in range(population_size/2): 
            # compare fitness and choose the one with higher fitness
            if population_fitness[i][4] > population_fitness[i + (population_size/2)][4] :
                p = population_fitness[i]
            else :
                p = population_fitness[i + (population_size/2)]
            t_1, m_1 = Mutation.HyperMutation.First_Order_Hyper_Mutation(llm_model, p[0], p[1], e)
            t_2, m_2 = Mutation.HyperMutation.First_Order_Hyper_Mutation(llm_model, p[2], p[3], e)
            # Append the choosen task-prompt and its mutated-task-prompt 
            mutated_population.append([p[0],p[1],p[2],p[3]], [t_1,m_1,t_2,m_2])  

    if switch == 7 : # LAMARCKIAN MUTATION
        for i in range(population_size/2): 
            # compare fitness and choose the one with higher fitness
            if population_fitness[i][4] > population_fitness[i + (population_size/2)][4] :
                p = population_fitness[i]
            else :
                p = population_fitness[i + (population_size/2)]
            r_workingout = random.randint(0,(len(train_list)-1))
            t_1 = Mutation.Lamarckian_mutation.Working_out_to_task_prompt(llm_model, train_list[r_workingout], train_list[r_workingout + 1], e)
            r_workingout = random.randint(0,(len(train_list)-1))
            t_2 = Mutation.Lamarckian_mutation.Working_out_to_task_prompt(llm_model, train_list[r_workingout], train_list[r_workingout + 1], e)
            # Append the choosen task-prompt and its mutated-task-prompt 
            mutated_population.append([p[0],p[1],p[2],p[3]], [t_1,p[1],t_2,p[3]]) 

        # if switch == 8 :        
