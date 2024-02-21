class Evolotion :
    def Initialization(problem_description, no_population, llm_model):
       import random
       from LLMs import LLMs
       import pandas as pd
       df_t = pd.read_csv('thinking_styles.csv')
       thinking_styles = df_t.to_numpy().flatten()
       df_m = pd.read_csv('mutator_prompts.csv')
       mutation_prompts = df_m.to_numpy().flatten()
       population = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(no_population)] # 2 x 2 x n

       for i in range(no_population) :
        # Both the mutation-prompt and the thinking-style are randomly sampled from an initial set of mutation-prompts and a set of thinkingstyles
        t1 = thinking_styles[random.randrange(len(thinking_styles))]
        t2 = thinking_styles[random.randrange(len(thinking_styles))]
        m1 = mutation_prompts[random.randrange(len(mutation_prompts))]
        m2 = mutation_prompts[random.randrange(len(mutation_prompts))]

        # generate the initial task-prompts by concatenating a m and a t  to the problem description, and provide that to the LLM to produce a continuation, resulting in an initial task-prompt.
        prompt1 = m1 + " " + t1 + " " + "INSTRUCTION: \n" + problem_description + " \nINSTRUCTION MUTANT = "
        task_prompt1 = LLMs.response(llm_model, prompt1)
        prompt2 = m2 + " " + t2 + " " + "INSTRUCTION: \n" + problem_description + " \nINSTRUCTION MUTANT = "
        task_prompt2 = LLMs.response(llm_model, prompt2)

        # append to population
        population[i][0][0] = task_prompt1
        population[i][0][1] = m1
        population[i][1][0] = task_prompt2
        population[i][1][1] = m2

       return population
    
    def Evaluate_fitness(population) :
       return
