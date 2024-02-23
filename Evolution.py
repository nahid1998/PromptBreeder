import random
from LLMs import LLMs
import pandas as pd

class Evolution :

   def Initialization(problem_description, no_population, llm_model, dataset_name):
      df_t = pd.read_csv('thinking_styles.csv')
      thinking_styles = df_t.to_numpy().flatten()
      df_m = pd.read_csv('mutator_prompts.csv')
      mutation_prompts = df_m.to_numpy().flatten()
      population = [[0 for _ in range(4)] for _ in range(no_population)] # 4 x n

      for i in range(no_population) :
         # Both the mutation-prompt and the thinking-style are randomly sampled from an initial set of mutation-prompts and a set of thinkingstyles
         t1 = thinking_styles[random.randrange(len(thinking_styles))]
         t2 = thinking_styles[random.randrange(len(thinking_styles))]
         m1 = mutation_prompts[random.randrange(len(mutation_prompts))]
         m2 = mutation_prompts[random.randrange(len(mutation_prompts))]

         # generate the initial task-prompts by concatenating a m and a t  to the problem description, and provide that to the LLM to produce a continuation, resulting in an initial task-prompt.
         prompt1 = m1 + " " + t1 + " " + "INSTRUCTION: \n" + problem_description + " \nINSTRUCTION MUTANT = "
         task_prompt1 = LLMs.Response(llm_model, prompt1, temperature= 1, max_token= 50)
         prompt2 = m2 + " " + t2 + " " + "INSTRUCTION: \n" + problem_description + " \nINSTRUCTION MUTANT = "
         task_prompt2 = LLMs.Response(llm_model, prompt2, temperature=1, max_token= 50)

         # append to population
         population[i][0] = task_prompt1
         population[i][1] = m1
         population[i][2] = task_prompt2
         population[i][3] = m2
      
      df_population = pd.DataFrame(population)
      df_population.columns = ['task_prompt1', 'mutation_prompt1', 'task_prompt2', 'mutation_prompt2']
      file_path = llm_model + "_on_" + dataset_name + "/population_initialize.csv"
      df_population.to_csv(file_path,index=False)
      return population
    
   def Evaluate_fitness(population, qa_dataset, llm_model, dataset_name, epoch) :
      population_fitness = []
      for index, p in enumerate(population) :
         print("population : " + str(index))
         p_fit = 0
         for qa in qa_dataset :
            question = qa[0]
            answer = qa[1]
            
            continuation = LLMs.Response(
               llm_model = llm_model,
               prompt = p[0] + question,
               temperature = 0,
               max_token = 30 )
            # print(continuation)
            
            llm_answer = LLMs.Response(
               llm_model = llm_model,
               prompt = continuation + p[2],
               temperature = 0,
               max_token = 5 )
            # print(llm_answer)
            
            evaluate_prompt = "correct answer = " + str(answer) + " \nalgorithm answer = " + str(llm_answer) +" \nIs the algorithm answer correct?(Yes or No)"
            evaluate = LLMs.Response(llm_model = llm_model, prompt = evaluate_prompt, temperature = 0, max_token = 1 )
            print(evaluate_prompt)
            print(evaluate)
            
            if evaluate == "yes" or evaluate == "Yes" :
               p_fit += 1
         #population_fitness.append([p[0], p[1], p[2], p[3], (p_fit / len(qa_dataset))])
         population[index].append((p_fit / len(qa_dataset))) 
         df_population = pd.DataFrame(population)
         df_population.columns = ['task_prompt1', 'mutation_prompt1', 'task_prompt2', 'mutation_prompt2', 'fitness']
         file_path = llm_model + "_on_" + dataset_name + "/population_fitness_" + str(epoch) + ".csv"
         df_population.to_csv(file_path,index=False)
      return population
