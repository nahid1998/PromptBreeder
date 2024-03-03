from LLMs import LLMs

from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np


class Direct_mutation :
    
    def Zero_order_Prompt_Generation(population, llm_model, problem_description) :
        prompt = problem_description + " \nA list of 100 hints : "
        response = LLMs.Response(llm_model, prompt, temperature=1, max_token=50)
        return response

    def First_order_Prompt_Generation(llm_model, mutation_prompt, task_prompt) :
        print("in mutation")
        prompt = mutation_prompt + "   INSTRUCTION: " + task_prompt + "   INSTRUCTION MUTANT: "
        response = LLMs.Response(llm_model, prompt, temperature=0, max_token=50)
        return response
    

class Estimation_of_Distribution_mutation :

    def _filter_population(population):
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        filtered_population = [population[0]]
        for p in population[1:]:
            _flag = True
            for f in filtered_population:
                inputs1 = tokenizer(p[0], return_tensors='pt', padding=True, truncation=True)
                inputs2 = tokenizer(f[0], return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    outputs1 = model(**inputs1)
                    embeddings1 = outputs1.last_hidden_state.mean(dim=1)  
                    outputs2 = model(**inputs2)
                    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
                # Calculate cosine similarity
                similarity = F.cosine_similarity(embeddings1, embeddings2).item()
                if similarity >= 0.95 :
                    _flag = False
                    break
            if _flag :
                filtered_population.append(p)
        return filtered_population
    

    def Filtered_List(population, is_sorted = False) :
        print("population : ")
        print(len(population))
        filtered_population = Estimation_of_Distribution_mutation._filter_population(population)
        print("filtered_population :")
        print(len(filtered_population))

        if is_sorted :
            filtered_population = sorted(filtered_population, key=lambda x: x[4])
        else :
            np.random.shuffle(filtered_population)

        list = ""
        index = 1
        for p in filtered_population :
            p1 = "PROMPT : " + p[0]
            p2 = "PROMPT : " + p[2]
            list += str(index) + " .\t" + p1 + " \n"
            index += 1
            list += str(index) + " .\t" + p2 + " \n"
            index += 1
        return list , len(filtered_population)               


    def EDA_Mutation(llm_model, filtered_list, mutation_prompt, epoch):
        prompt = "INSTRUCTION : " + mutation_prompt + "\ncontinue the following list. " + "\n\n" + filtered_list
        print("PROMPT :")
        print(prompt)
        response = LLMs.Response(llm_model,prompt, 1, 2500)
        return response
    
    def _test_EDA_Mutation():
        llm_model = 'gpt-3.5'
        dataset_name = 'gsm8k'
        file_path = llm_model + "_on_" + dataset_name + "/population_initialize.csv"
        df_population = pd.read_csv(file_path)
        population = df_population.to_numpy()
        filtered_list = Estimation_of_Distribution_mutation.Filtered_List(population[5:10])
        mutation = Estimation_of_Distribution_mutation.EDA_Mutation(llm_model, filtered_list, population[5][1],0)
        print("ANSWER : ")
        print(mutation)

    def EDA_Rank_Index_Mutation(llm_model, filtered_list, len_filtered_population, mutation_prompt, epoch):
        prompt = "INSTRUCTION: " + mutation_prompt + "\n A List of Responses in descending order of score." + str(len_filtered_population + 1) + "is the best response. It resembles" + str(len_filtered_population) + "more than it does (1)\n\n\n" + filtered_list
        print("PROMPT :")
        print(prompt)
        response = LLMs.Response(llm_model,prompt, 1, 2500)
        return response
    
    
    def _test_EDA_Rank_Mutation():
        llm_model = 'gpt-3.5'
        dataset_name = 'gsm8k'
        file_path = llm_model + "_on_" + dataset_name + "/population_initialize.csv"
        df_population = pd.read_csv(file_path)
        population = df_population.to_numpy()
        filtered_list , len_f = Estimation_of_Distribution_mutation.Filtered_List(population[5:10], True)
        mutation = Estimation_of_Distribution_mutation.EDA_Rank_Index_Mutation(llm_model, filtered_list, len_f, population[5][1],0)
        print("ANSWER : ")
        print(mutation)


    def List_Best_So_Far(best_so_fars):
        list = ""
        index = 1
        for p in best_so_fars:
            p1 = "PROMPT : " + p[0]
            p2 = "PROMPT : " + p[2]
            list += str(index) + " .\t" + p1 + " \n"
            index += 1
            list += str(index) + " .\t" + p2 + " \n"
            index += 1
        return list


    def Lineage_Based_Mutation(llm_model, best_so_fars, mutation_prompt, epoch):
        prompt = "INSTRUCTION: " + mutation_prompt + "\nGENOTYPES FOUND IN ASCENDING ORDER OF QUALITY: \n" + best_so_fars
        print("PROMPT :")
        print(prompt)
        response = LLMs.Response(llm_model,prompt, 1, 2500)
        return response




    
