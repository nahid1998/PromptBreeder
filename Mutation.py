class Direct_mutation :
    
    def Zero_order_Prompt_Generation(llm_model, problem_description) :
        from LLMs import LLMs
        prompt = problem_description + " \nA list of 100 hints : "
        response = LLMs.response(llm_model, prompt)
        return response

    def First_order_Prompt_Generation(llm_model, mutation_prompt, task_prompt) :
        from LLMs import LLMs
        prompt = mutation_prompt + "   INSTRUCTION: " + task_prompt + "   INSTRUCTION MUTANT: "
        response = LLMs.response(llm_model, prompt)
        return response