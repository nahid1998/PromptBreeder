from LLMs import LLMs

class Direct_mutation :
    
    def Zero_order_Prompt_Generation(llm_model, problem_description) :
        prompt = problem_description + " \nA list of 100 hints : "
        response = LLMs.Response(llm_model, prompt, temperature=0, max_token=50)
        return response

    def First_order_Prompt_Generation(llm_model, mutation_prompt, task_prompt, ) :
        print("in mutation")
        prompt = mutation_prompt + "   INSTRUCTION: " + task_prompt + "   INSTRUCTION MUTANT: "
        response = LLMs.Response(llm_model, prompt, temperature=0, max_token=50)
        return response
    
