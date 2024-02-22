from LLMs import LLMs

class Direct_mutation :
    
    def Zero_order_Prompt_Generation(llm_model, problem_description) :
        prompt = problem_description + " \nA list of 100 hints : "
        response = LLMs.response(llm_model, prompt)
        return response

    def First_order_Prompt_Generation(llm_model, mutation_prompt, task_prompt) :
        print("in mutation")
        prompt = mutation_prompt + "   INSTRUCTION: " + task_prompt + "   INSTRUCTION MUTANT: "
        response = LLMs.response(llm_model, prompt)
        return response
    
################################### TEST #####################################################

m = """Say that instruction again in another way. DON’T use any of the words in the original instruction there’s a good chap."""
p = """Solve the math word problem, giving your answer as an arabic numeral."""

response  = Direct_mutation.First_order_Prompt_Generation("gpt-3.5", m, p)

print(response)
