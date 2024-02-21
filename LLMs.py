class LLMs :
  
  def response (llm_model, prompt) :
    if llm_model == 'gpt_3.5' :
      LLMs.gpt_response(prompt)
    if llm_model == 'gemini' :
      LLMs.gemini_response(prompt)
    if llm_model == 'palm' :
      LLMs.palm_response(prompt)
    if llm_model == 'llama' :
      LLMs.llama_response(prompt)
    if llm_model == 'mistral' :
      LLMs.mistral_response(prompt)
    

  def gpt_response(prompt, temperature=0) :
    import openai
    import os
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI()
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=temperature, )
    return response.choices[0].message.content
  
  def gemini_response(prompt) :
    import os
    from dotenv import load_dotenv, find_dotenv
    import google.generativeai as genai
    _ = load_dotenv(find_dotenv())
    genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, stream=False)
    return response.text
  
  def palm_response(prompt) :
    import os
    from dotenv import load_dotenv, find_dotenv
    import google.generativeai as genai
    _ = load_dotenv(find_dotenv())
    genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('text-bison-001')
    response = model.generate_content(prompt, stream=False)
    return response.text
  
  def llama_response():

    return 
  
  def mistral_response():

    return