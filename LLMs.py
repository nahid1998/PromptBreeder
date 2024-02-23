import openai
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
_ = load_dotenv(find_dotenv())

class LLMs :
  
  def Response (llm_model, prompt, temperature, max_token) :
    if llm_model == "gpt-3.5" :
      return LLMs.gpt_response(prompt, temperature, max_token)
    if llm_model == 'gemini' :
      return LLMs.gemini_response(prompt, temperature, max_token)
    if llm_model == 'palm' :
      return LLMs.palm_response(prompt, temperature, max_token)
    if llm_model == 'llama' :
      return LLMs.llama_response(prompt, temperature, max_token)
    if llm_model == 'mistral' :
      return LLMs.mistral_response(prompt, temperature, max_token)
    

  def gpt_response(prompt, temperature, max_token) :
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI()
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, temperature=temperature, max_tokens=max_token )
    return response.choices[0].message.content
  
  def gemini_response(prompt,temperature, max_token) :
    genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, stream=False)
    return response.text
  
  def palm_response(prompt,temperature, max_token) :
    genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('text-bison-001')
    response = model.generate_content(prompt, stream=False)
    return response.text
  
  def llama_response(prompt,temperature, max_token):

    return 
  
  def mistral_response(prompt,temperature, max_token):

    return
  



############################# TEST #####################################