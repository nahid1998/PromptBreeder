import openai
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
_ = load_dotenv(find_dotenv())

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

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
    generation_config = genai.GenerationConfig(
      stop_sequences= None,
      temperature= temperature,
      max_output_tokens= max_token
    )
    response = model.generate_content(
      contents=prompt,
      generation_config= generation_config,
      stream=False)
    return response.text
  
  def palm_response(prompt,temperature, max_token) :
    genai.configure(api_key= os.getenv('GOOGLE_API_KEY'))
    model = genai.GenerativeModel('text-bison-001')
    response = model.generate_content(prompt, stream=False)
    return response.text
  
  def llama_response(prompt,temperature, max_token):

    return 
  
  def mistral_response(prompt,temperature, max_token):

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    # Setting `max_new_tokens` allows you to control the maximum length
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_token)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response
  



############################# TEST #####################################