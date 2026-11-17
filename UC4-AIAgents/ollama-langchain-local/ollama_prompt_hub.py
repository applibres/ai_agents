#%% packages
from langchain_classic import hub
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint

#from dotenv import load_dotenv
#load_dotenv('.env')

#%% fetch prompt
prompt = hub.pull("hardkothari/prompt-maker")

#%% get input variables
prompt.input_variables
print(prompt.input_variables)

#%% model
#model = ChatOpenAI(model="gpt-4o-mini", 
#                   temperature=0)
model = OllamaLLM(model="gemma3:1b",temperature=0.7, topp=0.9)

#%% chain
chain = prompt | model | StrOutputParser()

# %% invoke chain
lazy_prompt = "summer, vacation, beach"
task = "Shakespeare poem"

#lazy_prompt = "optimal, path, navigate"
#task = "Gymnasium Frozen Lake v1 optimal navigation"
improved_prompt = chain.invoke({"lazy_prompt": lazy_prompt, "task": task})
#%%
print("Improved prompt:...")
print(improved_prompt)

#%% run model with improved prompt
print("Run model with improved prompt:...")
res = model.invoke(improved_prompt)
print(res)

# %%
#print("Run again the model with lazy prompt:...")
#res = model.invoke("summer, vacation, beach, Shakespeare poem")
#res = model.invoke("optimal, path, navigate, Gymnasium Frozen Lake")
#print(res)
# %%
