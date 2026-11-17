#%%packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


#%%set up prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that translates English into another language."),
    ("user", "Translate this sentence: '{input}' into {target_language}"),
])

#prompt_template.invoke({"input": "I love programming.", "target_language": "Japanese"})
model = OllamaLLM(model="gemma3:1b",temperature=0.1)

#%%chain
chain = prompt_template | model | StrOutputParser()

#%%invoke chain
res = chain.invoke({"input": "I love programming.", "target_language": "Spanish"})
#res = chain.invoke({})

#res
print(res)
# %%
# %%