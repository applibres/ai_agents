#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

#%%
model = OllamaLLM(model="gemma3:1b",temperature=0.2)

#%% first run
messages = [
        ("system", "You are an author and write a childs book.respond short and concise. End your answer with a specific question, that provides a new direction for the story."),
        ("user", "A mouse and a cat are best friends."),
    ]
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
output = chain.invoke({})
print(output)

# %% next run
messages.append(("ai", output))
messages.append(("user", "The dog is running after the cat."))

#print(messages)
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
output = chain.invoke({})
print(output)

# %%
