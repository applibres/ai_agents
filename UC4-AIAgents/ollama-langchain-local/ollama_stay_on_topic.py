#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


#Zero-shot classification prompt
template = """
You are a zero-shot text classifier.

Text:
\"\"\"{text}\"\"\"

Candidate Labels:
{labels}

Rules:
- Select ONLY one label.
- Answer with only the label.
- Do not explain.

Best label:
"""

#%% zero shot classifier
def zero_shot_classify(text, labels, model="gemma3:1b"):
    model = OllamaLLM(model=model,temperature=0)

    prompt = ChatPromptTemplate.from_messages(("system",template),)

    chain = (prompt | model | StrOutputParser())

    return chain.invoke({
        "text": text,
        "labels": ", ".join(labels)
    })



#%% guard medical
def guard_medical_prompt(prompt: str) -> str:
    candidate_labels = ["politics", "finance", "technology", "healthcare", "sports"]
    
    result=zero_shot_classify(
    prompt,
    candidate_labels)

    print(f"Label predicted:{result}")
    if result.strip().lower() == "healthcare":
        return "valid"
    else:
        return "invalid"


#%% guarded chain
def guarded_chain(user_input: str):
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answers questions about healthcare."),
        ("user", '{user_input}'),
    ])

    model = OllamaLLM(model="gemma3:1b",temperature=0)

    # Guard step
    if guard_medical_prompt(user_input) == "invalid":
        return "Sorry, I can only answer questions related to healthcare."
    
    # Proceed with the chain
    chain = prompt_template | model | StrOutputParser()
    return chain.invoke({"user_input": user_input})
    

#%% TEST guard_medical_prompt
#user_prompt = "Should I buy stocks of Apple, Google, or Amazon?"
#user_prompt = "I have a headache"
#guard_medical_prompt(user_prompt)

# %% TEST guarded_chain
user_prompt = "Should I buy stocks of Apple, Google, or Amazon?"
#user_prompt = "I have a headache"
#user_prompt = "What are the rules of football soccer"
#user_prompt = "How many pepople died during covid19 pandemic?"
answer=guarded_chain(user_prompt)
print(answer)
# %%
