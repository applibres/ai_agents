#%%packages
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#%%load dataset
persist_directory = "rag_store"
if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, 
        embedding_function=OllamaEmbeddings(model="embeddinggemma"))
else:
    data = WikipediaLoader(
        query="Human History",
        load_max_docs=50,
        doc_content_chars_max=1000000,
    ).load()

    # split the data
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, 
        chunk_overlap=200).split_documents(data)

    # create persistent vector store
    vector_store = Chroma.from_documents(chunks, 
        embedding=OllamaEmbeddings(model="embeddinggemma"), 
        persist_directory="rag_store")

#%%
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)


#%% bundle everything in a function
def simple_rag_system(question: str) -> str:
    
    ##Retrieval##
    relevant_docs = retriever.invoke(question)
    
    #%% print content of relevant docs
    for doc in relevant_docs:
        print(doc.page_content[: 100])
        print("\n--------------")

    ##Augmentation##
    context = "\n".join([doc.page_content for doc in relevant_docs])
    messages = [
        ("system", "You are an AI assistant that can answer questions \
            about the history of human civilization. \
            You are given a question and a list of documents \
            and need to answer the question.\
            Answer the question only based on these documents.\
            These documents can help you answer the question: {context}.\
            If you are not sure about the answer, you can say \
            'I don't know' or 'I don't know the answer to that question.'"),
        ("human", "{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    model = OllamaLLM(model="gemma3:1b",temperature=0.1)
    
    ##Generation##
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer

#%% Testing the function
#question = "What is a black hole?"
question = "What was the main cause for World War I?"
#question = "What are the most important paints from Picasso?"

answer=simple_rag_system(question=question)
print(answer)
#%%
