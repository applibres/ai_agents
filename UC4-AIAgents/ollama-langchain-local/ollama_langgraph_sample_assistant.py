#%% packages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import OllamaLLM
from IPython.display import Image, display

#%% define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]

#%% set up the assistant
llm = OllamaLLM(model="gemma3:1b")



def assistant(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

#%% create the graph
graph_builder = StateGraph(State)
graph_builder.add_node("assistant", assistant)
graph_builder.add_node("assistant2", assistant)
graph_builder.add_edge(START, "assistant")
graph_builder.add_edge("assistant", "assistant2")
graph_builder.add_edge("assistant2", END)

#%% compile the actual graph
graph = graph_builder.compile()

#%% display graph
display(Image(graph.get_graph().draw_mermaid_png()))

#%% invoke the graph
res = graph.invoke({"messages": [("user", "What do you know about LangGraph?")]})
#%% display the result
#res["messages"]

#%%
from pprint import pprint
pprint(res["messages"])
#%% extension ideas: add memory
