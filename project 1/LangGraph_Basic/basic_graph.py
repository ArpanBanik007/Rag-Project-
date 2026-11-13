
from typing import TypedDict

from langgraph.graph import END, StateGraph


# 1. Define State
class State(TypedDict):
    number: int


# 2. Double Node
def double(state: State) -> dict:
    boardnum = state["number"]
    newnum = boardnum * 2

    print("In double")
    print("Number:", newnum)

    return {"number": newnum}


# 3. Finish Node
def finish(state: State) -> dict:
    boardnum = state["number"]

    print("In finish")
    print("Final Number:", boardnum)

    return {"number": boardnum}


# 4. Decision Function
def decision(state: State) -> str:
    if state["number"] < 100:
        return "double"
    else:
        return "finish"


# 5. Build Graph
builder = StateGraph(State)

builder.add_node("double", double)
builder.add_node("finish", finish)

# Entry Point
builder.set_entry_point("double")

# Conditional Edges
builder.add_conditional_edges(
    "double",
    decision,
    {
        "double": "double",
        "finish": "finish"
    }
)

# Finish node -> END
builder.add_edge("finish", END)


# 6. Compile Graph
graph = builder.compile()


# 7. Execute Graph
if __name__ == "__main__":

    result = graph.invoke({"number": 5})

    print("\nFinal Result:")
    print(result)




# -------------------- What I Learned --------------------

# 1. State:
# State is used to store and share data between different nodes.
# In this example, we use TypedDict to define the structure of the state.
# Here, the state contains one variable called "number".

# 2. Node:
# A node is a Python function that performs a specific task.
# Each node receives the current state as input.
# It can process the data and return an updated state.

# 3. StateGraph:
# StateGraph is used to create and manage the workflow.
# It connects different nodes and controls the flow of data.

# 4. Entry Point:
# The entry point defines which node will execute first.
# In this example, the "double" node is the starting node.

# 5. Edge:
# An edge connects one node to another node.
# It defines the execution flow between nodes.

# 6. Conditional Edge:
# A conditional edge decides which node will execute next.
# It uses a decision function to check a condition.
# If the number is less than 100, it executes the "double" node.
# Otherwise, it executes the "finish" node.

# 7. END:
# END is used to stop the graph execution.
# After the finish node completes its task, the graph stops.

# 8. Compile:
# compile() converts the StateGraph into an executable graph.

# 9. Invoke:
# invoke() starts the graph execution with the initial state.
# It returns the final updated state after execution.

# 10. Overall Workflow:
# First, the graph starts with the initial number.
# The double node multiplies the number by 2.
# The conditional edge checks whether the number is less than 100.
# If yes, the double node runs again.
# If no, the finish node runs and the graph stops.  