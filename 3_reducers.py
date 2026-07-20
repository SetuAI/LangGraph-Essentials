"""
LangGraph Essentials | Module 03 | REDUCERS


There are three different things in play

1. what the state HAD          ["started"]

2. what the node RETURNED      ["checked"]        <- the node's output

3. what the state BECOMES      ???                <- the reducer decides this


Line 2 is not the state. It's just what the function handed back. The reducer takes lines 1 and 2 and produces line 3.

Without a reducer (default = replace):

had:      ["started"]
returned: ["checked"]
becomes:  ["checked"]        the new value wins, "started" is gone


With add:
had:      ["started"]
returned: ["checked"]
becomes:  ["started", "checked"]

So the two possible outcomes are ["checked"] or ["started", "checked"]. 
Getting back just ["started"] would mean the node's work was thrown away — that never happens.

An easy way to remember it: think of add literally as the + sign, because that's exactly what it is: 

python["started"] + ["checked"]   ==   ["started", "checked"]
#    old          new                    result


"""

from typing import Annotated, TypedDict
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==============================================================================
# 1. THE DEFAULT IS REPLACE
# ==============================================================================

class SimpleState(TypedDict):
    counter: int          # no reducer
    log: list[str]        # no reducer


def demo_replace():
    banner("1 | DEFAULT RULE = REPLACE")

    def step_one(state):
        return {"counter": 1, "log": ["step one ran"]}

    def step_two(state):
        return {"counter": 2, "log": ["step two ran"]}

    builder = StateGraph(SimpleState)
    builder.add_node("one", step_one)
    builder.add_node("two", step_two)
    builder.add_edge(START, "one")
    builder.add_edge("one", "two")
    builder.add_edge("two", END)

    final = builder.compile().invoke({"counter": 0, "log": []})

    print(f"   counter : {final['counter']}     <- correct, we wanted the latest")
    print(f"   log     : {final['log']}   <- wrong, 'step one ran' was overwritten")
    print("\n   Same rule. Right for one field, wrong for the other.")
    print("   So the rule belongs to the FIELD, not to the graph.")


# ==============================================================================
# 2. PARALLEL NODES WITHOUT A REDUCER -> CRASH
# ==============================================================================

class BrokenState(TypedDict):
    findings: list[str]       # no reducer, and two nodes will write it


def demo_crash():
    banner("2 | TWO PARALLEL NODES, NO REDUCER")

    def check_a(state):
        return {"findings": ["A: concentration is high"]}

    def check_b(state):
        return {"findings": ["B: liquidity is thin"]}

    builder = StateGraph(BrokenState)
    builder.add_node("check_a", check_a)
    builder.add_node("check_b", check_b)

    # Two edges out of START = both nodes run in the SAME step.
    # This is how you get parallelism: the edge shape, nothing else.
    builder.add_edge(START, "check_a")
    builder.add_edge(START, "check_b")
    builder.add_edge("check_a", END)
    builder.add_edge("check_b", END)

    try:
        builder.compile().invoke({"findings": []})
    except Exception as exc:
        print(f"   {type(exc).__name__}: {exc}")
        print("\n   LangGraph got two values for 'findings' at once.")
        print("   It will not guess which one you meant. You have to say.")


# ==============================================================================
# 3. THE FIX
# ==============================================================================

class FixedState(TypedDict):
    #                    type       merge rule
    findings: Annotated[list[str], add]


def demo_fix():
    banner("3 | SAME GRAPH, ONE WORD ADDED")

    def check_a(state):
        return {"findings": ["A: concentration is high"]}

    def check_b(state):
        return {"findings": ["B: liquidity is thin"]}

    builder = StateGraph(FixedState)
    builder.add_node("check_a", check_a)
    builder.add_node("check_b", check_b)
    builder.add_edge(START, "check_a")
    builder.add_edge(START, "check_b")
    builder.add_edge("check_a", END)
    builder.add_edge("check_b", END)

    final = builder.compile().invoke({"findings": []})

    for item in final["findings"]:
        print(f"   - {item}")
    print("\n   findings: list[str]                 -> crash")
    print("   findings: Annotated[list[str], add] -> works")
    print("\n   Order is NOT guaranteed. Parallel nodes finish when they finish.")


# ==============================================================================
# 4. add_messages : THE REDUCER FOR CHAT HISTORY
# ==============================================================================

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def demo_add_messages():
    banner("4 | add_messages")

    def node_user(state):
        # A bare string works. add_messages converts it to a HumanMessage.
        return {"messages": "What is the risk on my portfolio?"}

    def node_assistant(state):
        return {"messages": AIMessage(content="Checking...", id="ans-1")}

    def node_correction(state):
        # Same id as above -> REPLACES that message instead of appending.
        return {"messages": AIMessage(content="Risk score is 7/10.", id="ans-1")}

    builder = StateGraph(ChatState)
    builder.add_node("user", node_user)
    builder.add_node("assistant", node_assistant)
    builder.add_node("correction", node_correction)
    builder.add_edge(START, "user")
    builder.add_edge("user", "assistant")
    builder.add_edge("assistant", "correction")
    builder.add_edge("correction", END)

    final = builder.compile().invoke({
        "messages": [HumanMessage(content="Hello", id="greeting-1")]
    })

    print(f"   {len(final['messages'])} messages:\n")
    for m in final["messages"]:
        print(f"   [{m.type:>5}] id={m.id[:12]:<12} {m.content}")

    print("\n   Three messages, not four. 'Checking...' was replaced,")
    print("   because both used id='ans-1'.")
    print("\n   add_messages does what add cannot:")
    print("     - converts strings/dicts into Message objects")
    print("     - replaces an existing message when the id matches")
    print("     - auto-assigns an id when you omit one")


# ==============================================================================
# SUMMARY
# ==============================================================================

SUMMARY = """
----------------------------------------------------------------------
 CHOOSING A REDUCER: is this field a SNAPSHOT or a HISTORY?

   SNAPSHOT -> default (replace)
     risk score, final decision, user query, status flag, counter

   HISTORY  -> reducer
     chat messages, audit log, findings from parallel checks,
     retrieved documents, accumulated errors

 DECLARATIONS
   x: int                              replace, newest wins
   x: Annotated[list, add]             old + new
   x: Annotated[list, add_messages]    chat-aware, replaces by id
   x: Annotated[T, my_func]            my_func(old, new)

 HARD RULE
   If two nodes can write the same field in the same step, that field
   MUST have a reducer. Otherwise: InvalidUpdateError.

 TWO MISTAKES EVERYONE MAKES ONCE

   return {"log": "done"}        WRONG  add(list, str) explodes
   return {"log": ["done"]}      RIGHT  return the field's own type

   state["log"].append(x)        WRONG  LangGraph never sees it
   return {"log": [x]}           RIGHT  nodes describe changes,
                                        they do not perform them
----------------------------------------------------------------------
"""

if __name__ == "__main__":
    demo_replace()
    demo_crash()
    demo_fix()
    demo_add_messages()
    print(SUMMARY)