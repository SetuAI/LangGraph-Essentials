"""
================================================================================
 LangGraph Essentials  |  Module 02  |  YOUR FIRST COMPLETE GRAPH
================================================================================

WHAT WE ARE BUILDING
--------------------
A "news analyst" workflow that searches for news on a topic, grades the quality
of what it found, and either refines the search or writes the final briefing.

This is the first module where all four LangGraph primitives appear together:

        STATE   - the shared dictionary that flows through everything
        NODE    - a plain function: state in, partial update out
        EDGE    - a fixed connection: "after A, always run B"
        ROUTER  - a function that CHOOSES the next node at runtime

                            START
                              |
                              v
                       [search_node]           finds news items
                              |
                              v
                      [evaluate_node]          scores them 0-10
                              |
                   +----------+----------+
              score < threshold     score >= threshold
                   |                      |
                   v                      v
             [refine_node]          [summarize_node]   writes the briefing
                   |                      |
                   +----------+-----------+
                              v
                             END

NOTE THE SHAPE: refine_node flows FORWARD to summarize_node. It does NOT loop
back to evaluate_node. That means this graph can never run forever -- every path
reaches END in at most four nodes. Loops are powerful and we will build one in a
later module, but a loop needs a counter and an escape hatch. A first example
should not need either.
================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------

import os                                    # to read environment variables
from typing import TypedDict, Optional       # our state schema (see Module 01)

from dotenv import load_dotenv               # loads the .env file into os.environ
from pydantic import BaseModel, Field        # runtime validation (see Module 01)
from langchain_openai import ChatOpenAI      # the LLM client

# The three names we need from LangGraph:
#   StateGraph - the builder object we add nodes and edges to
#   START      - the virtual node representing "the beginning"
#   END        - the virtual node representing "stop here"
#
# START and END are not functions you write. They are sentinel values LangGraph
# provides so that entry and exit look like every other edge in your code.
from langgraph.graph import StateGraph, START, END


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

# Read .env so OPENAI_API_KEY becomes available to the ChatOpenAI client.
load_dotenv(override=True)

# Fail early with a clear message rather than deep inside an HTTP 401 traceback.
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env file containing:\n"
        "    OPENAI_API_KEY=sk-..."
    )

# ------------------------------------------------------------------------------
# !! READ THIS BEFORE TEACHING !!
#
# An LLM cannot browse the internet. It only knows what was in its training data,
# which has a cutoff date. If you ask it for "recent news", it will produce
# realistic-looking headlines with names, numbers and dates -- and INVENT them.
#
# So there are two honest ways to run this module:
#
#   USE_REAL_SEARCH = False  -> the LLM writes news from memory. Perfectly fine
#                               for teaching GRAPH MECHANICS, as long as you say
#                               out loud that the content is simulated.
#
#   USE_REAL_SEARCH = True   -> a real search tool fetches live results and the
#                               LLM only formats them. This is what a production
#                               system does. Needs: pip install langchain-tavily
#                               and a TAVILY_API_KEY in your .env
#
# ------------------------------------------------------------------------------
USE_REAL_SEARCH = True


# ------------------------------------------------------------------------------
# THE MODELS
# ------------------------------------------------------------------------------

# The "worker" model. temperature=0.2 gives it a little room to write naturally.
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# The "judge" model. temperature=0 makes it as deterministic as possible.
# A grader that returns 7 one run and 4 the next is not a grader -- it is noise.
# Using a separate object (not the same `llm`) makes that intent explicit.
judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ==============================================================================
# SECTION 1 - THE STATE
# ==============================================================================
#
# One TypedDict, shared by every node. Fields start empty and get filled in as
# execution proceeds. This is exactly the "half-filled state" idea from Module 01
# and is why we use TypedDict rather than a Pydantic model here.
# ------------------------------------------------------------------------------

class NewsAnalystState(TypedDict):
    """Shared state that flows through every node in the graph."""

    # --- filled by the caller, before the graph starts ---
    query: str                          # the topic to research
    quality_threshold: int              # score at or above which we skip refining

    # --- filled by search_node ---
    raw_results: Optional[str]          # the first set of news items

    # --- filled by evaluate_node ---
    quality_score: Optional[int]        # 0-10 grade for raw_results
    quality_reason: Optional[str]       # one-sentence justification for the grade

    # --- filled by refine_node (only on the poor-quality path) ---
    refined_results: Optional[str]      # a better second attempt

    # --- filled by summarize_node ---
    final_summary: Optional[str]        # the finished briefing
    route_taken: Optional[str]          # which branch ran, for teaching/debugging


# WHY quality_threshold LIVES IN THE STATE
# ----------------------------------------
# It would be easier to hard-code `if score < 6`. But putting it in the state
# lets you run the SAME graph twice with DIFFERENT thresholds and watch it take
# different paths. That is how you demonstrate a conditional edge without faking
# any data. 


# ==============================================================================
# SECTION 2 - STRUCTURED OUTPUT FOR THE EVALUATOR
# ==============================================================================
#
# The evaluator has to return a NUMBER and a SENTENCE. The tempting approach is
# to ask for "SCORE: 7" in the prompt and pull it out with a regular expression.
# Do not do that. It breaks the moment the model writes "Score: 7/10" or adds a
# preamble, and the usual fallback -- `score = 5 if parsing failed` -- silently
# turns a PARSE failure into a QUALITY failure. You then debug the wrong thing.
#
# This is the "validate at the doors" rule from Module 01. An LLM response is a
# door. So we define a Pydantic schema and let the model fill it.
# ------------------------------------------------------------------------------

class QualityEvaluation(BaseModel):
    """The exact shape we require back from the evaluator model."""

    # `ge=0, le=10` are enforced at RUNTIME. If the model returns 11, this raises
    # instead of quietly poisoning the routing decision downstream.
    #
    # The `description=` text is not a comment -- it is sent to the model as part
    # of the schema. Write it as an instruction, because that is what it is.
    score: int = Field(
        ...,
        ge=0,
        le=10,
        description="Overall quality of the research results, from 0 to 10.",
    )
    reason: str = Field(
        ...,
        description="One sentence explaining the score. Name the weakest criterion.",
    )


# `.with_structured_output(...)` wraps the model so that instead of returning a
# message full of text, it returns a real QualityEvaluation object -- already
# parsed and already validated. No regex, no fallback, no format instructions
# cluttering the prompt.
structured_judge = judge_llm.with_structured_output(QualityEvaluation)

# with_structured_output takes your model and wraps it. Under the hood it converts your
# Pydantic class into a JSON schema and sends that to OpenAI as a tool definition, 
# so the API is now constrained — it cannot reply with prose. 
# It must return JSON matching your shape. LangChain then parses that JSON and hands you back a real object.


'''
If you want the quality criterion to be enforced more strictlly : 

class QualityEvaluation(BaseModel):
    relevance: int = Field(..., ge=0, le=10, description="Does it answer the query?")
    specificity: int = Field(..., ge=0, le=10, description="Real names, numbers, dates?")
    recency: int = Field(..., ge=0, le=10, description="How current is the content?")
    coverage: int = Field(..., ge=0, le=10, description="Multiple angles covered?")
    reason: str = Field(..., description="One sentence naming the weakest criterion.")

    @property
    def score(self) -> float:
        """Overall score = mean of the four criteria, computed in Python."""
        return (self.relevance + self.specificity + self.recency + self.coverage) / 4

'''




# ==============================================================================
# SECTION 3 - THE NODES
# ==============================================================================
#
# THE NODE CONTRACT, one more time, because everything depends on it:
#
#       def node(state: MyState) -> dict:
#           ...
#           return {"only_the_keys": "I_changed"}
#
# A node NEVER returns the whole state. It returns a small dict of updates, and
# LangGraph merges that into the state for you. Returning the whole state is the
# most common beginner error -- it appears to work, then breaks the moment two
# nodes run in parallel.
# ------------------------------------------------------------------------------

def _banner(icon: str, name: str) -> None:
    """Print a node header. Purely cosmetic -- keeps the demo output readable."""
    print("\n" + "=" * 60)
    print(f"{icon}  NODE: {name}")
    print("=" * 60)


# ── Node 1 ────────────────────────────────────────────────────────────────────
def search_node(state: NewsAnalystState) -> dict:
    """
    Find news items about the topic in state["query"].

    Reads:   query
    Writes:  raw_results
    """
    _banner("[1]", "search_node")
    print(f"   Query : {state['query']}")

    if USE_REAL_SEARCH:
        # ---- REAL SEARCH PATH ------------------------------------------------
        # The tool fetches live pages; the LLM only reformats them. The model is
        # never asked to recall facts, so it cannot invent them.
        from langchain_tavily import TavilySearch

        search_tool = TavilySearch(max_results=5)
        hits = search_tool.invoke({"query": state["query"]})
        print(f"   Mode  : REAL search  ({len(hits.get('results', []))} hits)")

        response = llm.invoke([
            ("system",
             "Reformat the supplied search results into exactly this layout, "
             "one line per item:\n"
             "[N]. HEADLINE | DETAIL | SIGNIFICANCE\n"
             "Use ONLY facts present in the supplied results. Invent nothing."),
            ("human", f"Topic: {state['query']}\n\nSearch results:\n{hits}"),
        ])
    else:
        # ---- SIMULATED PATH --------------------------------------------------
        # Honest framing: we are asking the model to recall, not to research.
        print("   Mode  : SIMULATED (LLM recall -- content may be invented)")

        response = llm.invoke([
            ("system",
             "You are a news research assistant. Given a topic, write 5 news "
             "items about it from your training data. Include a headline, a key "
             "detail, and why it matters. Be specific with names, numbers and "
             "dates. Format each item as:\n"
             "[N]. HEADLINE | DETAIL | SIGNIFICANCE"),
            ("human", f"Find news about: {state['query']}"),
        ])

    raw_results = response.content

    # Compute the line count BEFORE the f-string. Cleaner than embedding chr(10).
    line_count = len(raw_results.splitlines())
    print(f"   Found : {line_count} lines")
    print(f"   Preview: {raw_results[:110]}...")

    # PARTIAL update -- one key only.
    return {"raw_results": raw_results}


# ── Node 2 ────────────────────────────────────────────────────────────────────
def evaluate_node(state: NewsAnalystState) -> dict:
    """
    Grade the quality of raw_results from 0 to 10.

    Reads:   query, raw_results
    Writes:  quality_score, quality_reason

    Note this node makes NO routing decision. It only produces a score. Deciding
    what to do with that score is the router's job (Section 4). Keeping "measure"
    and "decide" in separate functions is what makes graphs easy to change later.
    """
    _banner("[2]", "evaluate_node")

    # `structured_judge` returns a QualityEvaluation OBJECT, not a message.
    # So there is no .content, no parsing, and no chance of a malformed score.
    evaluation: QualityEvaluation = structured_judge.invoke([
        ("system",
         "You are a strict quality evaluator for news research. Grade the "
         "results on four criteria: (1) relevance to the query, (2) specificity "
         "- real names, numbers, dates, (3) recency, (4) coverage of multiple "
         "angles. Be harsh; a generic answer should score below 5."),
        ("human",
         f"Query: {state['query']}\n\nResults to grade:\n{state['raw_results']}"),
    ])

    print(f"   Score  : {evaluation.score}/10")
    print(f"   Reason : {evaluation.reason}")

    return {
        "quality_score": evaluation.score,
        "quality_reason": evaluation.reason,
    }


# ── Node 3 ────────────────────────────────────────────────────────────────────
def refine_node(state: NewsAnalystState) -> dict:
    """
    Second attempt, targeted at the specific weakness the evaluator named.

    Reads:   query, raw_results, quality_reason
    Writes:  refined_results, route_taken

    This runs ONLY on the poor-quality branch. Feeding the evaluator's critique
    back into the prompt is what makes this a correction rather than a retry.
    """
    _banner("[3]", "refine_node   <- POOR QUALITY PATH")
    print(f"   Weakness to fix: {state['quality_reason']}")

    response = llm.invoke([
        ("system",
         "You are an expert researcher performing a corrective second pass. "
         "The previous attempt was graded low for this specific reason: "
         f"'{state['quality_reason']}'. Fix that weakness directly. Be more "
         "concrete: named people, named companies, dates, figures. Format each "
         "item as:\n[N]. HEADLINE | DETAIL | SIGNIFICANCE"),
        ("human",
         f"Improve the research for: {state['query']}\n\n"
         f"Previous (weak) attempt:\n{state['raw_results']}"),
    ])

    refined = response.content
    print(f"   Preview: {refined[:110]}...")

    return {
        "refined_results": refined,
        "route_taken": "POOR -> refine_node -> summarize_node",
    }


# ── Node 4 ────────────────────────────────────────────────────────────────────
def summarize_node(state: NewsAnalystState) -> dict:
    """
    Write the final briefing.

    Reads:   query, refined_results OR raw_results, route_taken
    Writes:  final_summary, route_taken

    This node is reached from BOTH branches, so it must not assume which one ran.
    That is why it checks for refined_results and falls back to raw_results --
    a node that is a merge point has to handle every path that leads to it.
    """
    _banner("[4]", "summarize_node")

    # `state.get(...)` returns None for a missing/empty key instead of raising.
    # The `or` then falls through to raw_results. Reads as: "the refined version
    # if we produced one, otherwise the original".
    content = state.get("refined_results") or state.get("raw_results") or ""
    source = "refined" if state.get("refined_results") else "raw"
    print(f"   Using : {source} results")

    response = llm.invoke([
        ("system",
         "You are a senior intelligence analyst. Produce a clean briefing in "
         "exactly this structure:\n\n"
         "## News Briefing: <topic>\n\n"
         "### Key Developments\n"
         "<3-5 bullets, each a bolded headline plus 1-2 sentences>\n\n"
         "### Why This Matters\n"
         "<2-3 sentences of broader significance>\n\n"
         "### Bottom Line\n"
         "<one sentence -- the single most important takeaway>"),
        ("human", f"Topic: {state['query']}\n\nResearch:\n{content}"),
    ])

    print("   Briefing generated.")

    return {
        "final_summary": response.content,
        # If refine_node ran it already set route_taken; preserve it.
        # Otherwise this run came straight down the good path.
        "route_taken": state.get("route_taken") or "GOOD -> summarize_node",
    }


# ==============================================================================
# SECTION 4 - THE ROUTER
# ==============================================================================
#
# A router is NOT a node. Critical distinction:
#
#       A NODE   returns a dict  -> it CHANGES the state
#       A ROUTER returns a string -> it CHOOSES the next node
#
# A router must never modify state. Its only job is to answer "where next?".
# The string it returns is looked up in the mapping passed to
# add_conditional_edges (Section 5).
# ------------------------------------------------------------------------------

def quality_router(state: NewsAnalystState) -> str:
    """
    Decide whether the results need refining.

    Returns:
        "refine"    -> go to refine_node
        "summarize" -> go to summarize_node
    """
    score = state["quality_score"]
    threshold = state["quality_threshold"]

    if score < threshold:
        print(f"\n   >> ROUTER: {score} < {threshold}  -> refine_node")
        return "refine"

    print(f"\n   >> ROUTER: {score} >= {threshold}  -> summarize_node")
    return "summarize"


# ==============================================================================
# SECTION 5 - BUILDING THE GRAPH
# ==============================================================================
#
# Six steps, always in this order:
#   1. create the builder, bound to a state schema
#   2. register every node
#   3. connect START to the first node
#   4. add the fixed edges
#   5. add the conditional edge
#   6. compile
# ------------------------------------------------------------------------------

# --- Step 1: the builder, bound to our state schema ---------------------------
# StateGraph needs the schema so it knows what keys exist and how to merge
# the partial dicts your nodes return.
builder = StateGraph(NewsAnalystState)

# --- Step 2: register the nodes -----------------------------------------------
# add_node("name_used_in_edges", the_function)
# The string is the node's identity in the graph. The function is the behaviour.
builder.add_node("search_node", search_node)
builder.add_node("evaluate_node", evaluate_node)
builder.add_node("refine_node", refine_node)
builder.add_node("summarize_node", summarize_node)

# --- Step 3: where execution begins -------------------------------------------
# You may also see `builder.set_entry_point("search_node")` in older tutorials.
# It does exactly the same thing. Prefer this form: START and END then look like
# ordinary nodes, and there is only one idiom to remember instead of two.
builder.add_edge(START, "search_node")

# --- Step 4: the fixed edges --------------------------------------------------
# "After A, ALWAYS run B." No condition, no choice.
builder.add_edge("search_node", "evaluate_node")     # always grade what we found
builder.add_edge("refine_node", "summarize_node")    # a refine always leads to a summary
builder.add_edge("summarize_node", END)              # the summary is the last step

# --- Step 5: the conditional edge ---------------------------------------------
# "After evaluate_node, ask quality_router where to go."
#
# Three arguments:
#   1. the source node
#   2. the router function
#   3. a mapping from the router's return string -> the destination node name
#
# The mapping is optional (you can return node names directly) but you should
# always include it: it documents every possible branch in one place, and
# LangGraph can then draw the labelled diagram in Section 6.
builder.add_conditional_edges(
    "evaluate_node",
    quality_router,
    {
        "refine": "refine_node",        # router said "refine"
        "summarize": "summarize_node",  # router said "summarize"
    },
)

# --- Step 6: compile ----------------------------------------------------------
# compile() validates the structure -- unreachable nodes, missing destinations,
# no path to END -- and returns a runnable object. The builder is the blueprint;
# `app` is the working machine. Nothing executes until you call app.invoke().
app = builder.compile()
print("Graph compiled successfully.\n")


# ==============================================================================
# SECTION 6 - RUNNING IT
# ==============================================================================

def draw_graph() -> None:
    """
    Render the graph as a diagram.

    In Jupyter this shows a PNG. In a terminal it prints Mermaid text you can
    paste into mermaid.live. Get your students to look at this every single time
    they change the wiring -- reading the picture is how graph bugs get found.
    """
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:
        print("Mermaid source (paste into https://mermaid.live):\n")
        print(app.get_graph().draw_mermaid())


def run(query: str, quality_threshold: int, label: str) -> dict:
    """
    Execute the graph once and print the result.

    Args:
        query:             the topic to research.
        quality_threshold: the score needed to skip refining. Raise it to force
                           the poor-quality branch.
        label:             a heading for the console output.

    Returns:
        The final state after END is reached.
    """
    print("\n" + "#" * 70)
    print(f"# RUN: {label}   (threshold = {quality_threshold})")
    print("#" * 70)

    # NOTE: we pass only the two fields the caller is responsible for. We do NOT
    # need to pass raw_results=None, quality_score=None and so on. Every other
    # key is simply absent until a node writes it -- which is precisely why the
    # state is a TypedDict and not a Pydantic model (Module 01, Section 5).
    final_state = app.invoke({
        "query": query,
        "quality_threshold": quality_threshold,
    })

    # THE ORIGINAL VERSION FORGOT THIS PART. invoke() returns the final state,
    # and if you never print it, the graph does all its work invisibly.
    print("\n" + "-" * 70)
    print(f"PATH TAKEN : {final_state['route_taken']}")
    print(f"SCORE      : {final_state['quality_score']}/10")
    print("-" * 70)
    print(final_state["final_summary"])

    return final_state


if __name__ == "__main__":
    draw_graph()

    TOPIC = "Suggest what are the current NVIDIA AI GPU infrastructure news and developments as of 2026 . "

    # RUN A -- a normal threshold. The evaluator will usually pass the results,
    # so the graph goes straight to the summary. Three nodes execute.
    run(TOPIC, quality_threshold=6, label="GOOD-QUALITY PATH")

    # RUN B -- an unreasonably high bar. Almost nothing scores 9+, so the router
    # sends us through refine_node. Four nodes execute.
    #
    # Same graph. Same query. Different path. THIS is the demonstration -- and it
    # works without faking any data, because the threshold lives in the state.
    run(TOPIC, quality_threshold=9, label="POOR-QUALITY PATH")

