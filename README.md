# LangGraph Essentials

**From "what is a node" to a production-shaped agent with a human in the loop.**

A teaching repository for AI engineers who have just started their first job and
will shortly be asked to build agentic systems for an enterprise. Every module is
a runnable file, every example is drawn from insurance or asset management, and
nothing is hand-waved.


---

## Who this is for

You can write Python. You have used an LLM API. You have possibly copied a
LangGraph example off the internet, got it working, and could not have explained
why it worked.

That last sentence is the problem this repository solves.

**You do not need:** prior LangGraph experience, a cloud account, or a background
in insurance or finance. The domain examples explain themselves.

---

## The idea behind the sequence

Most LangGraph tutorials start with `create_react_agent`, produce something
impressive in fifteen lines, and leave you unable to debug it. This repository
goes the other way round. You build each mechanism by hand first, feel where it
hurts, and only then meet the shortcut that removes the pain.

So the order is deliberate:

| You learn | Because the next thing is impossible without it |
|---|---|
| State | A node has nothing to read or write |
| Nodes and edges | There is no graph |
| Reducers | Parallel nodes crash and agent loops forget |
| Tools | The model can only talk, never act |
| The agent loop | Nothing decides for itself |
| Human-in-the-loop | Nobody sane deploys it |

---

## Repository layout

```
LangGraph-Essentials/
│
├── 01_state_typeddict_vs_pydantic.py     State. TypedDict vs Pydantic.
├── 02_news_analyst_workflow.py           First complete graph. Conditional edges.
├── 03a_reducers.py                       Reducers, in isolation.
├── 03b_portfolio_workflow.py             Reducers applied. Parallel nodes. Streaming.
├── 04_claim_settlement_agent.py          Tools, agent loop, scoring, human approval.
│
├── .env                                  Your API keys. Never commit this.
├── .env.example                          Template. Commit this instead.
├── requirements.txt
└── README.md
```

Files are numbered in teaching order. Run them in order. Each one assumes the
previous one has landed.

---

## Setup

```bash
git clone <your-repo-url>
cd LangGraph-Essentials

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # then open .env and paste your keys
```

`.env` needs:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...            # only for module 02 with real search
```

### If you get a 401

This trips up nearly everyone once. `load_dotenv()` does **not** overwrite a
variable that already exists in your shell. If you ever ran
`export OPENAI_API_KEY=...` with an old key, that stale key wins and your `.env`
is silently ignored.

```bash
echo $OPENAI_API_KEY               # anything printed here is your culprit
unset OPENAI_API_KEY
```

Every file in this repo uses `load_dotenv(override=True)` for exactly this
reason. Check your `.env` has no spaces around `=`, no quotes, and no trailing
space after the key.

---

## The modules

### 01 — State: TypedDict vs Pydantic

**No API key needed.**

Before you can build a graph you must decide what shape your state is and who
guarantees that shape is correct.

Runs the same broken payload — a missing field, three wrong types, an unknown key
— through a plain dict, a TypedDict, and a Pydantic model. The dict and the
TypedDict accept all of it without a murmur. Pydantic reports four errors, each
pointing at the line that caused it.

**The takeaway:** TypedDict is checked by your editor and erased at runtime.
Pydantic is checked at runtime. Both are correct tools for different jobs.

**The rule this repo follows throughout:**

> Validate at the doors, not in the corridors.
> The doors are Pydantic. The corridor is a TypedDict.

Doors are anywhere untrusted data enters — an API request, an LLM's response, a
tool's arguments. The corridor is your graph state, passing between nodes that
have already been checked.

**Why graph state is a TypedDict and not a Pydantic model:** graph state is a
work-in-progress document. It is *supposed* to be half-empty at node one. A
Pydantic model demands a complete, valid object every time it is touched, which
is exactly the condition graph state is never in until END.

---

### 02 — Your first complete graph

**Needs an API key.**

A news analyst that searches a topic, grades the quality of what it found, and
either refines the search or writes the final briefing.

This is where all four primitives appear together for the first time:

- **State** — the shared dictionary
- **Node** — a plain function: state in, partial update out
- **Edge** — a fixed connection: after A, always B
- **Router** — a function that picks the next node at runtime

**A router is not a node.** This distinction confuses everyone once:

```
A NODE   returns a dict    ->  it CHANGES the state
A ROUTER returns a string  ->  it CHOOSES the next node
```

**Two design decisions worth understanding:**

*The quality threshold lives in the state, not in the code.* This means you can
run the identical graph twice with different thresholds and watch it take
different paths — a live demonstration of a conditional edge that works every
time, without faking any data.

*`refine_node` flows forward to `summarize_node` rather than back to
`evaluate_node`.* That means this graph cannot loop forever. Loops are powerful
and come later, but a loop needs a counter and an escape hatch, and a first
example should need neither.

**An honesty note that matters.** An LLM cannot browse the web. Ask one for
"recent news" and it will produce realistic headlines with names, numbers and
dates, and invent them. The same model then grades its own fabrication highly.
The `USE_REAL_SEARCH` flag makes this choice explicit rather than hidden. Set it
to `True` with a Tavily key for real results; leave it `False` and say out loud
that the content is simulated. A pipeline that hallucinates and then approves its
own hallucination is the most common beginner agent mistake there is.

---

### 03a — Reducers

**No API key needed.**

When a node returns `{"x": 5}`, LangGraph has to write that 5 into the state. A
**reducer** is the rule for how.

The word comes from `functools.reduce` — turning many values into one. A reducer
is any function that takes two things and returns one:

```python
def reducer(old, new):
    return ...one value...
```

At every step LangGraph holds two values and needs one: what is already in state,
and what the node just returned.

```
default rule    ->  return new              REPLACE, the old value is discarded
Annotated[list, add]  ->  return old + new  COMBINE, both survive
```

There is no such thing as a field without a reducer. There is only the default
one and one you chose.

**Four demos, each a throwaway graph:**

1. The default is replace — `counter` ends correctly, `log` loses a line. Same
   rule, right for one field and wrong for the other. So the rule belongs to the
   **field**, not the graph.
2. Two parallel nodes, no reducer — a real `InvalidUpdateError`. LangGraph got
   two values for one key in one step. It will not guess.
3. The same graph with one word added — works.
4. `add_messages` — the reducer built for chat history.

**`add_messages` is not a history of your state.** It only ever sees two things:
the messages currently in the field and the messages a node just returned. It is
`add`, specialised for messages: it converts bare strings into Message objects,
assigns ids, and — the part that matters — **replaces** an existing message when
a new one carries the same id, instead of appending a duplicate.

**Choosing a reducer — ask: is this field a snapshot or a history?**

| Snapshot (default, replace) | History (needs a reducer) |
|---|---|
| current risk score | chat messages |
| final decision | audit log |
| the user's query | findings from parallel checks |
| a status flag | retrieved documents |
| retry counter | anything written by parallel nodes |

**The hard rule:** if two nodes can write the same field in the same step, that
field **must** have a reducer. Not "should" — LangGraph raises.

**Two mistakes everyone makes once:**

```python
return {"log": "done"}        # WRONG — add(list, str) explodes
return {"log": ["done"]}      # RIGHT — return the field's own type

state["log"].append(x)        # WRONG — LangGraph never sees it
return {"log": [x]}           # RIGHT — nodes describe changes,
                              #         they do not perform them
```

---

### 03b — Portfolio risk review

**No API key needed.**

A risk desk at an asset management firm. A portfolio arrives, three independent
checks run on it — concentration, liquidity, credit — the results combine into
one score, and the portfolio is either auto-approved or escalated to the
investment committee.

This is where reducers stop being academic. Remove the reducer on `findings` and
`aggregate` averages one finding instead of three. The risk score is wrong, no
error appears anywhere, and the portfolio is approved on a third of the evidence.

**Parallelism is free and invisible.** No threads, no async, no locks, no join.
Three edges out of a node and three edges into another:

```python
# FAN-OUT — all three run in the same step
builder.add_edge("load_portfolio", "concentration_check")
builder.add_edge("load_portfolio", "liquidity_check")
builder.add_edge("load_portfolio", "credit_check")

# FAN-IN — LangGraph waits for ALL THREE before running aggregate.
# You write no waiting logic. The edges declare the dependency.
builder.add_edge("concentration_check", "aggregate")
builder.add_edge("liquidity_check", "aggregate")
builder.add_edge("credit_check", "aggregate")
```

The shape of the edges **is** the parallelism.

**Order is not guaranteed.** Run it a few times — findings come back in different
orders. If order matters, sort afterwards or timestamp each item.

**The checks are plain Python, not LLM calls, on purpose.** If a check called a
model, `severity: 8.0` would be unexplainable. Here you can read
`8.0 if heavy else 2.0` and trace every number. Swapping any check for an LLM
later is a one-function change — which is itself the architectural point.

**The audit log is not decoration.** In insurance and asset management, "why was
this approved?" is a regulatory question. The reducer is what makes an answer
exist.

**Stream modes**, at the end of the file:

| Mode | Yields | Use it for |
|---|---|---|
| `invoke()` | final state only | scripts, batch jobs |
| `stream("updates")` | each node's delta | debugging, teaching, progress bars |
| `stream("values")` | full state per step | showing state fill up |
| `stream("debug")` | verbose internals | deep troubleshooting |
| `stream("messages")` | LLM tokens as generated | chat UIs |
| `stream(["a","b"])` | several, tagged `(mode, payload)` | UI plus logging |

Streaming does not make the graph faster. It changes when you **see** it.

---

### 04 — Claim settlement agent with a live adjuster

**Needs an API key. This is the capstone.**

**The problem.** A motor insurer receives thousands of claims a day. Most are
small and obvious — settling them by hand is slow and expensive. Some are large
or suspicious — settling those automatically is reckless. So the question is not
"can a machine settle claims", it is: *which claims need a human, and how does
that human step in without stalling everything else?*

**What it does.**

1. An **agent** investigates using tools — policy, claim history, garage
   estimate. The LLM chooses which tools to call and how many times.
2. **Python** scores the claim. Not the LLM.
3. A **router** checks score and amount. Small and clean settles automatically.
   Large or risky stops and asks a person.
4. The **adjuster** approves, modifies the amount, or rejects — and each answer
   takes a different path.

```
START -> [investigate] <--+
              |           | tool calls
              v           |
           [tools] -------+
              |  no more tool calls
              v
          [assess]
              |
      +-------+--------+
  low risk          high risk or big payout
  small amount            |
      |                   v
      |            [human_review]   <-- graph PAUSES here
      |                   |
      |        +----------+----------+
      |     approve     modify     reject
      |        |          |          |
      v        v          v          v
 [auto_settle]  [settle]        [deny]
      |             |              |
      +-------------+--------------+
                    v
                   END
```

#### Four things to understand here

**1. The loop is what makes it an agent.**
`investigate -> tools -> investigate` is a cycle. The model decides how many
times it goes round, not you. That is the entire difference between a workflow
and an agent. Everything in modules 02 and 03 was a workflow.

**2. The LLM does not run your code.**
It cannot. It has no interpreter. When you "give a model tools" you send it
*descriptions* of your functions. The model replies with a **request** — just
structured JSON saying "please call `get_policy('POL-1001')`". Your code runs it
and sends the result back. `ToolNode` is that exchange, packaged. Nothing magic.

The exit condition is simple: **no `tool_calls` means the model is done.**

**3. It only works because of `add_messages`.**
Each pass appends tool results to the conversation. Swap in the default reducer
and the agent forgets every result, re-requests the same tools, and spins until
it hits the recursion limit. Worth breaking on purpose once.

**4. The LLM gathers. Python decides.**
`assess()` is arithmetic — `if not policy_active: score += 5.0`. The amount that
leaves the bank is never chosen by a language model. That is not timidity, it is
auditability: when a regulator asks "why ₹1,50,000?", there is a readable answer.

#### The human is a branch, not a gate

`interrupt()` stops the graph mid-node and returns control to your program. The
state is saved to a checkpointer.

**While it waits, nothing is running.** No thread is blocked. No tokens are being
spent. The pause can last a second or three days. You resume with
`Command(resume=<what the human said>)`, and that value becomes the return value
of `interrupt()` inside the node — execution continues from that exact line.

This is why a **checkpointer is required** for human-in-the-loop. Without
somewhere to save the paused state, there is nothing to come back to.

`ask_adjuster()` deliberately lives **outside** the graph. The graph does not
know whether the answer arrived from a terminal, a web form, a Slack button, or a
queue three days later. It only knows a dict came back. Swapping the terminal
prompt for a real adjuster queue changes one function and zero graph code.

#### What to expect when you run it

**Claim CLM-5001** — active policy, no prior claims, network garage, ₹62,000
estimate. Risk 0.0. Minus the ₹5,000 deductible → **auto-settles at ₹57,000**, no
human involved.

**Claim CLM-5002** — three claims in three years, garage outside the network,
₹3,85,000 estimate. Risk 7.5. **The graph stops and asks you.**

Let the cursor sit there for ten seconds before you type. That silence is the
lesson.

Then run it three times and answer differently each time:

| Your answer | Outcome |
|---|---|
| `1` approve | SETTLED ₹3,75,000 |
| `2` modify, 150000 | SETTLED ₹1,50,000, both figures on the audit trail |
| `3` reject | DENIED, different terminal node entirely |

Same claim. Same code. Three business outcomes. The only thing that changed was a
keystroke.

---

## The four primitives, on one page

```python
# STATE — the shared dictionary that flows through everything
class MyState(TypedDict):
    query: str                                  # replace (default)
    findings: Annotated[list, add]              # accumulate
    messages: Annotated[list, add_messages]     # chat history

# NODE — state in, PARTIAL update out. Never return the whole state.
def my_node(state: MyState) -> dict:
    return {"findings": [something]}

# ROUTER — returns a STRING. Never changes state.
def my_router(state: MyState) -> str:
    return "path_a" if state["score"] > 5 else "path_b"

# GRAPH
builder = StateGraph(MyState)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")                    # fixed edge
builder.add_conditional_edges("my_node", my_router,   # conditional edge
                              {"path_a": "a", "path_b": "b"})
app = builder.compile(checkpointer=InMemorySaver())   # checkpointer for HITL
```

---

## Errors you will hit, and what they mean

| Error | Cause | Fix |
|---|---|---|
| `InvalidUpdateError: can receive only one value per step` | Two parallel nodes wrote one field | Add a reducer: `Annotated[list, add]` |
| `401 invalid_api_key` | Stale key in your shell overriding `.env` | `unset OPENAI_API_KEY`, use `load_dotenv(override=True)` |
| `GraphRecursionError` | A cycle with no exit, often a forgetful agent | Check your loop's exit condition and `add_messages` |
| `TypeError: can only concatenate list` | Returned a bare item to a list field | `return {"log": ["x"]}` not `{"log": "x"}` |
| State change silently lost | Mutated state in place | Return the update, do not `.append()` |
| `interrupt()` does nothing | No checkpointer | `compile(checkpointer=InMemorySaver())` |

**Reading a LangGraph traceback:** it will be dominated by Pregel internals —
`runner.tick`, `run_with_retry`, `_runnable.invoke`. Ignore all of it. Read from
the **bottom up**, and look for the line `During task with name '<node>'`. That
tells you exactly which node failed.

---

## Roadmap

| Module | Topic |
|---|---|
| 05 | Checkpointers, threads, short vs long-term memory, time travel |
| 06 | Subgraphs, `Command`, the `Send` API for dynamic map-reduce |
| 07 | Multi-agent: supervisor, network, hierarchical teams |
| 08 | Retries, error handling, durable execution, caching |
| 09 | Testing graphs, LangSmith tracing, evals |
| 10 | Deployment: LangGraph Platform, FastAPI, Docker |

---

## Case study domains

Two domains run through the repository, chosen because they are rich in
documents, rules, approvals and audit requirements — which is what agentic
systems are actually bought for.

**Insurance** — claims intake, triage, coverage checking, fraud signals, adjuster
approval, payout. Natural human-in-the-loop moments and a real compliance story.

**Asset management** — portfolio risk review, investment memos, RFP responses,
suitability checks. More numeric, more parallel, better for demonstrating
fan-out.

**All data in this repository is synthetic and invented for teaching.** No real
policies, claims, holdings or customers appear anywhere.

---

## A note on style

Every file in this repository is written to be read aloud. Comments explain
**why** a line exists, not what it does. Concept files are separate from
application files, so a participant learning reducers is not simultaneously
learning about portfolio risk.

Where a design choice was made, the reasoning is in the file. Where an example is
simplified, the simplification is stated rather than hidden.

---

## Contributing

Found something unclear? That is a bug. Open an issue describing what confused
you and at which line — confusion is more useful feedback than a patch.

---

*Tarka Upskilling and Engineering Co. — enterprise AI training.*
*[tarkaupskilling.com](https://tarkaupskilling.com)*
