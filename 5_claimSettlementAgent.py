"""
================================================================================
 LangGraph Essentials | Module 04 | CLAIM SETTLEMENT AGENT WITH A LIVE ADJUSTER
================================================================================

 THE PROBLEM
 -----------
 A motor insurance company receives thousands of claims a day. Two facts are in
 tension:

   - Most claims are small and obvious. Paying them by hand is slow and costly.
   - Some claims are large or suspicious. Paying those automatically is reckless.

 So the business question is not "can a machine settle claims?" It is:

        WHICH claims can be settled without a person,
        and for the rest, how does a person get involved
        without stopping the whole pipeline?

 WHAT WE ARE BUILDING
 --------------------
 A graph that investigates a claim, scores it, and then decides WHO decides.

   1. An AGENT gathers facts using tools (policy, claim history, garage quote).
      The LLM chooses which tools to call and how many times.

   2. PYTHON scores the claim. Not the LLM. The amount that leaves the bank is
      decided by arithmetic you can read, because a regulator will ask why.

   3. A ROUTER checks the score and the amount:
        small and clean   -> settle automatically, no human
        large or risky    -> PAUSE and ask a human adjuster

   4. The ADJUSTER can APPROVE, MODIFY the amount, or REJECT. Each answer sends
      the graph down a different path. The human is a branch, not a rubber stamp.

    START -> [investigate] <--+
                  |           | tool calls
                  v           |
               [tools] -------+
                  |  no more tool calls
                  v
              [assess]  score = f(evidence)
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
     [auto_settle] [settle] [settle]   [deny]
          |             |               |
          +-------------+--------------+
                        v
                       END

================================================================================
"""

import os
import sys
from operator import add
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)          # override=True so .env beats a stale shell var
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing. Put it in a .env file.")


# ==============================================================================
# SYNTHETIC DATA -- invented for teaching. Not real policies or claims.
# ==============================================================================

POLICIES = {
    "POL-1001": {"holder": "A. Rao",   "sum_insured": 800000, "active": True,
                 "deductible": 5000,  "vintage_years": 6},
    "POL-1002": {"holder": "R. Menon", "sum_insured": 450000, "active": True,
                 "deductible": 10000, "vintage_years": 1},
}

CLAIM_HISTORY = {
    "POL-1001": {"claims_last_3y": 0, "total_paid": 0},
    "POL-1002": {"claims_last_3y": 3, "total_paid": 210000},
}

GARAGE_ESTIMATES = {
    "CLM-5001": {"garage": "Authorised - Andheri", "estimate": 62000,  "network": True},
    "CLM-5002": {"garage": "Local - Unlisted",     "estimate": 385000, "network": False},
}


# ==============================================================================
# TOOLS
# ==============================================================================
# The docstring is the model's only instruction on when to use a tool.

@tool
def get_policy(policy_id: str) -> dict:
    """Fetch policy details: holder, sum insured, whether it is active, the
    deductible, and how long the customer has been insured."""
    return POLICIES.get(policy_id, {"error": "policy not found"})


@tool
def get_claim_history(policy_id: str) -> dict:
    """Fetch how many claims this policy made in the last 3 years and the total
    already paid out. Use this to judge whether the pattern is unusual."""
    return CLAIM_HISTORY.get(policy_id, {"error": "no history"})


@tool
def get_repair_estimate(claim_id: str) -> dict:
    """Fetch the garage repair estimate: garage name, quoted amount, and whether
    the garage is in the approved network."""
    return GARAGE_ESTIMATES.get(claim_id, {"error": "no estimate"})


TOOLS = [get_policy, get_claim_history, get_repair_estimate]


# ==============================================================================
# STATE
# ==============================================================================

class ClaimState(TypedDict):
    claim_id: str
    policy_id: str
    claim_text: str

    # grows every loop -> needs a reducer
    messages: Annotated[list, add_messages]

    risk_score: Optional[float]
    risk_reasons: Optional[list[str]]
    payout_amount: Optional[float]

    human_decision: Optional[str]
    human_note: Optional[str]
    final_status: Optional[str]

    # every node appends -> needs a reducer
    audit: Annotated[list[str], add]


# ==============================================================================
# THE AGENT
# ==============================================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

PROMPT = """You are a motor insurance claim investigator.

Gather the facts needed to assess this claim using the tools available: the
policy, the claim history, and the garage estimate. Call every tool you need.

When you have the facts, reply with a short factual summary. Do not decide
whether to pay -- that is not your job. Just state what you found."""


def investigate(state: ClaimState) -> dict:
    """
    The agent. Reads the conversation and either calls a tool or answers.

    Called repeatedly. On the first pass it sees only the claim; on later passes
    it also sees the tool results, because add_messages accumulated them. That
    accumulation is the only reason the loop terminates.
    """
    if not state["messages"]:
        messages = [
            SystemMessage(content=PROMPT),
            HumanMessage(content=(
                f"Claim {state['claim_id']} on policy {state['policy_id']}.\n"
                f"Customer says: {state['claim_text']}"
            )),
        ]
    else:
        messages = state["messages"]

    response = llm_with_tools.invoke(messages)
    n = len(response.tool_calls)
    return {
        "messages": [response],
        "audit": [f"investigate: requested {n} tool(s)" if n else "investigate: finished"],
    }


def should_continue(state: ClaimState) -> str:
    """Agent loop exit condition: no tool_calls means the model is done."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "assess"


# ToolNode reads tool_calls off the last message, runs each tool, and appends
# the results. It is the manual request/execute/reply loop, packaged.
tool_node = ToolNode(TOOLS)


# ==============================================================================
# SCORING -- deterministic, never the LLM
# ==============================================================================

class ClaimFacts(BaseModel):
    """Facts pulled out of the agent's free-text summary."""
    estimate_amount: float = Field(..., ge=0, description="Garage repair estimate in INR")
    in_network_garage: bool = Field(..., description="Is the garage in the approved network?")
    claims_last_3y: int = Field(..., ge=0, description="Claims in the last 3 years")
    policy_active: bool = Field(..., description="Is the policy currently active?")


fact_extractor = llm.with_structured_output(ClaimFacts)


def assess(state: ClaimState) -> dict:
    """
    Turn findings into a risk score and a payout figure.

    Step 1: extract structured facts from the summary (Pydantic at the door).
    Step 2: score them with plain arithmetic you can read and defend.
    """
    summary = state["messages"][-1].content
    facts = fact_extractor.invoke(
        f"Extract the claim facts from this investigation summary:\n\n{summary}"
    )

    score, reasons = 0.0, []

    if not facts.policy_active:
        score += 5.0
        reasons.append("policy is not active")
    if facts.claims_last_3y >= 3:
        score += 3.0
        reasons.append(f"{facts.claims_last_3y} claims in 3 years")
    elif facts.claims_last_3y >= 1:
        score += 1.0
        reasons.append(f"{facts.claims_last_3y} prior claim(s)")
    if not facts.in_network_garage:
        score += 2.5
        reasons.append("garage is outside the approved network")
    if facts.estimate_amount > 150000:
        score += 2.0
        reasons.append(f"large estimate: INR {facts.estimate_amount:,.0f}")

    deductible = POLICIES.get(state["policy_id"], {}).get("deductible", 0)
    payout = max(0.0, facts.estimate_amount - deductible)

    return {
        "risk_score": round(min(score, 10.0), 2),
        "risk_reasons": reasons or ["no risk flags"],
        "payout_amount": payout,
        "audit": [f"assess: risk {score:.1f}/10, payout INR {payout:,.0f}"],
    }


# ==============================================================================
# ROUTING
# ==============================================================================

AUTO_PAYOUT_LIMIT = 100000.0    # above this, a human must approve
AUTO_RISK_LIMIT = 3.0           # at or above this, a human must approve


def settlement_router(state: ClaimState) -> str:
    """Two independent triggers. Either one sends the claim to a person."""
    if state["risk_score"] >= AUTO_RISK_LIMIT:
        return "human_review"
    if state["payout_amount"] > AUTO_PAYOUT_LIMIT:
        return "human_review"
    return "auto_settle"


# ==============================================================================
# HUMAN IN THE LOOP
# ==============================================================================

def human_review(state: ClaimState) -> dict:
    """
    Pause and wait for an adjuster.

    interrupt() stops the graph here and hands the dict below back to the caller.
    Nothing runs while it waits. Resuming with Command(resume=X) makes X the
    return value of interrupt(), and this function continues from that line.
    """
    decision = interrupt({
        "task": "Approve this claim settlement",
        "claim_id": state["claim_id"],
        "policy_id": state["policy_id"],
        "holder": POLICIES.get(state["policy_id"], {}).get("holder", "unknown"),
        "claim_text": state["claim_text"],
        "proposed_payout": state["payout_amount"],
        "risk_score": state["risk_score"],
        "risk_reasons": state["risk_reasons"],
        "options": ["approve", "modify", "reject"],
    })

    # Everything below runs only AFTER a human answers.
    action = decision["action"]
    update = {
        "human_decision": action,
        "human_note": decision.get("note", ""),
        "audit": [f"human_review: {action} ({decision.get('note', '')})"],
    }
    if action == "modify":
        update["payout_amount"] = float(decision["amount"])
    return update


def human_router(state: ClaimState) -> str:
    """approve and modify both settle. reject denies."""
    return "deny" if state["human_decision"] == "reject" else "settle"


# ==============================================================================
# TERMINAL NODES
# ==============================================================================

def auto_settle(state: ClaimState) -> dict:
    return {
        "final_status": f"AUTO-SETTLED INR {state['payout_amount']:,.0f}",
        "audit": ["auto_settle: below both thresholds, no human needed"],
    }


def settle(state: ClaimState) -> dict:
    return {
        "final_status": (f"SETTLED INR {state['payout_amount']:,.0f} "
                         f"(adjuster: {state['human_decision']})"),
        "audit": ["settle: paid after human approval"],
    }


def deny(state: ClaimState) -> dict:
    return {
        "final_status": f"DENIED ({state['human_note'] or 'no reason given'})",
        "audit": ["deny: rejected by adjuster"],
    }


# ==============================================================================
# BUILD
# ==============================================================================

def build_graph():
    b = StateGraph(ClaimState)

    b.add_node("investigate", investigate)
    b.add_node("tools", tool_node)
    b.add_node("assess", assess)
    b.add_node("human_review", human_review)
    b.add_node("auto_settle", auto_settle)
    b.add_node("settle", settle)
    b.add_node("deny", deny)

    b.add_edge(START, "investigate")

    # THE AGENT LOOP. investigate -> tools -> investigate, until the model stops
    # asking. The LLM decides how many times round. That is what makes it an
    # agent rather than a workflow.
    b.add_conditional_edges("investigate", should_continue,
                            {"tools": "tools", "assess": "assess"})
    b.add_edge("tools", "investigate")

    b.add_conditional_edges("assess", settlement_router,
                            {"human_review": "human_review", "auto_settle": "auto_settle"})

    b.add_conditional_edges("human_review", human_router,
                            {"settle": "settle", "deny": "deny"})

    b.add_edge("auto_settle", END)
    b.add_edge("settle", END)
    b.add_edge("deny", END)

    # A checkpointer is REQUIRED for interrupt(). It holds the paused state.
    # InMemorySaver is fine for teaching; use SqliteSaver to survive a restart.
    return b.compile(checkpointer=InMemorySaver())


# ==============================================================================
# THE ADJUSTER CONSOLE -- this is what asks the human
# ==============================================================================

def ask_adjuster(payload: dict) -> dict:
    """
    Show the claim to a human and collect their decision from the keyboard.

    Note where this lives: OUTSIDE the graph. The graph does not know or care
    whether the answer comes from a terminal, a web form, a Slack message, or a
    queue three days later. It only cares that a dict comes back.
    """
    print("\n" + "!" * 72)
    print("  ADJUSTER DECISION REQUIRED")
    print("!" * 72)
    print(f"  Claim         : {payload['claim_id']}   Policy: {payload['policy_id']}")
    print(f"  Policy holder : {payload['holder']}")
    print(f"  Incident      : {payload['claim_text']}")
    print(f"\n  Proposed payout : INR {payload['proposed_payout']:,.0f}")
    print(f"  Risk score      : {payload['risk_score']}/10")
    print("  Flags raised    :")
    for r in payload["risk_reasons"]:
        print(f"      - {r}")

    print("\n  Your options:")
    print("      1. approve   pay the proposed amount")
    print("      2. modify    pay a different amount you specify")
    print("      3. reject    deny the claim")

    # Non-interactive fallback, so piping the script does not crash.
    if not sys.stdin.isatty():
        print("\n  [no terminal attached -- defaulting to reject]")
        return {"action": "reject", "note": "no adjuster available"}

    while True:
        choice = input("\n  Choose [1/2/3]: ").strip()

        if choice in ("1", "approve"):
            note = input("  Note (optional): ").strip()
            return {"action": "approve", "note": note or "approved as proposed"}

        if choice in ("2", "modify"):
            raw = input(f"  New payout amount in INR "
                        f"(proposed {payload['proposed_payout']:,.0f}): ").strip()
            try:
                amount = float(raw.replace(",", ""))
            except ValueError:
                print("  Not a number. Try again.")
                continue
            note = input("  Reason for the change: ").strip()
            return {"action": "modify", "amount": amount,
                    "note": note or "amount adjusted by adjuster"}

        if choice in ("3", "reject"):
            note = input("  Reason for rejection: ").strip()
            return {"action": "reject", "note": note or "rejected by adjuster"}

        print("  Please enter 1, 2 or 3.")


# ==============================================================================
# RUN
# ==============================================================================

CLAIMS = [
    {
        "claim_id": "CLM-5001", "policy_id": "POL-1001",
        "claim_text": "Minor rear-end collision at a traffic signal. Bumper and "
                      "boot damaged. Car taken to the authorised garage in Andheri.",
    },
    {
        "claim_id": "CLM-5002", "policy_id": "POL-1002",
        "claim_text": "Major front-end damage after hitting a divider at night. "
                      "Car towed to a local garage. Engine work needed.",
    },
]


def run_claim(graph, claim: dict, thread: str) -> None:
    """Run one claim, asking a human if the graph pauses."""
    print("\n" + "=" * 72)
    print(f"CLAIM {claim['claim_id']}  |  policy {claim['policy_id']}")
    print("=" * 72)

    # thread_id identifies this run to the checkpointer. Resuming means passing
    # the SAME thread_id -- that is how it finds the paused state.
    config = {"configurable": {"thread_id": thread}}

    result = graph.invoke({**claim, "messages": [], "audit": []}, config)

    # A paused graph returns __interrupt__ instead of a finished state.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value

        print(f"\n  >> GRAPH PAUSED at node: {graph.get_state(config).next}")
        print("     Nothing is running. The state is saved. It will wait.")

        answer = ask_adjuster(payload)

        # Command(resume=...) hands this value back to interrupt() inside the
        # node, which then continues from that exact line.
        result = graph.invoke(Command(resume=answer), config)

    print(f"\n  FINAL STATUS : {result['final_status']}")
    print("  AUDIT TRAIL  :")
    for line in result["audit"]:
        print(f"      - {line}")


BRIEF = """
========================================================================
 CLAIM SETTLEMENT DESK
========================================================================
 THE PROBLEM
   Most motor claims are small and obvious -- settling them by hand is
   slow and expensive. Some are large or suspicious -- settling those
   automatically is reckless. So: which claims need a human, and how
   does that human step in without stopping everything else?

 WHAT THIS PROGRAM DOES
   1. An AGENT investigates the claim using tools. The LLM chooses
      which tools to call.
   2. PYTHON scores the claim. Not the LLM -- the payout figure has to
      be auditable.
   3. A ROUTER checks score and amount:
        small + clean  -> settle automatically
        large or risky -> stop and ask a human
   4. The ADJUSTER approves, modifies the amount, or rejects. Each
      answer takes a different path through the graph.

 TWO CLAIMS FOLLOW.
   The first settles on its own. The second will stop and ask you.
========================================================================"""


if __name__ == "__main__":
    print(BRIEF)
    graph = build_graph()

    for i, claim in enumerate(CLAIMS, start=1):
        run_claim(graph, claim, thread=f"claim-{i}")

    print("""
========================================================================
 WHAT JUST HAPPENED
========================================================================
 1. THE LOOP. investigate -> tools -> investigate is a CYCLE. The model
    decided how many times it went round, not you. That cycle is the
    difference between a workflow and an agent.

 2. IT WORKS BECAUSE OF add_messages. Each pass appended the tool
    results to the conversation. With the default reducer the agent
    would forget every result and loop until it hit the limit.

 3. THE LLM GATHERED, PYTHON DECIDED. assess() is arithmetic. No
    language model chose the amount that left the bank.

 4. THE HUMAN WAS A BRANCH. Run it again and answer differently --
    approve, modify, reject take three different paths to three
    different outcomes, with no code change.

 5. PAUSED MEANT PAUSED. While it waited for you, nothing was running
    and no tokens were being spent. The state sat in the checkpointer
    under its thread_id. An adjuster answering three days later would
    be the identical call.

 TRY THIS: run it again and choose 2 (modify) with a very small amount.
 Then run it once more and choose 3. Same claim, same code, three
 different business outcomes.
========================================================================""")