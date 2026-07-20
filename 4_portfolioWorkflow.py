"""
LangGraph Essentials | Module 03b | PORTFOLIO RISK REVIEW + STREAM MODES

Applies Module 03.

Scenario: 
A portfolio arrives. Three independent risk checks run on it — concentration, liquidity, credit. 
Their results get combined into one score. If the score is above the desk's tolerance, it goes to the investment committee. 
If not, it's approved automatically. Everything that happened is logged.
That's it. That's a real workflow you'd find on a risk desk, in about 200 lines.


    START
      |
      v
 [load_portfolio]
      |
  +---+---+--------------+          THREE CHECKS IN PARALLEL
  v       v              v
[concentration] [liquidity] [credit]
  |       |              |
  +---+---+--------------+
      v
  [aggregate]                       fan-in: combine into one score
      |
   +--+--+
 high   ok                          conditional edge
   |     |
   v     v
[escalate] [auto_approve]
   |     |
   +--+--+
      v
     END
"""

from typing import Annotated, TypedDict, Optional
from operator import add

from langgraph.graph import StateGraph, START, END


# ==============================================================================
# STATE
# ==============================================================================

class PortfolioState(TypedDict):
    # supplied by the caller
    portfolio_id: str
    escalation_threshold: float

    # written once -> replace is correct
    holdings: Optional[list[dict]]
    risk_score: Optional[float]
    decision: Optional[str]

    # written by THREE PARALLEL nodes -> reducer is mandatory , not optional
    findings: Annotated[list[dict], add]

    # written by every node -> reducer needed for the same reason
    audit_log: Annotated[list[str], add]


HOLDINGS = [
    {"name": "Reliance Industries", "weight_pct": 34.0, "liquidity": "high", "rating": "AAA"},
    {"name": "HDFC Bank",           "weight_pct": 22.0, "liquidity": "high", "rating": "AAA"},
    {"name": "Unlisted RE Fund II", "weight_pct": 18.0, "liquidity": "low",  "rating": "BBB"},
    {"name": "Infosys",             "weight_pct": 14.0, "liquidity": "high", "rating": "AA"},
    {"name": "SME Credit Pool",     "weight_pct": 12.0, "liquidity": "low",  "rating": "BB"},
]


# ==============================================================================
# NODES
# ==============================================================================

def load_portfolio(state: PortfolioState) -> dict:
    """Fetch holdings. Writes: holdings, audit_log."""
    return {
        "holdings": HOLDINGS,
        # A single line still goes in a LIST -- it must match the field's type.
        "audit_log": [f"Loaded {len(HOLDINGS)} holdings for {state['portfolio_id']}"],
    }


# The three checks below run at the SAME TIME.
# Each reads only `holdings` and writes only findings/audit_log.
# That independence is what makes running them in parallel safe.

def concentration_check(state: PortfolioState) -> dict:
    """Flag any single holding above 30%."""
    heavy = [h for h in state["holdings"] if h["weight_pct"] > 30.0]
    return {
        "findings": [{
            "check": "concentration",
            "severity": 8.0 if heavy else 2.0,
            "detail": f"{len(heavy)} holding(s) over 30%: {[h['name'] for h in heavy]}"
                      if heavy else "No holding exceeds 30%",
        }],
        "audit_log": ["concentration_check complete"],
    }


def liquidity_check(state: PortfolioState) -> dict:
    """Flag more than 25% in low-liquidity assets."""
    illiquid = sum(h["weight_pct"] for h in state["holdings"] if h["liquidity"] == "low")
    return {
        "findings": [{
            "check": "liquidity",
            "severity": 7.0 if illiquid > 25.0 else 3.0,
            "detail": f"{illiquid:.1f}% is low-liquidity",
        }],
        "audit_log": ["liquidity_check complete"],
    }


def credit_check(state: PortfolioState) -> dict:
    """Flag sub-investment-grade holdings."""
    weak = [h for h in state["holdings"] if h["rating"] in ("BB", "B", "CCC")]
    return {
        "findings": [{
            "check": "credit",
            "severity": 6.0 if weak else 1.0,
            "detail": f"Sub-investment-grade: {[h['name'] for h in weak]}"
                      if weak else "All investment grade",
        }],
        "audit_log": ["credit_check complete"],
    }


def aggregate(state: PortfolioState) -> dict:
    """
    Fan-in. Average the three severities into one risk score.

    This only works because the reducer kept all three findings. Without it,
    state["findings"] would hold whichever check finished last.
    """
    severities = [f["severity"] for f in state["findings"]]
    score = sum(severities) / len(severities)
    return {
        "risk_score": round(score, 2),
        "audit_log": [f"Aggregated {len(severities)} findings -> {score:.2f}"],
    }


def escalate(state: PortfolioState) -> dict:
    return {
        "decision": "ESCALATED to investment committee",
        "audit_log": ["Escalated for committee review"],
    }


def auto_approve(state: PortfolioState) -> dict:
    return {
        "decision": "AUTO-APPROVED",
        "audit_log": ["Auto-approved, no committee review"],
    }


def risk_router(state: PortfolioState) -> str:
    """Router: returns a STRING (where next). Never changes state."""
    return "escalate" if state["risk_score"] >= state["escalation_threshold"] else "approve"


# ==============================================================================
# GRAPH
# ==============================================================================

def build_graph():
    builder = StateGraph(PortfolioState)

    for name, fn in [
        ("load_portfolio", load_portfolio),
        ("concentration_check", concentration_check),
        ("liquidity_check", liquidity_check),
        ("credit_check", credit_check),
        ("aggregate", aggregate),
        ("escalate", escalate),
        ("auto_approve", auto_approve),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "load_portfolio")

    # FAN-OUT: three edges out of one node -> all three run in the same step.
    builder.add_edge("load_portfolio", "concentration_check")
    builder.add_edge("load_portfolio", "liquidity_check")
    builder.add_edge("load_portfolio", "credit_check")

    # FAN-IN: three edges into one node -> LangGraph waits for ALL THREE
    # before running aggregate. You write no waiting logic; the edges say it.
    builder.add_edge("concentration_check", "aggregate")
    builder.add_edge("liquidity_check", "aggregate")
    builder.add_edge("credit_check", "aggregate")

    builder.add_conditional_edges(
        "aggregate",
        risk_router,
        {"escalate": "escalate", "approve": "auto_approve"},
    )

    builder.add_edge("escalate", END)
    builder.add_edge("auto_approve", END)

    return builder.compile()


# ==============================================================================
# RUN
# ==============================================================================

def run_reviews():
    print("\n" + "=" * 70)
    print("PORTFOLIO RISK REVIEW")
    print("=" * 70)

    graph = build_graph()

    # Same portfolio, same score, two different escalation thresholds.
    # The score does not change -- the BAR does. That is what makes the
    # conditional edge visible on demand.
    for threshold, label in [(5.0, "STRICT desk"), (8.0, "TOLERANT desk")]:
        print(f"\n--- {label}: escalate if risk >= {threshold} ---")

        final = graph.invoke({
            "portfolio_id": "PF-2291",
            "escalation_threshold": threshold,
        })

        print(f"  Findings : {len(final['findings'])} (all three survived)")
        for f in final["findings"]:
            print(f"    {f['check']:<15} {f['severity']:<4} | {f['detail']}")
        print(f"  Risk     : {final['risk_score']}")
        print(f"  Decision : {final['decision']}")
        print(f"  Audit    : {len(final['audit_log'])} entries")


# ==============================================================================
# STREAM MODES
# ==============================================================================

def demo_streaming():
    """
    invoke() runs everything and returns the final state -- you see nothing
    until it is done. stream() yields after each step instead.
    stream_mode controls WHAT it yields.
    """
    print("\n" + "=" * 70)
    print("STREAM MODES")
    print("=" * 70)

    graph = build_graph()
    inputs = {"portfolio_id": "PF-2291", "escalation_threshold": 5.0}

    print("\n[A] invoke()  -- final state only")
    final = graph.invoke(inputs)
    print(f"    one dict, {len(final)} keys, nothing shown until done")

    # 'updates' yields {node_name: the partial update it returned}.
    # Best mode for teaching: it shows the node contract happening live.
    print("\n[B] stream_mode='updates'  -- each node's delta")
    for chunk in graph.stream(inputs, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"    {node:<22} wrote {list(update.keys())}")

    # 'values' yields the whole state after each step.
    print("\n[C] stream_mode='values'  -- full state per step")
    for step, snap in enumerate(graph.stream(inputs, stream_mode="values")):
        filled = [k for k, v in snap.items() if v not in (None, [], "")]
        print(f"    step {step}: {len(filled)} fields -> {filled}")

    # Pass a LIST for several modes; chunks arrive as (mode, payload).
    print("\n[D] stream_mode=['updates','values']  -- both, tagged")
    for mode, payload in graph.stream(inputs, stream_mode=["updates", "values"]):
        if mode == "updates":
            print(f"    ({mode}) {list(payload.keys())}")

    print("""
   Reducers stop being academic. In 03 file, add merely saved a log line — annoying to lose, but survivable. Here, drop the reducer and aggregate averages one finding instead of three. Your risk score is wrong, no error appears anywhere, and the portfolio gets approved on a third of the evidence. That's the version of the lesson that sticks.
Parallelism is free and invisible. No async, no threads, no locks, no join. Three edges out, three edges in. That surprises people, and it's one of the strongest arguments for using LangGraph at all.
Fan-in requires no waiting code. aggregate doesn't check whether the others finished. The edges declare the dependency and LangGraph blocks until all three land. Freshers expect to write that logic themselves.
Nodes stay dumb, routing stays separate. Each check knows one rule and nothing about the graph. risk_router knows nothing about liquidity. You can add a fourth check by writing one function and two edges — no existing node changes. That's the architectural payoff, and it's why anyone chooses a graph over a script.
The audit log is not decoration. In insurance and asset management, "why was this approved?" is a regulatory question. The reducer is what makes an answer exist. Worth saying out loud when you teach it — it reframes reducers from a syntax quirk into a compliance feature.""")


if __name__ == "__main__":
    run_reviews()
    demo_streaming()