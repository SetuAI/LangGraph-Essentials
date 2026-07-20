"""
================================================================================
 LangGraph Essentials  |  Module 01  |  STATE: TypedDict vs Pydantic
================================================================================

WHY THIS FILE EXISTS
--------------------
Before you can build a single LangGraph node, you must understand ONE idea:

        A graph does not "pass variables around".
        A graph carries a single object called STATE from node to node.

Every node in LangGraph is just a Python function that:
        1. RECEIVES the current state
        2. RETURNS a dictionary of the fields it wants to change aka partial update

So the very first design decision in any LangGraph project is:
        "What shape is my state, and who guarantees that shape is correct?"

There are three common answers, and we will look at all three:
        (a) a plain dict        -> zero guarantees
        (b) a TypedDict         -> guarantees at EDIT time 
        (c) a Pydantic model    -> guarantees at RUN time 

HOW TO RUN
----------
        pip install pydantic
        python 01_state_typeddict_vs_pydantic.py

Read the printed output next to the code. The whole point of this file is that
the difference between TypedDict and Pydantic is INVISIBLE until you run it.
================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------

# `TypedDict` lets us describe the *keys* a dictionary should have and the *type*
# of each value. It lives in `typing` (Python 3.12+). On older Pythons, prefer
# `from typing_extensions import TypedDict` -- LangGraph itself recommends this
# because typing_extensions gets fixes earlier.
from typing import TypedDict

# `Optional[X]` is shorthand for "either an X, or None".
# `List[X]` describes a list whose elements are all of type X.
from typing import List, Optional

# Pydantic is a third-party library (not part of Python). Its job is RUNTIME
# validation: it actually checks your data while the program is running and
# raises an error if the data is wrong.
#   - BaseModel  : the class you inherit from to define a validated model
#   - Field      : lets you attach rules/metadata to a single field
#   - ValidationError : the exception Pydantic raises when data is invalid
from pydantic import BaseModel, Field, ValidationError


# ==============================================================================
# SECTION 0 - A tiny helper so the output is readable
# ==============================================================================

def banner(title: str) -> None:
    """
    Print a section header.

    This has nothing to do with LangGraph. It only makes the terminal output
    easy to follow .

    Args:
        title: the text to display inside the banner.
    """
    # `"=" * 78` repeats the "=" character 78 times -> a horizontal rule.
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ==============================================================================
# SECTION 1 - THE NAIVE APPROACH: a plain dictionary
# ==============================================================================
#
# Before TypedDict and Pydantic, let us see WHY we need them at all.
# A plain dict is the simplest possible "state". It also gives you no help
# whatsoever.
# ------------------------------------------------------------------------------

def demo_plain_dict() -> None:
    """Check plain dict silently accepts anything passed."""

    banner("SECTION 1 | PLAIN DICT  -> no protection at all")

    # A dictionary is just key -> value. Nothing describes what keys SHOULD exist.
    claim_state = {
        "claim_id": "CLM-001",
        "claim_amount": 25000,
    }
    print("Correct-looking state :", claim_state)

    # PROBLEM 1: typo in a key. Python is perfectly happy.
    # We *meant* to update claim_amount, but we typed "claim_amnt".
    # No error. The real field is untouched. This bug will surface 200 lines later.
    claim_state["claim_amnt"] = 99999
    print("After a typo'd key    :", claim_state)
    print("  -> notice 'claim_amount' is STILL 25000. The typo created a new key.")

    # PROBLEM 2: wrong type. A claim amount is now a string. Still no error.
    claim_state["claim_amount"] = "twenty five thousand"
    print("After a wrong type    :", claim_state)
    print("  -> your next line of maths (amount * 1.18) will crash somewhere else.")

    # This is the core lesson: with a plain dict, mistakes are DISCOVERED LATE,
    # far away from where they were CAUSED. TypedDict and Pydantic both exist to
    # move that discovery earlier.


# ==============================================================================
# SECTION 2 - TypedDict
# ==============================================================================
# TypedDict is a Python 3.12+ feature. 
# It is a way to describe the *shape* of a
# dictionary: which keys exist, and what type each value should be. 

# KEY FACT (memorise this):
#       TypedDict does NOTHING at runtime. Zero. It is erased.
# ------------------------------------------------------------------------------

class ClaimStateTD(TypedDict):
    """
    An insurance claim state, described as a TypedDict.

    Reading this class tells a human (and a type-checker) exactly which keys
    exist and what type each one holds. At runtime, an instance of this is
    literally just a `dict`.
    """

    claim_id: str            # e.g. "CLM-001"  -> a string identifier
    claimant_name: str       # e.g. "Asha Rao"
    claim_amount: float      # e.g. 25000.0    -> money, so a number
    is_fraud_suspected: bool # e.g. False      -> True/False flag
    documents: List[str]     # e.g. ["policy.pdf", "invoice.pdf"]


def demo_typeddict() -> None:
    """Show that TypedDict helps your editor but never stops your program."""

    banner("SECTION 2 | TypedDict  -> checked by your IDE, IGNORED at runtime")

    # --- 2a. The happy path -----------------------------------------------
    # We construct it exactly as described. Everything is fine.
    good: ClaimStateTD = {
        "claim_id": "CLM-001",
        "claimant_name": "Asha Rao",
        "claim_amount": 25000.0,
        "is_fraud_suspected": False,
        "documents": ["policy.pdf", "invoice.pdf"],
    }
    print("Valid TypedDict state :", good)

    # PROOF that it is "just a dict": ask Python what type it actually is.
    print("Actual runtime type   :", type(good))
    print("  -> <class 'dict'>. The class ClaimStateTD is not in the picture at all.")

    # --- 2b. Deliberately wrong data --------------------------------------
    # Every single line below is WRONG according to ClaimStateTD:
    #   claim_amount should be a float, not a string
    #   is_fraud_suspected should be a bool, not the string "maybe"
    #   documents should be a list, not None
    #   claimant_name is MISSING entirely
    #   "random_key" is not part of the schema at all
    #
    # Your editor will underline this in red. `mypy` will fail.
    # But Python itself will run it without a whisper of complaint.
    bad: ClaimStateTD = {          # type: ignore[typeddict-item]
        "claim_id": "CLM-002",
        "claim_amount": "twenty five thousand",
        "is_fraud_suspected": "maybe",
        "documents": None,
        "random_key": "I don't belong here",
    }

    print("\nINVALID TypedDict state was accepted without any error:")
    print(bad)
    print("  -> missing key, three wrong types, one unknown key. Program: fine.")

    # And here is where it finally hurts -- far from the actual mistake.
    try:
        gst_inclusive = bad["claim_amount"] * 1.18
        print(gst_inclusive)
    except TypeError as exc:
        print("\nThe crash finally happens later, during maths:")
        print("  TypeError:", exc)
        print("  -> This is the whole problem. The error appears where the data is")
        print("     USED, not where the data was WRONGLY SET.")


# ==============================================================================
# SECTION 3 - Pydantic
# ==============================================================================
#
# A Pydantic model is a real class. When you create an instance, Pydantic runs
# actual checks: types, required fields, custom rules, defaults, coercion.
# Bad data raises `ValidationError` immediately, at the exact line that caused it.
# ------------------------------------------------------------------------------

class ClaimStatePD(BaseModel):
    """
    The same insurance claim state, described as a Pydantic model.

    Compare this class body with ClaimStateTD above -- they look almost
    identical. The difference is entirely in what happens at runtime.
    """

    # A plain required field. If it is missing, Pydantic raises an error.
    claim_id: str

    # `Field(...)` attaches extra rules.
    #   min_length=2  -> the name must be at least 2 characters long
    #   description   -> free-text documentation, useful later for LLM
    #                    structured output, where the LLM literally reads it
    claimant_name: str = Field(
        ...,                       # the literal Ellipsis means "REQUIRED, no default"
        min_length=2,
        description="Full name of the person filing the claim",
    )

    # `gt=0` means "greater than 0". A claim of -5000 is now impossible.
    # `le=1_000_000` means "less than or equal to 1,000,000" (an upper cap).
    claim_amount: float = Field(
        ...,
        gt=0,
        le=1_000_000,
        description="Claim amount in INR",
    )

    # A field WITH a default. If you omit it, you get False instead of an error.
    is_fraud_suspected: bool = False

    # `default_factory=list` gives every new object its OWN empty list.
    # NEVER write `documents: List[str] = []` -- in Python that single list would
    # be shared by every instance, a classic and very painful bug.
    documents: List[str] = Field(default_factory=list)

    # `Optional[str]` = "a string or None". Default None means it is optional.
    adjuster_notes: Optional[str] = None


def demo_pydantic() -> None:
    """Show that Pydantic rejects bad data immediately, at the point of creation."""

    banner("SECTION 3 | Pydantic  -> checked at RUNTIME, fails loudly and early")

    # --- 3a. The happy path -----------------------------------------------
    good = ClaimStatePD(
        claim_id="CLM-001",
        claimant_name="Asha Rao",
        claim_amount=25000.0,
        documents=["policy.pdf", "invoice.pdf"],
        # is_fraud_suspected and adjuster_notes omitted -> defaults are used
    )
    print("Valid Pydantic state  :", good)
    print("Actual runtime type   :", type(good))
    print("  -> a real ClaimStatePD object, NOT a dict.")

    # Access is by ATTRIBUTE (dot), not by key.
    #   TypedDict : state["claim_amount"]
    #   Pydantic  : state.claim_amount
    print("Attribute access      :", good.claim_amount)

    # Convert to a plain dict when you need one (e.g. to log it or send it as JSON).
    print("As a dict             :", good.model_dump())

    # --- 3b. Automatic type COERCION --------------------------------------
    # We pass claim_amount as the STRING "25000". Pydantic sees that the field is
    # declared `float`, checks that the string is a valid number, and converts it.
    # This is a feature, not an accident -- LLMs and web forms return strings.
    coerced = ClaimStatePD(
        claim_id="CLM-003",
        claimant_name="Ravi Menon",
        claim_amount="25000",          # <- a string went IN
    )
    print("\nCoercion demo         :", coerced.claim_amount, type(coerced.claim_amount))
    print("  -> a float came OUT. Pydantic converted '25000' -> 25000.0 for us.")

    # --- 3c. Bad data now EXPLODES immediately ----------------------------
    # These are the SAME mistakes we made in Section 2, where nothing happened.
    print("\nNow the same invalid data as Section 2:")
    try:
        ClaimStatePD(
            claim_id="CLM-002",
            # claimant_name is MISSING          -> error: field required
            claim_amount="twenty five thousand",# -> error: not parseable as a number
            is_fraud_suspected="maybe",         # -> error: not a valid boolean
            documents=None,                     # -> error: not a valid list
        )
    except ValidationError as exc:
        # `exc.errors()` gives a structured list of every problem found.
        # Pydantic reports ALL errors at once, not just the first one.
        print(f"\n  ValidationError -- {len(exc.errors())} problems caught at once:")
        for err in exc.errors():
            # err["loc"] is a tuple pointing at the offending field
            # err["msg"] is the human-readable explanation
            field = ".".join(str(part) for part in err["loc"])
            print(f"    - {field:<20} : {err['msg']}")
        print("\n  -> Every mistake was caught on the line that CAUSED it.")

    # --- 3d. Business rules, not just types -------------------------------
    # A negative claim amount is the correct TYPE (a float) but the wrong VALUE.
    # TypedDict can never catch this. Pydantic can, because of `gt=0`.
    print("\nBusiness-rule check (claim_amount must be > 0):")
    try:
        ClaimStatePD(claim_id="CLM-004", claimant_name="Priya S", claim_amount=-5000)
    except ValidationError as exc:
        print("  ValidationError:", exc.errors()[0]["msg"])
        print("  -> This is the class of bug TypedDict is structurally unable to see.")


# ==============================================================================
# SECTION 4 - SIDE BY SIDE
# ==============================================================================

def demo_side_by_side() -> None:
    """Run the identical bad payload through both approaches, back to back."""

    banner("SECTION 4 | SAME BAD DATA, BOTH APPROACHES")

    # One dictionary of clearly invalid data. We will feed it to both.
    bad_payload = {
        "claim_id": "CLM-999",
        "claim_amount": "not a number",
        "is_fraud_suspected": "maybe",
    }
    print("Payload:", bad_payload, "\n")

    # --- TypedDict ---------------------------------------------------------
    # `ClaimStateTD(**bad_payload)` looks like a constructor call, but a TypedDict
    # "constructor" is really just `dict(...)`. It performs no checking.
    td_result = ClaimStateTD(**bad_payload)  # type: ignore[typeddict-item]
    print("TypedDict :  ACCEPTED  ->", td_result)

    # --- Pydantic ----------------------------------------------------------
    try:
        pd_result = ClaimStatePD(**bad_payload)
        print("Pydantic  :  ACCEPTED  ->", pd_result)
    except ValidationError as exc:
        print(f"Pydantic  :  REJECTED  -> {len(exc.errors())} validation errors")
        for err in exc.errors():
            print(f"               - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")


# ==============================================================================
# SECTION 5 - WHAT THIS MEANS FOR LANGGRAPH
# ==============================================================================

def demo_langgraph_relevance() -> None:
    """
    Connect the lesson back to LangGraph.

    No LangGraph import is needed yet -- we are only simulating what a node does,
    so that the shape of a node function is familiar before Module 02.
    """

    banner("SECTION 5 | HOW STATE IS ACTUALLY USED IN LANGGRAPH")

    # This is the ENTIRE contract of a LangGraph node.
    # It takes the current state and returns a dict of ONLY the fields it changed.
    # LangGraph merges that dict back into the state for you.
    def triage_node(state: ClaimStateTD) -> dict:
        """
        A pretend LangGraph node: flag large claims for fraud review.

        Args:
            state: the current graph state.

        Returns:
            A PARTIAL state update. Note we do not return the whole state --
            only the key we want to change. This is the single most important
            habit to build.
        """
        # Read from state...
        amount = state["claim_amount"]

        # ...decide something...
        suspicious = amount > 100_000

        # ...and return ONLY the delta.
        return {"is_fraud_suspected": suspicious}

    # Simulate what LangGraph does internally.
    state: ClaimStateTD = {
        "claim_id": "CLM-777",
        "claimant_name": "Nikhil Bose",
        "claim_amount": 250000.0,
        "is_fraud_suspected": False,
        "documents": [],
    }
    print("State BEFORE node :", state)

    update = triage_node(state)              # the node runs and returns a delta
    print("Node returned     :", update)

    state.update(update)                     # LangGraph merges the delta for you
    print("State AFTER node  :", state)

    print(
        "\nTakeaway: a node is a plain function. State in, partial-update out.\n"
        "Everything else in LangGraph is about deciding WHICH node runs next."
    )


# ==============================================================================
# SECTION 6 - THE VERDICT
# ==============================================================================

VERDICT = """
--------------------------------------------------------------------------------
 WHICH ONE SHOULD I USE?
--------------------------------------------------------------------------------

                        TypedDict                 Pydantic BaseModel
 ---------------------------------------------------------------------------
 Runtime type check     No                        Yes
 Value/business rules   No                        Yes  (gt, min_length, ...)
 Defaults               No                        Yes
 Type coercion          No                        Yes  ("25" -> 25.0)
 Field descriptions     No                        Yes  (LLMs can read them)
 Runtime object         a plain dict              a real object
 Access style           state["x"]                state.x
 Speed / overhead       zero                      small validation cost
 Extra dependency       no                        yes (pydantic)

 RULE OF THUMB FOR YOUR PROJECTS
 -------------------------------
 * Use TypedDict for the MAIN GRAPH STATE.
     It is the LangGraph default, it is cheap, and node updates are plain dicts.
     Most tutorials and most of the LangGraph docs use it.

 * Use Pydantic when correctness matters more than convenience:
     - the graph's INPUT schema, where untrusted data enters the system
     - LLM STRUCTURED OUTPUT -- this is the big one. When you ask a model to
       fill a schema, Pydantic is what actually verifies the model obeyed.
     - TOOL arguments
     - anything money-, compliance-, or safety-related

 * In real enterprise work you will use BOTH, in the same graph. That is normal.
   Validate hard at the edges; keep the internal state light.

 KEY THING TO REMEMBER
 ----------------------
 If your graph state is a Pydantic model, validation runs on every update, which
 is a genuine safety win but costs a little performance. Also, some LangGraph
 features assume dict-like state. Start with TypedDict; reach for Pydantic
 deliberately, not by default.
--------------------------------------------------------------------------------
"""


# ==============================================================================
# ENTRY POINT
# ==============================================================================

# `if __name__ == "__main__":` means "only run this when the file is executed
# directly, not when it is imported by another file". Standard Python hygiene.
if __name__ == "__main__":
    demo_plain_dict()          # Section 1: why we need any of this
    demo_typeddict()           # Section 2: help at edit time only
    demo_pydantic()            # Section 3: help at run time
    demo_side_by_side()        # Section 4: the difference, made obvious
    demo_langgraph_relevance() # Section 5: what a node actually looks like
    print(VERDICT)             # Section 6: the summary table