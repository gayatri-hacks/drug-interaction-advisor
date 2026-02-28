"""
engine.py — ReAct Agent (Sprint 3)
====================================
LLM + LangChain Agent connected to the three API Sandwich tools.
Supports OpenAI (GPT-4o) or Groq (Llama 3.1 70B) - free alternative.

The agent follows a mandatory sequence:
  1. check_local_interactions  → local dict (with RxNorm normalization)
  2. assess_patient_risk       → allergies + Beers + conditions
  3. openfda_lookup            → official FDA label for the new drug
  4. LLM synthesizes final verdict

Usage:
    from engine import run_analysis
    result = run_analysis(age=72, allergies=["Penicillin"],
                          conditions=["CKD"], current_drugs=["Metformin"],
                          new_drug="Ibuprofen 400mg")
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from langchain import agents
from langchain_core.agents import AgentFinish
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.callbacks.base import BaseCallbackHandler

# Support both OpenAI and Groq
from langchain_openai import ChatOpenAI
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

from tools import (
    rxnav_normalize,
    normalize_drug_list,
    openfda_lookup,
    check_local_interactions,
    assess_patient_risk,
)


# ── Trace capture ─────────────────────────────────────────────────────────────
class ReActTraceHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps: list[dict] = []

    def on_agent_action(self, action, **kwargs):
        self.steps.append({"type": "action", "tool": action.tool,
                            "input": str(action.tool_input)})

    def on_tool_end(self, output, **kwargs):
        self.steps.append({"type": "observation", "output": str(output)[:600]})

    def on_agent_finish(self, finish, **kwargs):
        self.steps.append({"type": "final",
                            "output": finish.return_values.get("output", "")})


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm():
    """
    Get LLM client. Supports:
    - Groq (Llama 3.1) - free, fast, no quota issues
    - OpenAI (GPT-4o) - if API key available and has credits
    
    Priority: Groq > OpenAI (Groq is free)
    """
    # First, try Groq (free alternative)
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key and groq_key not in ("your-groq-key-here", ""):
        try:
            return ChatGroq(
                model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
                temperature=0,
                groq_api_key=groq_key,
            )
        except Exception as e:
            print(f"Groq initialization failed: {e}")
    
    # Fall back to OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key or openai_key in ("sk-your-openai-key-here", ""):
        raise ValueError("No valid API key found. Add GROQ_API_KEY or OPENAI_API_KEY to .env")
    
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        openai_api_key=openai_key,
    )


# ── Prompt ────────────────────────────────────────────────────────────────────
AGENT_PROMPT = PromptTemplate.from_template("""
You are a clinical pharmacology AI assistant. Your job is to evaluate whether a proposed
new drug is SAFE to prescribe to the given patient. Be methodical. Cite every tool result.

You have access to these tools:
{tools}

MANDATORY REASONING SEQUENCE — follow in this order:
  Step 1: Call check_local_interactions with ALL drugs (current + proposed).
          This also performs RxNorm normalization, so "Advil" becomes "ibuprofen".
  Step 2: Call assess_patient_risk with ALL drugs to check allergies, Beers Criteria,
          and condition-specific contraindications.
  Step 3: Call openfda_lookup for the PROPOSED NEW DRUG only, to get official FDA data.
  Step 4: Synthesize everything into a final safety verdict using this exact structure:

OVERALL RISK: [SAFE / USE WITH CAUTION / NOT RECOMMENDED / CONTRAINDICATED]
DRUG INTERACTIONS: (from check_local_interactions)
PATIENT-SPECIFIC RISKS: (from assess_patient_risk — allergy, Beers, conditions)
FDA LABEL DATA: (from openfda_lookup — key warnings and interactions)
CLINICAL GUIDANCE: (specific, actionable advice for the prescribing doctor)
TRACEABILITY: (list each finding and which tool detected it)

---
Question: {input}
Thought: {agent_scratchpad}
""")


# ── Tool wrappers (bind patient context) ─────────────────────────────────────
def make_tools(age: int, allergies: list[str], conditions: list[str]) -> list[Tool]:

    def _interactions(drug_list_str: str) -> str:
        try:
            drugs = json.loads(drug_list_str)
            if not isinstance(drugs, list):
                raise ValueError
        except Exception:
            drugs = [d.strip() for d in drug_list_str.split(",")]
        return check_local_interactions(drugs)["summary"]

    def _patient_risk(drug_list_str: str) -> str:
        try:
            drugs = json.loads(drug_list_str)
            if not isinstance(drugs, list):
                raise ValueError
        except Exception:
            drugs = [d.strip() for d in drug_list_str.split(",")]
        return assess_patient_risk(age, allergies, conditions, drugs)["summary"]

    def _fda(drug_name: str) -> str:
        return openfda_lookup(drug_name.strip().strip('"\''))["summary"]

    return [
        Tool(
            name="check_local_interactions",
            func=_interactions,
            description=(
                "Check for known high-risk drug-drug interactions AND normalize drug names "
                "via RxNorm (so 'Advil' → 'ibuprofen', 'Tylenol' → 'acetaminophen'). "
                "Input: JSON array of ALL drug names (current + new), "
                'e.g. ["Metformin 500mg", "Lisinopril 10mg", "Ibuprofen 400mg"]. '
                "Returns severity, mechanism, and recommended action. CALL THIS FIRST."
            ),
        ),
        Tool(
            name="assess_patient_risk",
            func=_patient_risk,
            description=(
                "Check patient-specific risks: allergy cross-reactivity (penicillin, sulfa, "
                "aspirin, codeine, etc.), Beers Criteria flags for elderly patients (age is "
                "pre-bound), and disease-drug contraindications (CKD, heart failure, etc.). "
                "Input: JSON array of ALL drug names. CALL THIS SECOND."
            ),
        ),
        Tool(
            name="openfda_lookup",
            func=_fda,
            description=(
                "Query the official FDA Drug Label database for a single drug. Returns "
                "drug interactions, adverse reactions, and black box warnings from the "
                "FDA-approved label. Also normalizes the name via RxNorm first. "
                "Input: single drug name string, e.g. \"ibuprofen\" or \"Advil\". "
                "CALL THIS THIRD for the proposed new drug."
            ),
        ),
    ]


# ── Fallback: Local synthesis when LLM is unavailable ────────────────────────
def synthesize_locally(
    age: int,
    allergies: list[str],
    conditions: list[str],
    current_drugs: list[str],
    new_drug: str,
    interaction_result: dict,
    patient_risk_result: dict,
    fda_result: dict,
) -> str:
    """
    Synthesize a clinical verdict without LLM using rule-based logic.
    Called when LLM is rate-limited or unavailable.
    """
    # Extract findings from each tool
    interactions = interaction_result.get("interactions", [])
    patient_risks = patient_risk_result
    fda_data = fda_result
    
    # Determine overall risk level
    risk_factors = []
    
    # Check for contraindications from interactions
    for i in interactions:
        if i.get("severity") in ("CONTRAINDICATED", "HIGH"):
            risk_factors.append(("CRITICAL", f"Drug interaction: {i.get('drug_a')} + {i.get('drug_b')} - {i.get('effect', '')}"))
    
    # Check patient-specific risks
    allergy_conflicts = patient_risks.get("allergy_conflicts", [])
    for ac in allergy_conflicts:
        if ac.get("severity") in ("CONTRAINDICATED", "HIGH"):
            risk_factors.append(("CRITICAL", f"Allergy conflict: {ac.get('message', '')}"))
        elif ac.get("severity") == "MODERATE":
            risk_factors.append(("WARNING", f"Allergy concern: {ac.get('message', '')}"))
    
    # Check Beers Criteria
    beers_flags = patient_risks.get("beers_flags", [])
    for b in beers_flags:
        if b.get("risk") == "HIGH":
            risk_factors.append(("WARNING", f"Beers Criteria: {b.get('title', '')} for elderly"))
    
    # Check condition warnings
    condition_warnings = patient_risks.get("condition_warnings", [])
    for w in condition_warnings:
        if "[HIGH]" in w:
            risk_factors.append(("WARNING", w))
        elif "[MODERATE]" in w:
            risk_factors.append(("CAUTION", w))
    
    # Check FDA warnings
    if fda_data.get("boxed_warning"):
        risk_factors.append(("CRITICAL", f"FDA Black Box Warning: {fda_data.get('boxed_warning', [''])[0][:200]}"))
    
    # Determine overall risk level
    has_critical = any(rf[0] == "CRITICAL" for rf in risk_factors)
    has_warning = any(rf[0] == "WARNING" for rf in risk_factors)
    
    if has_critical:
        overall_risk = "CONTRAINDICATED"
    elif has_warning:
        overall_risk = "NOT RECOMMENDED"
    elif risk_factors:
        overall_risk = "USE WITH CAUTION"
    else:
        overall_risk = "SAFE"
    
    # Build the final answer
    lines = []
    lines.append(f"OVERALL RISK: {overall_risk}")
    lines.append("")
    lines.append("DRUG INTERACTIONS:")
    if interactions:
        for i in interactions:
            lines.append(f"  • [{i.get('severity', 'UNKNOWN')}] {i.get('drug_a')} + {i.get('drug_b')}: {i.get('effect', 'N/A')}")
    else:
        lines.append("  No high-risk interactions identified.")
    lines.append("")
    lines.append("PATIENT-SPECIFIC RISKS:")
    if allergy_conflicts:
        for ac in allergy_conflicts:
            lines.append(f"  • [{ac.get('severity', 'UNKNOWN')}] {ac.get('message', '')}")
    if beers_flags:
        for b in beers_flags:
            lines.append(f"  • [BEERS {b.get('risk', '')}] {b.get('title', '')}")
    if condition_warnings:
        for w in condition_warnings:
            lines.append(f"  • {w}")
    if not allergy_conflicts and not beers_flags and not condition_warnings:
        lines.append("  No patient-specific risks identified.")
    lines.append("")
    lines.append("FDA LABEL DATA:")
    if fda_data.get("found"):
        if fda_data.get("boxed_warning"):
            lines.append(f"  ⚠ BLACK BOX: {fda_data.get('boxed_warning', [''])[0][:300]}")
        if fda_data.get("drug_interactions"):
            lines.append(f"  Drug Interactions: {fda_data.get('drug_interactions', [''])[0][:300]}")
        if fda_data.get("warnings"):
            lines.append(f"  Warnings: {fda_data.get('warnings', [''])[0][:300]}")
    else:
        lines.append(f"  {fda_data.get('summary', 'No FDA data available.')}")
    lines.append("")
    lines.append("CLINICAL GUIDANCE:")
    if overall_risk == "CONTRAINDICATED":
        lines.append("  This combination is CONTRAINDICATED. Do not prescribe.")
    elif overall_risk == "NOT RECOMMENDED":
        lines.append("  This combination is NOT RECOMMENDED due to significant risks.")
        lines.append("  Consider alternative therapies or consult with a pharmacist.")
    elif overall_risk == "USE WITH CAUTION":
        lines.append("  Use with CAUTION. Monitor patient closely.")
    else:
        lines.append("  Generally safe. Continue standard monitoring.")
    lines.append("")
    lines.append("TRACEABILITY:")
    lines.append(f"  • Drug interactions: check_local_interactions ({len(interactions)} found)")
    lines.append(f"  • Patient risks: assess_patient_risk (allergies: {len(allergy_conflicts)}, Beers: {len(beers_flags)}, conditions: {len(condition_warnings)})")
    lines.append(f"  • FDA data: openfda_lookup ({'found' if fda_data.get('found') else 'not found'})")
    lines.append("")
    lines.append("[Note: This analysis was generated using local clinical rules due to LLM rate limiting. For complex cases, please consult clinical guidelines or a pharmacist.]")
    
    return "\n".join(lines)


# ── Public entry point ────────────────────────────────────────────────────────
def run_analysis(
    age:           int,
    allergies:     list[str],
    conditions:    list[str],
    current_drugs: list[str],
    new_drug:      str,
    use_fallback:  bool = False,  # Force fallback mode when LLM unavailable
) -> dict:
    """
    Run the full ReAct loop. Called by the Streamlit UI (ui/main.py).

    Args:
        age: Patient age
        allergies: List of patient allergies
        conditions: List of patient conditions
        current_drugs: List of current medications
        new_drug: Proposed new drug to evaluate
        use_fallback: If True, skip LLM and use local synthesis

    Returns:
        final_answer  — full formatted safety report
        react_steps   — list of {thought, action, action_input, observation} dicts
        risk_level    — extracted OVERALL RISK string
        query         — the input sent to the agent
    """
    all_drugs = current_drugs + [new_drug]

    # Pre-normalize new drug for display (fast, cached after first call)
    rxnorm_new = rxnav_normalize(new_drug)

    query = (
        f"Patient: {age}y. "
        f"Conditions: {', '.join(conditions) or 'none'}. "
        f"Allergies: {', '.join(allergies) or 'NKDA'}. "
        f"Current medications: {', '.join(current_drugs) or 'none'}. "
        f"Proposed new drug: {new_drug} "
        f"(RxNorm pre-resolved: '{rxnorm_new['generic_name']}'). "
        f"All drugs to analyze: {json.dumps(all_drugs)}. "
        f"Is it safe to prescribe {new_drug} to this patient?"
    )

    # Create trace handler
    trace = ReActTraceHandler()
    
    # Create tools
    tools = make_tools(age, allergies, conditions)
    
    # Initialize variables for fallback
    llm_unavailable = False
    llm_error_message = None
    
    # First, run all the tools to get the data (these always work)
    intermediate_steps = []
    tool_results = {}
    
    try:
        # Tool 1: Check local interactions
        interaction_input = json.dumps(all_drugs)
        interaction_result = check_local_interactions(all_drugs)
        tool_results["interactions"] = interaction_result
        intermediate_steps.append({
            "thought": "Checking local drug interaction database",
            "action": "check_local_interactions",
            "action_input": interaction_input[:300],
            "observation": interaction_result["summary"][:600],
        })
        
        # Tool 2: Assess patient risk
        patient_risk_result = assess_patient_risk(age, allergies, conditions, all_drugs)
        tool_results["patient_risk"] = patient_risk_result
        intermediate_steps.append({
            "thought": "Assessing patient-specific risks (allergies, Beers, conditions)",
            "action": "assess_patient_risk",
            "action_input": interaction_input[:300],
            "observation": patient_risk_result["summary"][:600],
        })
        
        # Tool 3: FDA lookup for the new drug
        fda_result = openfda_lookup(new_drug)
        tool_results["fda"] = fda_result
        intermediate_steps.append({
            "thought": "Looking up FDA label for the proposed drug",
            "action": "openfda_lookup",
            "action_input": new_drug,
            "observation": fda_result["summary"][:600],
        })
        
    except Exception as e:
        # If tools fail, return error
        return {
            "final_answer": f"Error running clinical tools: {str(e)}",
            "react_steps": [],
            "risk_level": "ERROR",
            "query": query,
            "rxnorm_new": rxnorm_new,
            "trace_steps": [],
        }
    
    # If fallback mode is requested, use local synthesis
    if use_fallback:
        final_answer = synthesize_locally(
            age, allergies, conditions, current_drugs, new_drug,
            tool_results["interactions"],
            tool_results["patient_risk"],
            tool_results["fda"],
        )
        
        # Update trace
        trace.steps.append({"type": "action", "tool": "synthesize_locally", "input": "Fallback mode triggered"})
        trace.steps.append({"type": "observation", "output": "Used local rule-based synthesis due to LLM unavailability"})
        
        # Extract risk level
        risk_level = "UNKNOWN"
        for level in ["CONTRAINDICATED", "NOT RECOMMENDED", "HIGH RISK",
                      "USE WITH CAUTION", "SAFE", "LOW RISK"]:
            if level in final_answer.upper():
                risk_level = level
                break
        
        return {
            "final_answer": final_answer,
            "react_steps": intermediate_steps,
            "risk_level": risk_level,
            "query": query,
            "rxnorm_new": rxnorm_new,
            "trace_steps": trace.steps,
        }
    
    # Try to use the LLM for synthesis
    try:
        # Create the agent using the new LangChain 1.x API
        agent = agents.create_agent(
            get_llm(),
            tools,
            system_prompt=AGENT_PROMPT.template,
        )
        
        # Run the agent with max iterations
        max_iterations = 8
        
        inputs = {"input": query, "agent_scratchpad": ""}
        
        for iteration in range(max_iterations):
            try:
                result = agent.invoke(inputs, config={"callbacks": [trace]})
                
                # Check if agent finished
                if isinstance(result, AgentFinish):
                    final_answer = result.return_values.get("output", "No response generated.")
                    break
                elif isinstance(result, dict):
                    # Agent returned an action
                    action = result.get("action")
                    action_input = result.get("action_input")
                    
                    if action and action_input:
                        # Execute the tool
                        for tool in tools:
                            if tool.name == action:
                                observation = tool.func(str(action_input))
                                intermediate_steps.append((action, action_input, observation))
                                
                                # Update trace
                                trace.steps.append({"type": "action", "tool": action, "input": str(action_input)})
                                trace.steps.append({"type": "observation", "output": str(observation)[:600]})
                                
                                # Prepare next input
                                inputs = {
                                    "input": query,
                                    "agent_scratchpad": f"Thought: I need to call {action}.\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"
                                }
                                break
                else:
                    # Unexpected result type
                    break
                    
            except Exception as e:
                error_str = str(e)
                # Check for rate limit or API errors
                if "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                    llm_unavailable = True
                    llm_error_message = "LLM API rate limit exceeded"
                else:
                    llm_unavailable = True
                    llm_error_message = f"LLM error: {str(e)[:100]}"
                break
        else:
            # Max iterations reached
            final_answer = "Maximum iterations reached. The analysis could not be completed."
            
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
            llm_unavailable = True
            llm_error_message = "LLM API rate limit exceeded"
        else:
            llm_unavailable = True
            llm_error_message = f"LLM initialization error: {str(e)[:100]}"
    
    # If LLM is unavailable, fall back to local synthesis
    if llm_unavailable:
        print(f"⚠ LLM unavailable ({llm_error_message}). Using local synthesis...")
        final_answer = synthesize_locally(
            age, allergies, conditions, current_drugs, new_drug,
            tool_results["interactions"],
            tool_results["patient_risk"],
            tool_results["fda"],
        )
        
        # Update trace
        trace.steps.append({"type": "action", "tool": "synthesize_locally", "input": f"Fallback triggered: {llm_error_message}"})
        trace.steps.append({"type": "observation", "output": "Used local rule-based synthesis"})

    # Build react_steps list for UI
    react_steps = []
    for step in intermediate_steps:
        if isinstance(step, dict):
            react_steps.append(step)
        else:
            action, action_input, observation = step
            react_steps.append({
                "thought":      f"Calling tool: {action}",
                "action":       action,
                "action_input": str(action_input)[:300],
                "observation":  str(observation)[:600],
            })

    # Extract risk level from the final answer text
    risk_level = "UNKNOWN"
    for level in ["CONTRAINDICATED", "NOT RECOMMENDED", "HIGH RISK",
                  "USE WITH CAUTION", "SAFE", "LOW RISK"]:
        if level in final_answer.upper():
            risk_level = level
            break

    return {
        "final_answer": final_answer,
        "react_steps":  react_steps,
        "risk_level":   risk_level,
        "query":        query,
        "rxnorm_new":   rxnorm_new,
        "trace_steps":  trace.steps,
    }


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Drug Interaction Advisor — Engine Demo")
    print("=" * 50)

    result = run_analysis(
        age=80,
        allergies=["Penicillin"],
        conditions=["Type 2 Diabetes", "CKD Stage 3"],
        current_drugs=["Metformin 500mg", "Lisinopril 10mg"],
        new_drug="Advil 400mg",   # ← brand name → RxNorm resolves to ibuprofen
    )

    print(f"\nRisk Level: {result['risk_level']}")
    print(f"\nFinal Answer:\n{result['final_answer']}")
    print(f"\nReAct Steps: {len(result['react_steps'])}")
    print(f"RxNorm for new drug: {result['rxnorm_new']['summary']}")

