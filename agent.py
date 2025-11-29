# agent.py
import json
from typing import Any, Dict
from llm import call_ollama_chat
from tools.translate import translate_report_input
from tools.validate_procedure import validate_procedure_tool
from tools.price_compare import get_quotes
from tools.audit_bill import audit_bill
from tools.cost_negotiator import negotiation_script
from tools.discharge import discharge_orchestrator

AGENT_SYSTEM = """
You are the Arogyamitra agent. Available tools: translate_report, validate_procedure, price_compare, cost_negotiator, audit_bill, discharge_orchestrator.
When you want to call a tool, output a single JSON object on its own line:
{"action":"call_tool","tool":"<tool_name>","args":{...}}
After the tool result is returned to you (as plain text), continue reasoning. When done, output:
{"action":"done","result":"<final textual answer>"}
"""

TOOLS = {
    "translate_report": lambda args: translate_report_input(args.get("report_text","")) if isinstance(args, dict) else translate_report_input(str(args)),
    "validate_procedure": lambda args: validate_procedure_tool(args.get("text","")) if isinstance(args, dict) else validate_procedure_tool(str(args)),
    "price_compare": lambda args: get_quotes(args.get("procedure",""), args.get("city","Bangalore")) if isinstance(args, dict) else get_quotes(str(args)),
    "audit_bill": lambda args: audit_bill(args.get("bill_text",""), use_llm=False) if isinstance(args, dict) else audit_bill(str(args), use_llm=False),
    "cost_negotiator": lambda args: negotiation_script(args.get("procedure",""), args.get("quotes",[])) if isinstance(args, dict) else negotiation_script(str(args), []),
    "discharge_orchestrator": lambda args: discharge_orchestrator(args.get("bill_text",""), args.get("hospital","")) if isinstance(args, dict) else discharge_orchestrator(str(args), "")
}

def run_agent_prompt(user_input: str, max_steps: int = 6) -> str:
    system = AGENT_SYSTEM
    prompt = f"User: {user_input}\n\nFollow protocol exactly."
    response = call_ollama_chat(system, prompt)
    steps = 0
    while steps < max_steps:
        steps += 1
        # find JSON in response
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            # ask again for JSON action
            response = call_ollama_chat(system, f"Previous output: {response}\n\nYou must reply with the action JSON only.")
            continue
        try:
            jtext = response[start:end+1]
            payload = json.loads(jtext)
        except Exception:
            response = call_ollama_chat(system, f"Could not parse JSON from: {response}\n\nPlease output the action JSON only.")
            continue
        action = payload.get("action")
        if action == "call_tool":
            tool = payload.get("tool")
            args = payload.get("args", {})
            tool_fn = TOOLS.get(tool)
            if not tool_fn:
                response = call_ollama_chat(system, json.dumps({"error": f"Unknown tool: {tool}"}))
                continue
            try:
                result = tool_fn(args)
            except Exception as e:
                result = {"error": str(e)}
            # send result back to model and ask it to continue
            response = call_ollama_chat(system, f"Tool result:\n{json.dumps(result, default=str)}\n\nContinue and follow protocol.")
            continue
        elif action == "done":
            return payload.get("result", "(no result provided)")
        else:
            response = call_ollama_chat(system, f"Unrecognized action: {payload}. Follow the protocol exactly.")
            continue
    return "[Agent error] max steps reached or failed to finish."
