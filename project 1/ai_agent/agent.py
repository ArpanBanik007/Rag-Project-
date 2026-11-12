"""A small Groq agent with Tavily web search and safe arithmetic tools."""

import ast
import json
import math
import operator
import os

from dotenv import load_dotenv
from groq import Groq
from tavily import TavilyClient

MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 8

# Groq's standard function-tool schema. The model chooses a tool when useful.
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web with Tavily for current facts, sources, and information. Use this when the user asks about recent or externally verifiable information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise web search query describing what to find.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression using numbers and +, -, *, /, //, %, and **. Never use this for code or non-arithmetic expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression, for example 2 * 3444 or (12 + 8) / 4.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]


def web_search(query: str, tavily: TavilyClient) -> str:
    """Search the web with Tavily and return relevant answers and sources."""
    try:
        data = tavily.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )
    except Exception as exc:
        return f"Tavily search failed: {exc}"

    results = []
    if data.get("answer"):
        results.append(f"Tavily answer: {data['answer']}")
    for item in data.get("results", []):
        results.append(
            f"Title: {item.get('title', 'Untitled')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
        )
    return "\n\n".join(results) if results else "Tavily returned no results."

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_math(node):
    """Evaluate only numeric constants and explicitly allowed arithmetic AST nodes."""
    if isinstance(node, ast.Expression):
        return _evaluate_math(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math(node.left)
        right = _evaluate_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent is too large (maximum absolute exponent is 1000).")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math(node.operand))
    raise ValueError("Only numbers and basic arithmetic operators are allowed.")


def calculate(expression: str) -> str:
    """Safely calculate a basic arithmetic expression without executing code."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_math(tree)
        if isinstance(result, float) and not math.isfinite(result):
            return "Error: result is not a finite number."
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"


def run_tool(name: str, arguments: dict, tavily: TavilyClient) -> str:
    """Dispatch a model-requested tool by its registered function name."""
    try:
        if name == "web_search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return "Error: query must be a non-empty string."
            return web_search(query.strip(), tavily)
        if name == "calculate":
            expression = arguments.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                return "Error: expression must be a non-empty string."
            return calculate(expression.strip())
        return f"Error: unknown tool {name!r}."
    except Exception as exc:
        # Return tool errors to the model so it can explain or recover.
        return f"Tool error: {exc}"


def ask_agent(client: Groq, tavily: TavilyClient, user_text: str) -> str:
    """Run the model/tool loop until the model returns a final answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use web_search when the user needs "
                "current or sourced information. Use calculate for arithmetic. "
                "You may answer directly when tools are unnecessary. Explain results clearly."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or "I couldn't produce a response."

        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                result = run_tool(tool_call.function.name, arguments, tavily)
            except (json.JSONDecodeError, ValueError) as exc:
                result = f"Invalid tool arguments: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": result,
                }
            )

    return "I stopped because the tool call limit was reached. Please try a more focused request."


def main() -> None:
    """Start a simple terminal chat session."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not groq_api_key:
        raise SystemExit("GROQ_API_KEY is missing. Add it to project 1/ai_agent/.env.")
    if not tavily_api_key:
        raise SystemExit("TAVILY_API_KEY is missing. Add it to project 1/ai_agent/.env.")

    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    print("Groq assistant ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        try:
            print(f"Assistant: {ask_agent(groq, tavily, user_text)}")
        except Exception as exc:
            print(f"Assistant error: {exc}")


if __name__ == "__main__":
    main()

