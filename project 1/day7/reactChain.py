
import os
import re
import ast
import operator
from dotenv import load_dotenv
from groq import Groq

# =========================
# 1. ENVIRONMENT SETUP
# =========================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY missing. Check your .env file."
    )

client = Groq(api_key=api_key)

# Supported Groq production model
model="openai/gpt-oss-120b"


# =========================
# 2. TOOL FUNCTIONS
# =========================

def get_product_price(product):
    prices = {
        "iPhone 17": 1000,
        "iPhone 15": 500
    }

    return prices.get(product, 0)


def calculator(expression):
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Only numbers are allowed")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in allowed_operators:
                raise ValueError("Unsupported operator")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if op_type in (ast.Div, ast.FloorDiv, ast.Mod):
                if right == 0:
                    raise ValueError("Division by zero")

            return allowed_operators[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in allowed_operators:
                raise ValueError("Unsupported operator")

            return allowed_operators[op_type](
                evaluate(node.operand)
            )

        raise ValueError("Invalid arithmetic expression")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)

        if isinstance(result, float) and not result.is_integer():
            return round(result, 6)

        return result

    except (ValueError, SyntaxError, TypeError,
            ZeroDivisionError, OverflowError) as error:
        return f"Calculator error: {error}"


# =========================
# 3. TOOL REGISTRY
# =========================

tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}


# =========================
# 4. REACT SYSTEM PROMPT
# =========================

system_prompt = """
You are a shopping assistant using the ReAct pattern.

Available tools:

1. get_product_price(product)
2. calculator(expression)

Follow these rules strictly:

1. Decide the next step.
2. Call only ONE tool at a time.
3. Never invent tool results.
4. After Action, do not write anything else.
5. Wait for the Observation before continuing.
6. Use calculator for arithmetic.
7. Use only the available tools.
8. When finished, give the Final Answer.

Required Action formats:

Thought: I need the product price.
Action: get_product_price("iPhone 17")

Thought: I need to calculate the remaining money.
Action: calculator("5000 - 1000")

Final response format:

Final Answer: The product costs ... and the remaining balance is ...

Do not use keyword arguments.
Do not call multiple tools in one response.
Do not calculate using an assumed tool result.
"""


# =========================
# 5. AGENT LOOP
# =========================

def run_agent(question, max_steps=5):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(max_steps):

        print("\n" + "=" * 40)
        print(f"STEP {step + 1}")
        print("=" * 40)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_completion_tokens=512,
                tool_choice="none"
            )

            answer = response.choices[0].message.content or ""

        except Exception as error:
            print(f"Groq API error: {error}")
            return

        print(answer)

        # Check final answer
        if re.search(r"Final Answer:", answer, re.IGNORECASE):
            print("\nAgent completed successfully.")
            return

        # Extract Action with a quoted string argument
        match = re.search(
            r'Action:\s*(\w+)\(\s*("(?:\\.|[^"\\])*")\s*\)',
            answer
        )

        if not match:
            messages.append({
                "role": "assistant",
                "content": answer
            })

            messages.append({
                "role": "user",
                "content": (
                    'Invalid format. Return exactly one action '
                    'like Action: calculator("5000 - 1000") '
                    'or a Final Answer. Do not call native tools.'
                )
            })
            continue

        tool_name = match.group(1)

        try:
            tool_input = ast.literal_eval(match.group(2))
        except (ValueError, SyntaxError):
            print("Invalid tool argument.")
            return

        # Execute one registered tool
        if tool_name not in tools:
            observation = f"Tool not found: {tool_name}"
        else:
            try:
                observation = tools[tool_name](tool_input)
            except Exception as error:
                observation = f"Tool execution error: {error}"

        print("\nObservation:", observation)

        # Save action and result
        messages.append({
            "role": "assistant",
            "content": answer
        })

        messages.append({
            "role": "user",
            "content": (
                f"Observation: {observation}\n"
                "Use this observation to decide your next step. "
                "Call one tool if needed, or provide the Final Answer."
            )
        })

    print("\nAgent stopped: maximum steps reached.")


# =========================
# 6. RUN AGENT
# =========================

if __name__ == "__main__":

    prompt = """
    I have 5000 rupees.
    What is the price of an iPhone 17?
    How much money will I have left?
    """

    run_agent(prompt)
