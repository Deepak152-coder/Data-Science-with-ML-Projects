from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print


# ---------------------- Tool ----------------------
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text."""
    return len(text)


tools = {
    "get_text_length": get_text_length
}


# ---------------------- LLM ----------------------
llm = ChatMistralAI(model="mistral-small-2506")

# Bind tools
llm_with_tool = llm.bind_tools([get_text_length])


# ---------------------- Chat ----------------------
messages = []

prompt = input("You: ")
messages.append(HumanMessage(content=prompt))

# First LLM call
result = llm_with_tool.invoke(messages)
messages.append(result)

# If the model wants to use a tool
if result.tool_calls:
    tool_call = result.tool_calls[0]

    tool_result = tools[tool_call["name"]].invoke(tool_call["args"])

    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    # Second LLM call after tool execution
    result = llm_with_tool.invoke(messages)

print(f"\nAI: {result.content}")