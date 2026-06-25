from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from tools import scrape_url, web_search

load_dotenv()

# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

# ---------------------------------------------------------
# Search Agent
# ---------------------------------------------------------

def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are a professional research search agent.

Your job is to search the web using the available tool.

Rules:
- ALWAYS use the web_search tool.
- Never answer from your own knowledge.
- Return the search results exactly as received.
- Do NOT summarize.
- Preserve all Titles, URLs and Snippets.
"""
    )

# ---------------------------------------------------------
# Reader Agent
# ---------------------------------------------------------

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are a research reading agent.

Rules:
- Read the search results carefully.
- Select the most relevant URL.
- Use the scrape_url tool.
- Return the scraped content only.
- Do not summarize unless the content exceeds the limit.
"""
    )

# ---------------------------------------------------------
# Writer
# ---------------------------------------------------------

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research writer.

Write professional, factual and well-structured reports.

Use markdown formatting.

Always include:

# Introduction

# Key Findings
(at least three detailed sections)

# Conclusion

# Sources
(List every URL found.)
""",
        ),
        (
            "human",
            """
Topic:
{topic}

Research:
{research}
""",
        ),
    ]
)

writer_chain = writer_prompt | llm | StrOutputParser()

# ---------------------------------------------------------
# Critic
# ---------------------------------------------------------

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict research reviewer.

Evaluate the report for:

• Accuracy
• Structure
• Completeness
• Clarity
• Source quality

Respond ONLY in this format:

Score: X/10

Strengths
- ...

Areas to Improve
- ...

Verdict
...
""",
        ),
        (
            "human",
            "{report}",
        ),
    ]
)

critic_chain = critic_prompt | llm | StrOutputParser()