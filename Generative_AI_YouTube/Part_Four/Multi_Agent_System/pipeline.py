from pprint import pprint

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # --------------------------------------------------
    # STEP 1 : SEARCH AGENT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 1 - SEARCH AGENT")
    print("=" * 60)

    search_agent = build_search_agent()

    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find recent, reliable and detailed information about: {topic}",
                )
            ]
        }
    )

    print("\nRAW SEARCH RESULT\n")
    pprint(search_result)

    state["search_results"] = search_result["messages"][-1].content

    print("\nSEARCH RESULT\n")
    print(state["search_results"])

    # --------------------------------------------------
    # STEP 2 : READER AGENT
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 2 - READER AGENT")
    print("=" * 60)

    reader_agent = build_reader_agent()

    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
Based on these search results about '{topic}',

1. Find the best URL.
2. Use the scrape_url tool.
3. Return ONLY the scraped text.

Search Results:

{state["search_results"]}
""",
                )
            ]
        }
    )

    print("\nRAW READER RESULT\n")
    pprint(reader_result)

    state["scraped_content"] = reader_result["messages"][-1].content

    print("\nSCRAPED CONTENT\n")
    print(state["scraped_content"])

    # --------------------------------------------------
    # STEP 3 : WRITER
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 3 - WRITER")
    print("=" * 60)

    research_combined = f"""
SEARCH RESULTS

{state["search_results"]}

SCRAPED CONTENT

{state["scraped_content"]}
"""

    state["report"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": research_combined,
        }
    )

    print("\nREPORT\n")
    print(state["report"])

    # --------------------------------------------------
    # STEP 4 : CRITIC
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 4 - CRITIC")
    print("=" * 60)

    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"],
        }
    )

    print("\nCRITIC\n")
    print(state["feedback"])

    return state


if __name__ == "__main__":

    topic = input("Enter research topic: ")

    run_research_pipeline(topic)