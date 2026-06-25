import streamlit as st
from pipeline import run_research_pipeline

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Research Assistant")
st.markdown(
    """
This application uses **multiple AI agents** to perform research.

### Workflow
1. 🔍 Search Agent → Searches the web
2. 📖 Reader Agent → Scrapes the best webpage
3. ✍️ Writer Agent → Generates a research report
4. 📝 Critic Agent → Reviews the report
"""
)

topic = st.text_input(
    "Enter a research topic",
    placeholder="Example: Artificial General Intelligence"
)

if st.button("Start Research", use_container_width=True):

    if topic.strip() == "":
        st.warning("Please enter a research topic.")
        st.stop()

    with st.spinner("Research agents are working..."):

        result = run_research_pipeline(topic)

    st.success("Research Completed!")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 Search Results",
            "📖 Scraped Content",
            "📄 Final Report",
            "📝 Critic Feedback",
        ]
    )

    with tab1:
        st.subheader("Search Results")
        st.write(result["search_results"])

    with tab2:
        st.subheader("Scraped Content")
        st.write(result["scraped_content"])

    with tab3:
        st.subheader("Research Report")
        st.markdown(result["report"])

    with tab4:
        st.subheader("Critic Feedback")
        st.markdown(result["feedback"])