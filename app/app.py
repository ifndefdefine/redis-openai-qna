import os
import streamlit as st
import langchain

from dotenv import load_dotenv
load_dotenv()

from urllib.error import URLError
from qna.llm import make_qna_chain, get_cache, openai_prompt


@st.cache_resource
def startup_qna_backend():
    return make_qna_chain()

@st.cache_resource
def fetch_llm_cache():
    return get_cache()


try:
    qna_chain = startup_qna_backend()
    langchain.llm_cache = fetch_llm_cache()

    default_question = ""
    default_answer = ""

    if 'question' not in st.session_state:
        st.session_state['question'] = default_question
    if 'response' not in st.session_state:
        st.session_state['response'] = {
            "choices" :[{
                "text" : default_answer
            }]
        }

    st.image(os.path.join('assets','RedisOpenAI.png'))

    col1, col2 = st.columns([4,2])
    with col1:
        st.write("# Q&A Application")
    # with col2:
    #     with st.expander("Settings"):
    #         st.tokens_response = st.slider("Tokens response length", 100, 500, 400)
    #         st.temperature = st.slider("Temperature", 0.0, 1.0, 0.1)


    question = st.text_input("*Ask management-related questions to IntuitionAI's chatbot", default_question)

    if question != '':
        if question != st.session_state['question']:
            st.session_state['question'] = question
            with st.spinner("OpenAI and Redis are working to answer your question..."):
                result = qna_chain({"query": question})
                st.session_state['context'], st.session_state['response'] = result['source_documents'], result['result']
                result2 = openai_prompt(question)
                st.session_state['response2'] = result2
            st.write("### Informed Response")
            st.write(f"{st.session_state['response']}")
            with st.expander("Show Q&A Context Documents"):
                if st.session_state['context']:
                    docs = "\n".join([doc.page_content for doc in st.session_state['context']])
                    st.text(docs)
            st.write("### Default Response")
            st.write(f"{st.session_state['response2']}")

    st.markdown("____")
    st.markdown("")
    st.write("## How does it work?")
    st.write("""
        The Q&A app exposes a dataset of management training articles. Ask questions like
        *"How should I structure a performance management conversation?"* or *"What are the key characteristics of a manager?"*, and get answers!

        Everything is powered by OpenAI's embedding and generation APIs and [Redis](https://redis.com/redis-enterprise-cloud/overview/) as a vector database.

        There are 3 main steps:

        1. OpenAI's embedding service converts the input question into a query vector (embedding).
        2. Redis' vector search identifies relevant wiki articles in order to create a prompt.
        3. OpenAI's generative model answers the question given the prompt+context.

        See the reference architecture diagram below for more context.
    """)

    st.image(os.path.join('assets', 'RedisOpenAI-QnA-Architecture.drawio.png'))



except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )
