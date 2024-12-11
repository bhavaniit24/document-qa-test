import streamlit as st

from podcast_app import PodcastApp
from research_app import ResearchApp

# Show title and description.
st.title("📄 Bhavan Research App")

openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    research_app = ResearchApp(api_key=openai_api_key)
    podcast_app = PodcastApp(api_key=openai_api_key)

    uploaded_file = st.file_uploader("Upload a document", type=("txt", "md", "pdf"))

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        document = research_app.get_current_file_full_content(uploaded_file)

        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        stream = research_app.openai_chat_completion(messages=messages)

        st.write(stream)
