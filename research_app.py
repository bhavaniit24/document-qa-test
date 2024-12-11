import os
from io import BytesIO
from typing import List

import fitz
import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.document_loaders import (
    ArxivLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
)
from langchain_community.retrievers import ArxivRetriever
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdown2 import markdown
from openai import OpenAI
from scidownl import scihub_download
from weasyprint import HTML


class ResearchPaperRetriever(BaseRetriever):
    storage_folder_path: str

    def _load_research_paper(self, file_name: str) -> List[Document]:
        file_path = f"{self.storage_folder_path}/{file_name}"
        loader = PyMuPDFLoader(file_path)
        return loader.load()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._load_research_paper(query)


class ResearchApp:

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.search_api = GoogleSearchAPIWrapper()
        self.model_name = model_name
        self.client = OpenAI()
        self.llm = ChatOpenAI(
            model=model_name,
        )

    def gpt_search(self, query: str) -> str:
        ai_msg = self.llm.invoke(query)
        return ai_msg.content

    def gpt_search_with_structure(self, query, structure):
        structured_llm = self.llm.with_structured_output(structure, method="json_mode")
        ai_msg = structured_llm.invoke(query)
        return ai_msg

    def google_search(self, query: str) -> str:
        tool = Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=self.search_api.run,
        )
        return tool.run(query)

    def ddg_search(self, query: str) -> str:
        tools = load_tools(["ddg-search"])
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )
        response = agent_executor.invoke({"input": query})
        return response["output"]

    def download_scihub_pdf(
        self, doi: str, base_directory: str = "/content/scihub/"
    ) -> str:
        paper_url = f"https://doi.org/{doi}"
        out_filename = f"paper_{doi.replace('/', '-')}.pdf"
        out_path = base_directory + out_filename

        try:
            scihub_download(
                paper_url,
                paper_type="doi",
                out=out_path,
                proxies={"http": "socks5://127.0.0.1:7890"},
            )
            return f"Downloaded: {doi} to {out_path}"
        except Exception as e:
            return f"Failed to download {doi}: {e}"

    def extract_text_from_pdf(self, pdf_file: BytesIO) -> str:
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text

    def extract_text_from_html(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return " ".join([p.get_text() for p in soup.find_all("p")])

    def extract_text_from_url(self, url: str) -> str:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            pdf_file = BytesIO(response.content)
            return self.extract_text_from_pdf(pdf_file)
        else:
            return self.extract_text_from_html(url)

    def semantic_search_agent(self, query: str) -> str:
        instructions = """You are an expert researcher."""
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        prompt = base_prompt.partial(instructions=instructions)
        tools = [SemanticScholarQueryRun()]
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )
        result = agent_executor.invoke({"input": query})
        return str(result["output"])

    def arxiv_search_agent(self, query: str) -> str:
        tools = load_tools(["arxiv"])
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )
        result = agent_executor.invoke({"input": query})
        return str(result["output"])

    def arxiv_load_documents_from_query(
        self, query: str, load_max_docs: int = 100, load_all_available_meta: bool = True
    ) -> str:
        loader = ArxivLoader(
            query=query,
            load_max_docs=load_max_docs,
            load_all_available_meta=load_all_available_meta,
        )

        docs = loader.load()
        text = ""
        for doc in docs:
            text += f"{doc.metadata['Title']} | {doc.metadata['entry_id']}" + "\n\n"
        return text

    def arxiv_load_document_from_id(
        self, paper_id: str, load_max_docs: int = 100
    ) -> str:
        retriever = ArxivRetriever(load_max_docs=load_max_docs)
        docs = retriever.invoke(paper_id)
        text = ""
        for doc in docs:
            text += f"{doc.page_content}" + "\n\n"
        return text

    def get_all_papers(self, pdf_file_path):
        files = os.listdir(pdf_file_path)
        return files

    def get_full_content(self, selected_file, drive_path):
        retriever = ResearchPaperRetriever(storage_folder_path=drive_path)
        documents = retriever._load_research_paper(selected_file)
        page_contents = [doc.page_content for doc in documents]
        full_content = " ".join(page_contents)
        return full_content

    def openai_chat_completion(self, messages, type="text"):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=16383,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": type},
        )

        return response.choices[0].message.content

    def extract_research_paper_info(self, prompt):
        return self.openai_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            type="json_object",
        )

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        return len(encoding.encode(string))

    def get_optimized_final_prompt(
        self,
        base_prompt: str,
        drive_path: str,
        pdf_filename: str,
        max_token_limit: int = 100000,
    ) -> str:
        full_text = self.get_full_content(pdf_filename, drive_path)
        paper_token_count = self.num_tokens_from_string(full_text, self.model_name)

        if paper_token_count > max_token_limit:
            encoding = tiktoken.encoding_for_model(self.model_name)
            encoded_tokens = encoding.encode(full_text)[:max_token_limit]
            full_text = encoding.decode(encoded_tokens)

        new_token_count = self.num_tokens_from_string(full_text, self.model_name)
        final_prompt = base_prompt.replace("{full_paper}", full_text)
        final_count = self.num_tokens_from_string(final_prompt, self.model_name)

        print(
            f"Original Tokens: {paper_token_count}, New Tokens: {new_token_count}, Final Tokens: {final_count}"
        )

        return final_prompt

    def get_retriever(
        self,
        doc_path,
        embedding_model="text-embedding-3-large",
        chunk_size=10000,
        chunk_overlap=200,
        separators=["\n---\n"],
        collection_name="rag_example",
        persist_directory="./chromadb",
    ):
        file_extension = os.path.splitext(doc_path)[1]

        if file_extension == ".pdf":
            loader = PyMuPDFLoader(doc_path)
        if file_extension == ".eml":
            loader = UnstructuredEmailLoader(doc_path)
        else:
            loader = TextLoader(doc_path)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

        vector_store = Chroma.from_documents(
            documents=text_splitter.split_documents(documents),
            embedding=OpenAIEmbeddings(model=embedding_model),
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        return vector_store.as_retriever()

    def create_rag_system(self, retriever, prompt, query):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | hub.pull(prompt)
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(query)

    def generate_pdf_report(
        self, title: str, markdown_content: str, output_pdf_path: str
    ):
        html_content = f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>{title}</title>
        </head>
        <body>
        {markdown(markdown_content)}
        </body>
        </html>"""

        HTML(string=html_content).write_pdf(output_pdf_path)
        print(f"PDF generated successfully at: {output_pdf_path}")

    def get_llm(self, model_name, provider):
        return init_chat_model(model_name, model_provider=provider, temperature=0)
