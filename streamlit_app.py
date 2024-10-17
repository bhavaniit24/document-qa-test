import streamlit as st
from openai import OpenAI
import requests


def find_preprint_1(title):
    url = f"https://api.unpaywall.org/v2/search?query={urllib.parse.quote_plus(title)}&email=unpaywall_01@example.com"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            results = []
            for result_response in data["results"]:
                new_doi_url = result_response["response"].get("doi_url", "DOI URL not available")
                publisher = result_response["response"].get("publisher", "Publisher not available")
                published_date = result_response["response"].get("published_date", "Published date not available")
                year = result_response["response"].get("year", "Year not available")

                results.append({
                    "New DOI URL": new_doi_url,
                    "Publisher": publisher,
                    "Published Date": published_date,
                    "Year": year,
                })
            return results
        else:
            print("unpaywall: No publication details found.")
    else:
        print(f"Error fetching data from Unpaywall: {response.status_code}")

    return None

def find_preprint_2(arxiv_id=None, title=None):
    if not arxiv_id and not title:
        return "Please provide either an arXiv ID or a paper title."

    arxiv_url = "http://export.arxiv.org/api/query"

    if arxiv_id:
        query = f"id_list={arxiv_id}"
    else:
        query = f"search_query=all:{title}"

    arxiv_response = requests.get(f"{arxiv_url}?{query}")

    if arxiv_response.status_code != 200:
        print("Error fetching data from arXiv")

    if "Journal-ref" in arxiv_response.text:
        start = arxiv_response.text.find("<arxiv:journal_ref>")
        end = arxiv_response.text.find("</arxiv:journal_ref>")
        journal_ref = arxiv_response.text[start + len("<arxiv:journal_ref>") : end]
        print(f"The paper has been published in: {journal_ref}")
    else:
        if title is None:
            title = arxiv_response.text[
                arxiv_response.text.find("<title>")
                + len("<title>") : arxiv_response.text.find("</title>")
            ]

        crossref_url = f"https://api.crossref.org/works?query.title={title}&rows=10"  # Increase rows to get more results
        crossref_response = requests.get(crossref_url)

        if crossref_response.status_code != 200:
            print("Error fetching data from CrossRef")

        data = crossref_response.json()
        if data["message"]["items"]:
            results = []
            for item in data["message"]["items"]:
                new_doi_url = item.get("URL", "DOI URL not found")
                publisher = item.get("publisher", "Publisher not available")
                published_date = item.get("published-print", {}).get("date-parts", [["Unknown"]])[0]
                published_date_str = (
                    "-".join([str(part) for part in published_date])
                    if published_date[0] != "Unknown"
                    else "Published date not available"
                )
                year = (
                    item.get("created", {}).get("date-parts", [[None]])[0][0]
                    if item.get("created", {}).get("date-parts")
                    else "Year not available"
                )

                results.append({
                    "New DOI URL": new_doi_url,
                    "Publisher": publisher,
                    "Published Date": published_date_str,
                    "Year": year,
                })
            return results
        else:
            print("No publication details found.")

    return None

def find_preprint_3(title):
    url = f"https://dblp.org/search/publ/api?q={urllib.parse.quote_plus(title)}&format=json"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "result" in data and "hits" in data["result"] and int(data["result"]["hits"]["@total"]) > 0:
            results = []
            for hit in data["result"]["hits"]["hit"]:
                info = hit["info"]
                title = info.get("title", "Title not available")

                # Fix: Extract the "author" names properly from dictionaries
                authors = ", ".join([author["text"] for author in info.get("authors", {}).get("author", [])])

                year = info.get("year", "Year not available")
                venue = info.get("venue", "Venue not available")
                url = info.get("url", "URL not available")

                results.append({
                    "Title": title,
                    "Authors": authors,
                    "Year": year,
                    "Venue": venue,
                    "URL": url,
                })
            return results
        else:
            print("dblp: No publication details found.")
    else:
        print(f"Error fetching data from DBLP: {response.status_code}")

    return None


st.title("Find Published Papers")

st.write("Provide the paper title and find publications")

title = st.text_input("Paper")

if title:
    st.write(find_preprint_1(title=title))
    st.write(find_preprint_2(title=title))
    st.write(find_preprint_2(arxiv_id=title))
    st.write(find_preprint_3(title=title))


# # Show title and description.
# st.title("📄 Bhavan Test App")
# st.write(
#     "Upload a document below and ask a question about it – GPT will answer! "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
# )

# # Ask user for their OpenAI API key via `st.text_input`.
# # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:

#     # Create an OpenAI client.
#     client = OpenAI(api_key=openai_api_key)

#     # Let the user upload a file via `st.file_uploader`.
#     uploaded_file = st.file_uploader(
#         "Upload a document (.txt or .md)", type=("txt", "md")
#     )

#     # Ask the user for a question via `st.text_area`.
#     question = st.text_area(
#         "Now ask a question about the document!",
#         placeholder="Can you give me a short summary?",
#         disabled=not uploaded_file,
#     )

#     if uploaded_file and question:

#         # Process the uploaded file and question.
#         document = uploaded_file.read().decode()
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"Here's a document: {document} \n\n---\n\n {question}",
#             }
#         ]

#         # Generate an answer using the OpenAI API.
#         stream = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             stream=True,
#         )

#         # Stream the response to the app using `st.write_stream`.
#         st.write_stream(stream)
