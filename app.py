import warnings
warnings.filterwarnings('ignore')
from IPython.display import JSON
import json
import re
import chromadb
import tempfile
import streamlit as st
from chromadb.config import Settings
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from transformers import pipeline
from unstructured.staging.base import dict_to_elements, elements_to_json
from unstructured.chunking.title import chunk_by_title
selectedLLM = "distilbert/distilgpt2"
pipe = pipeline("text-generation", model=selectedLLM)
client  = UnstructuredClient(
    api_key_auth="provide your API key here"
)
chapter_ids = {}
chromaClient = chromadb.Client(settings=chromadb.Settings(allow_reset=True))
def open_saved_pdf(filename):
    #============ Setup Chroma DB ========================
    #Initialize PersistentClient with the specified directory and create a chromaDB collection
    chromaClient.reset()
    collection = chromaClient.create_collection(
      name="skin_lesion_review",
      metadata={"hnsw:space": "cosine"}
    )
    with open(filename, "rb") as f:
      files=shared.Files(
          content=f.read(),
          file_name=filename,
      )
    print(files)
    req = shared.PartitionParameters(
        files=files,
        strategy='hi_res',
        pdf_infer_table_structure=True,
        languages=["eng"],
    )
    try:
      resp = client.general.partition(req)
      #============ Extract Chapters and keep them in a dictionary for metaadata ========================
      element_sanitized = []
      for element in resp.elements:
        #============ Remove header and Footers and images and tables ========================
        if element['type'] != 'Headers' and element['type'] !='Footer' and element['type'] !='Image' :
          # if element['element_id'] not in chapter_ids.keys():
              element_sanitized.append(element)
     #====================Chunking Startegy ================================
      elements = dict_to_elements(element_sanitized)
      chunks = chunk_by_title(
          elements,
          combine_text_under_n_chars=100,
          max_characters=500,
      )
      print(f"Elements for chuncking length : {len(elements)}")
      print(f"Chunks length : {len(chunks)}")
      # print(json.dumps(chunks[0].to_dict(), indent=2))
      for chunk in chunks:
        chunkJsonObj = chunk.to_dict()
        collection.add(
          documents=preprocess_text(chunkJsonObj["text"]),
          ids=chunkJsonObj["element_id"],
          metadatas={"page_no": chunkJsonObj["metadata"].get("page_number"), "filename": chunkJsonObj["metadata"].get("filename")}
        )
    except SDKError as e:
      print(e)


def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s.]", "", text)
    return text

def postprocess_text(text):
    # Trim any leading or trailing whitespace
    text = text.strip()

    # Ensure the first character is capitalized
    if len(text) > 0:
        text = text[0].upper() + text[1:]

    # Ensure the last character is a full stop
    if len(text) == 0 or text[-1] not in ['.', '!', '?']:
        text += '.'

    # Join any excessive whitespace between text segments
    text = re.sub(r'\s+', ' ', text)

    return text
# main
# # Define the GUI here using streamlit
st.header("Ask questions from your literature", anchor=None, help=None, divider=True)
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
if uploaded_files is not None:
  # st.button("Upload Literature", type="primary")
  if st.button("Upload Literature"):
     for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
      with st.spinner('Uploading, Chuncking and Storing...'):
            open_saved_pdf(temp_file_path)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question from uploaded literature?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    if prompt and not uploaded_files:
       st.write('Using already uploaded literature.', icon="🚨")  

    st.session_state.messages.append({"role": "user", "content": prompt})
    response=""
    with st.spinner('Running Inference...'):
    #============plug in LLM resp here=================
     # connect to a chromadb collection
      collectionName = "skin_lesion_review"
      collectionLookup = chromaClient.get_collection(collectionName)
      result = collectionLookup.query(
        query_texts=prompt,
        n_results=4
      )
      combined_string = ", ".join([string for array in result["documents"] for string in array])
    #  print(combined_string)
      output = pipe(combined_string, num_return_sequences=1 , truncation=False ,min_length=100 , max_length = 200, max_new_tokens=150 )
      response = output[0].get('generated_text')
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(postprocess_text(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})




