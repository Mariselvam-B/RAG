import os
import warnings
import psycopg2
import pandas as pd
import json
import re

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# Updated imports per LangChain deprecation warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Suppress TensorFlow deprecated warnings
os.environ["USE_TF"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR only
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def connect_to_db():
    conn = psycopg2.connect(
        dbname="your_db",
        user="your_user",
        password="your_pass",
        host="localhost",
        port="5432"
    )
    return conn

def get_all_tables(conn):
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'quain' AND table_type = 'BASE TABLE'
    """
    tables = pd.read_sql(query, conn)
    return tables['table_name'].tolist()

def filter_table_by_keyword(conn, table_name, keyword):
    df = pd.read_sql(f'SELECT * FROM "quain"."{table_name}"', conn)
    # Filter rows where any column contains keyword (case insensitive)
    filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(keyword, case=False).any(), axis=1)]
    return filtered_df

def rows_to_documents(df):
    docs = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {val}" for col, val in row.items()])
        docs.append(Document(page_content=content))
    return docs

def main():
    keyword = "Airhouse"
    conn = connect_to_db()

    print("📄 Fetching tables...")
    table_names = get_all_tables(conn)

    all_docs = []
    print(f"🔍 Filtering rows containing keyword '{keyword}'...")
    for table in table_names:
        filtered_df = filter_table_by_keyword(conn, table, keyword)
        if not filtered_df.empty:
            print(f" - Found {len(filtered_df)} rows in table '{table}'")
            docs = rows_to_documents(filtered_df)
            all_docs.extend(docs)

    if not all_docs:
        print("⚠️ No data found matching the keyword.")
        return

    # Split documents into chunks for embeddings
    splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"🧩 Created {len(chunks)} chunks from filtered documents.")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Initialize Ollama LLM (make sure your local Ollama server is running)
    llm = OllamaLLM(model="mistral")

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Example query
    question = f"Give me all {keyword} related details from the database. Give answer as json format"+" like json = ```{}```."
    response = qa.run(question)
    print("\n🧠 Answer:", response)

    try:
        # Flatten lists by joining items with commas
        def flatten_lists(d):
            flat = {}
            for k, v in d.items():
                if isinstance(v, list):
                    flat[k] = ", ".join(str(i) for i in v)
                elif v is None:
                    flat[k] = ""
                else:
                    flat[k] = v
            return flat
        
        # Try to extract JSON inside ```json ... ```
        def parse_response_to_dict(response_text: str):
            response_text = response_text.replace(r'\\', '')
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}")
            
        # Parse JSON string
        response_dict = parse_response_to_dict(response)
        flat_response = flatten_lists(response_dict)
        df = pd.DataFrame([flat_response])
        df.to_csv("final.csv", index=False, encoding="utf-8")
        print("Saved response to 'final.csv'")
    except Exception as e:
        print(f"⚠️ Error saving response as CSV: {e}")

    conn.close()

if __name__ == "__main__":
    main()
