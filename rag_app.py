# Standard Library Import
import time

# Third Party Imports
import chromadb
import streamlit as st
import torch
from transformers import BitsAndBytesConfig

# LlamaIndex Imports
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.vector_stores.chroma import ChromaVectorStore

st.set_page_config(page_title="RAG Query Interface", layout="wide")

st.title("RAG Project: Query Interface")
st.markdown("---")

@st.cache_resource
def setup_rag_pipeline():
    """
    Sets up the full RAG pipeline: chunking, embedding, vectorization, LLM, and reranker.
    Uses "st.cache_resource" to ensure the function runs only once.
    """

    st.info("Setting up RAG pipeline (This may take a few minutes on first run, especially for LLM download)...")
    start_time = time.time()

    # --- 1. TEXT CHUNKING ---
    cruz_file_path = "CRUZ_TEXAS_cleaned.txt"
    n6921c_file_path = "N6921C_cleaned.txt"

    documents = []
    try:
        with open(cruz_file_path, "r", encoding="utf-8") as f:
            text_cruz = f.read()
        with open(n6921c_file_path, "r", encoding="utf-8") as f:
            text_n6921c = f.read()

        # Create the document objects with with content and metadata
            documents = [
            Document(text=text_cruz, metadata={"filename": cruz_file_path, "source": "Cruz_Texas.txt"}),
            Document(text=text_n6921c, metadata={"filename": n6921c_file_path, "source": "N6921C.txt"})
        ]
        st.success(f"Successfully loaded {len(documents)} documents from files.")

    except FileNotFoundError as e:
        st.error(
            f"ERROR: One or both files not found. Please ensure '{cruz_file_path}' and '{n6921c_file_path}' are in the same directory as this script, or adjust the"
            f"paths.\nError: {e}"
        )
        return None, None

    Settings.chunk_size = 2000
    Settings.chunk_overlap = 300
    node_parser = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)

    nodes = []
    if documents:
        nodes = node_parser.get_nodes_from_documents(documents)
        st.info(f"\nGenerated {len(nodes)} chunks (nodes) from the documents.")
    else:
        st.error("\nNo documents loaded, skipping chunking demonstration.")

    # --- 2. Embedding Generation ---
    if nodes:
        embed_model_name = "BAAI/bge-large-en-v1.5"
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        Settings.embed_model = embed_model
        st.info(f"\nConfigured embedding model: {embed_model_name}")

        # --- 3. Generative Model ---
        # Configure BitsAndBytes for 4-bit quantization to reduce memory usage and speed up inference
        llm = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # Quantization type
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        try:
            llm_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            llm = HuggingFaceLLM(
                model_name=llm_model_name,
                tokenizer_name=llm_model_name,
                query_wrapper_prompt="[INST] {query_str} [/INST]",
                context_window=7000,
                max_new_tokens=4096,
                generate_kwargs={"temperature": 0.1, "do_sample": True}, # Generation parameters
                model_kwargs={"quantization_config": quantization_config}, # Model specific arguments
                device_map="auto", # Automatically map model layers to available devices
                tokenizer_kwargs={"max_length": 4096}
            )
            # Set the LLM in LlamaIndex settings
            Settings.llm = llm
            st.success(f"Configured Generative Model (LLM): {llm_model_name}")
    
        except Exception as e:
            st.error(f"Could not configure Mistral LLM.\nError: {e}")
            llm = None

        # --- 4. Vectorization (ChromaDB Storage) ---
        db = chromadb.PersistentClient(path="./chroma_db")
        collection_name = "RAG_Documents_bge_Mistral"
        chroma_collection = db.get_or_create_collection(collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection) # Create a ChromaVectorStore instance
        storage_context = StorageContext.from_defaults(vector_store=vector_store) # Create a storage context

        if llm:
            index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model, llm=llm)

            st.success(f"\nSuccessfully created ChromaDB collection '{collection_name}' and stored nodes.")
            st.info(f"Number of items in ChromaDB collection: {chroma_collection.count()}")

            # --- 5. Query Engine with Reranker ---
            reranker = ColbertRerank(
                model="BAAI/bge-reranker-large",
                top_n=5, # Number of top rank documents to return after reranking
            )
            st.info("Configured BAAI/bge-reranker-large post-processor.")

            query_engine = index.as_query_engine(
                # Retrieve top 10 similar nodes initially
                similarity_top_k=10,
                # Apply the reranker
                node_postprocessors=[reranker]
            )
            st.success("Query engine created successfully")

        else:
            st.error(f"LLM not configured, skipping index and query engine creation.")
            query_engine = None

        end_time = time.time()
        st.success(f"RAG pipeline setup completed in {end_time - start_time:.2f} seconds.")
        return query_engine

# --- Streamlit UI ---
query_engine = setup_rag_pipeline()

if query_engine:
    st.header("Ask a question about the documents:")
    user_query = st.text_area("Your Question:", height=100, placeholder="What is the tail number of the aircraft?")

    if st.button("Get Answer"):
        if user_query.strip():
            with st.spinner("Searching for answers..."):
                try:
                    response = query_engine.query(user_query)
                    st.subheader("Answer:")
                    st.write(response.response)

                    st.subheader("Sources: (Retrieved Chunks):")
                    for i, node in enumerate(response.source_nodes):
                        st.expander(f"Chunk {i + 1} (Score: {node.score:.2f}) - Source: {node.metadata.get('source', 'Not Available')}") \
                                            .markdown(f"**Text:**\n```\n{node.text}\n```")

                except Exception as e:
                    st.error(f"Encountered an error during query processing: {e}")
        else:
            st.warning("Please enter a question.")
else:
    st.error("RAG pipeline could not be fully set up.")
