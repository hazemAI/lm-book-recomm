import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv

print("--- Starting Semantic Book Recommender ---")

try:
    print("Loading dependencies...")
    from langchain_core.documents import Document
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    import gradio as gr

    print("Dependencies loaded successfully.")
except ImportError as e:
    print(f"Error loading dependencies: {e}")
    print("Please run: uv pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during import: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.cerebras.ai/v1")
MODEL_ID = os.getenv("MODEL_ID", "gpt-oss-120b")


def initialize_system():
    print("Initializing embeddings and vector store...")
    try:
        books = pd.read_csv("books_cleaned.csv")

        # Ensure local path for placeholder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        placeholder_path = os.path.join(current_dir, "cover-not-found.jpg")

        books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
        books["large_thumbnail"] = np.where(
            books["large_thumbnail"].isna(),
            placeholder_path,
            books["large_thumbnail"],
        )

        books.set_index("isbn13", inplace=True)

        embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="./fastembed_cache",
            threads=4,
        )

        # Load the vector store from disk if it exists, otherwise index (first 4 only)
        if os.path.exists("./chroma_db"):
            print("Loading existing vector database...")
            db_books = Chroma(
                persist_directory="./chroma_db", embedding_function=embeddings
            )
        else:
            print("Indexing books... this may take a few minutes for the first time.")
            raw_documents = [
                Document(
                    page_content=row["tagged_description"],
                    metadata={"isbn13": row.name},
                )
                for _, row in books.iterrows()
            ]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            documents = text_splitter.split_documents(raw_documents)
            db_books = Chroma.from_documents(
                documents, embeddings, persist_directory="./chroma_db"
            )

        print("System initialized successfully.")
        return books, db_books
    except Exception as e:
        print(f"Error during initialization: {e}")
        sys.exit(1)


# Global variables (initialized in main)
books = None
db_books = None
llm = None


def get_llm():
    global llm
    if llm is None:
        if not OPENAI_API_KEY:
            return None
        try:
            llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_API_BASE,
                model=MODEL_ID,
                temperature=0.2,
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return None
    return llm


def retrieve_semantic_recommendations(
    query: str,
    initial_top_k: int = 50,
    final_top_k: int = 5,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.metadata["isbn13"]) for rec in recs]
    book_recs = books.loc[books.index.intersection(books_list)].head(final_top_k)
    return book_recs


def format_book_markdown(row, rank):
    """Format book data into a Markdown card for the sidebar."""
    description = row.get("description", "No description available.")
    authors = row.get("authors", "Unknown Author").replace(";", ", ")

    img_src = row["large_thumbnail"]
    if not img_src.startswith("http"):
        # Use absolute path with file/ prefix for Gradio 6 local file serving
        img_src = "file/" + os.path.abspath(img_src).replace("\\", "/")

    return f"### {rank}. {row['title']}\n**By {authors}**\n\n![{row['title']}]({img_src})\n\n{description}\n\n---"


def process_query(message, history):
    """
    Process the user query:
    1. Retrieve books
    2. Update Markdown Gallery
    3. Generate LLM response
    """
    current_llm = get_llm()
    if not OPENAI_API_KEY or current_llm is None:
        error_msg = "Please provide a valid `OPENAI_API_KEY` in the `.env` file."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""

    try:
        # 1. Retrieve books
        recommendations = retrieve_semantic_recommendations(message)

        if recommendations.empty:
            response_msg = "I couldn't find any books matching your request. Could you try describing it differently?"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response_msg})
            return history, "No books found."

        # 2. Format for Markdown Gallery
        gallery_md = []
        context_list = []

        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            # Context for LLM
            context_list.append(
                f"Title: {row['title']}, Author: {row['authors']}, Description: {row['description']}"
            )
            # Item for Gallery
            gallery_md.append(format_book_markdown(row, i))

        context = "\n\n".join(context_list)

        # 3. LLM Response
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful book recommender. Based on the following books retrieved from our database, explain why they match the user's request. \n\nCRITICAL: Use a numbered list (1, 2, 3...) in Markdown to present the recommendations, reflecting their relevance rank. For each book, you MUST explicitly mention the **Title** and **Author** in bold. \n\nDo NOT include any introductory or concluding text. Start directly with the first recommendation. If no books seem relevant, say so.",
                ),
                ("user", "User Request: {message}\n\nRetrieved Books:\n{context}"),
            ]
        )

        chain = prompt | current_llm
        response = chain.invoke({"message": message, "context": context})

        # Update history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.content})

        return history, "\n\n".join(gallery_md)

    except Exception as e:
        error_msg = f"An error occurred: {e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


def main():
    global books, db_books
    books, db_books = initialize_system()

    with gr.Blocks() as dashboard:
        gr.Markdown("# 📚 Semantic Book Recommender")
        gr.Markdown(
            "Ask me anything like: *'I'm looking for a dark fantasy with complex characters'* or *'Suggest a book about the history of science'*. I'll find the best matches and explain why they fit!"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=700, label="Conversation")
                msg = gr.Textbox(
                    label="Describe the book you're looking for...",
                    placeholder="e.g., A story about forgiveness",
                    lines=2,
                )
                submit_btn = gr.Button("Find Recommendations", variant="primary")
                clear_btn = gr.Button("Clear")

            with gr.Column(scale=2):
                gr.Markdown("## Recommended Books")
                gallery = gr.Markdown(height=800)

        # Event Wiring
        submit_btn.click(
            fn=process_query, inputs=[msg, chatbot], outputs=[chatbot, gallery]
        ).then(
            fn=lambda: "",
            outputs=[msg],  # Clear input box
        )

        msg.submit(
            fn=process_query, inputs=[msg, chatbot], outputs=[chatbot, gallery]
        ).then(
            fn=lambda: "",
            outputs=[msg],
        )

        clear_btn.click(lambda: ([], []), outputs=[chatbot, gallery])

    print("Launching Gradio interface...")
    dashboard.launch(
        allowed_paths=[os.getcwd()],
        css=".gradio-container { max-width: 1400px !important; } .gallery-item .caption { font-size: 14px !important; line-height: 1.4 !important; }",
    )


if __name__ == "__main__":
    main()
