# Project Specifications: Semantic Book Recommender

## 1. Project Overview

The **Semantic Book Recommender** is a RAG based book recommendation system.

## 2. Tech Stack

- **Programming Language**: Python
- **Data Handling**: `pandas`, `numpy`
- **Machine Learning & NLP**:
  - `langchain`: Framework for building the RAG (Retrieval-Augmented Generation) pipeline.
  - `langchain-openai`: For integration with OpenAI-compatible LLM services (Cerebras).
  - `fastembed`: High-performance library for generating text embeddings.
  - `ChromaDB`: Vector database for efficient similarity search.
- **User Interface**: `gradio` (Conversational Hybrid UI)
- **LLM**: Cerebras `gpt-oss-120b`
- **Embeddings Model**: `BAAI/bge-small-en-v1.5` (via FastEmbed)

## 3. Data Processing Pipeline

1.  **Data Loading**: The system loads book metadata from `books_cleaned.csv`.
2.  **Feature Engineering**: A `tagged_description` field is created by combining the ISBN13 and the book description to provide more context for indexing.
3.  **Text Splitting**: Descriptions are split into smaller chunks using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 100) to ensure they fit within the model's context window and improve retrieval granularity.
4.  **Vectorization**: Text chunks are converted into high-dimensional vectors using the `bge-small-en-v1.5` model.
5.  **Indexing**: These vectors are stored in a local `ChromaDB` instance (`./chroma_db`) for persistent and fast retrieval.

## 4. Recommendation Logic

- **Semantic Search**: When a user enters a query (e.g., "A story about forgiveness"), the system:
  1.  Converts the query into a vector using the same embedding model.
  2.  Performs a similarity search in `ChromaDB` to find the top 50 most relevant chunks.
  3.  Maps these chunks back to their respective books using ISBN13 metadata.
- **Ranking**: The system returns the top 5 unique book recommendations based on the search results.
- **Conversational Explanation**: The LLM (Cerebras) takes the retrieved book metadata and generates a conversational response explaining why these specific books match the user's request.

## 5. User Interface (Gradio App)

- **Input**: A conversational chatbot interface where users can describe their preferences in natural language.
- **Output**: A hybrid layout featuring:
  - **Chatbot**: Provides ranked, conversational explanations for each recommendation.
  - **Markdown Gallery**: A dedicated sidebar displaying book covers, titles, authors, and full descriptions without truncation.

## 6. Key Features

- **Semantic Understanding**: Finds books based on themes, moods, and plot descriptions rather than just titles or authors.
- **Conversational AI**: Explains the reasoning behind each recommendation using a state-of-the-art LLM.
- **Local Persistence**: The vector database is stored locally, allowing for fast subsequent startups without re-indexing.

## 7. Project Structure

- `gradio_app.py`: The main entry point for the web application.
- `vector_search.ipynb`: Development notebook for testing the vector search logic.
- `data_exploration.ipynb`: Notebook for initial data analysis and cleaning.
- `books_cleaned.csv`: The primary dataset.
- `chroma_db/`: Directory containing the persisted vector database.
- `fastembed_cache/`: Directory for cached embedding models.
