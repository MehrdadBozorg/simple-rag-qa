# Simplified RAG-Based Document Retrieval and Question Answering System

This repository provides a minimal yet effective Retrieval-Augmented Generation (RAG) pipeline for document retrieval and question answering. It integrates a pre-trained embedding model to encode documents, a lightweight vector database for efficient similarity search, and a Large Language Model (LLM) to generate context-aware responses.

## Features

- **Document Embedding & Retrieval**: Uses SentenceTransformers for vectorization and FAISS index as the vector database for storage and retrieval.
- **LLM-Powered Answer Generation**: Retrieves relevant documents and utilizes an LLM to generate context-aware responses.
- **FastAPI-Based API**:
  - `/upload`: Uploads documents, extracts text, and stores embeddings.
  - `/ingest`: Retrieves relevant documents and generates answers.
- **Containerization**: Dockerfile(s) and `docker-compose.yml` for easy deployment.
- **Comprehensive Documentation**: Includes setup instructions, API usage, and well-structured, modular code.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.10+
- Poetry (for dependency management)
- Docker (optional, for containerized deployment)

### Setup locally

1- Clone the repository

```sh
# Clone the repository
git clone https://github.com/MehrdadBozorg/simple-rag-qa.git
cd simplified-rag
```

2- Install dependencies using `poetry` to run app locally

```sh
# Install Poetry if not installed
pip install poetry

# Install dependencies
poetry install
```

3- **Running the API**

```sh
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Upload Documents

- **Endpoint**: `POST /upload`
- **Description**: Accepts document files, extracts text, and stores embeddings.
- **Example Request**:
  - Upload a `.txt` or `.pdf` file using Postman or a cURL command:
  ```sh
  curl -X 'POST' \
    'http://127.0.0.1:8000/upload' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@document.pdf'
  ```

#### 2. Query for Answers

- **Endpoint**: `POST /ingest`
- **Description**: Accepts user queries, retrieves relevant documents, and generates an answer.
- **Example Request**:
  ```json
  {
    "query": "What is the content of the uploaded document?"
  }
  ```

## Deployment

### Docker

```sh
# Build and run the Docker container
docker build -t rag-system .
docker run -p 8000:8000 rag-system
```

### Using `docker-compose`

```sh
docker-compose up -d
```

## Contributing

Pull requests are welcome! Please open an issue for feature requests or bug reports.

## License

This project is licensed under the MIT License.