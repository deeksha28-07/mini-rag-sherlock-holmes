# Mini RAG System

This project is a small Retrieval-Augmented Generation (RAG) system built using OpenAI and Qdrant. It answers questions about five Sherlock Holmes stories and gives source citations.

## What the Project Does

The pipeline follows these steps:

```text
Story text files → chunks → embeddings → Qdrant database
Question → retrieve relevant chunks → GPT answer with citations
```

The project uses:

- `text-embedding-3-small` for embeddings
- Qdrant as the vector database
- `gpt-4o-mini` to generate answers
- Five Sherlock Holmes story files as the corpus

## Project Files

- `mp2_rag.py` — main working RAG pipeline
- `corpus/` — five Sherlock Holmes story text files
- `data/predefined_questions.jsonl` — two provided validation questions
- `data/learner_questions.jsonl` — three learner-created questions
- `.env` — private OpenAI and Qdrant keys; do not submit it
- `requirements.txt` — required Python libraries
- `mp2_validation.txt` — saved validation results
- `mp2_reflection.md` — project reflection

## Functions in mp2_rag.py

### load_corpus

This function reads every `.txt` file in the `corpus` folder. It saves the filename, story title, and complete text for each story.

### chunk_document

This function splits each story into smaller chunks of about 500 characters with overlap. Each chunk keeps its source filename, story title, section name, and text.

The project created 87 chunks from the five stories.

### embed_texts

This function sends chunk text or a question to OpenAI's `text-embedding-3-small` model. It returns vectors, which are numerical representations of meaning.

### setup_collection

This function creates the Qdrant collection named `mp2_sherlock`. It uses 1536 dimensions because that is the size of vectors created by `text-embedding-3-small`.

### ingest_chunks

This function embeds every story chunk and uploads the chunks, vectors, and metadata to Qdrant.

### retrieve

This function embeds the user question and searches Qdrant for the top three most relevant chunks. It returns the chunk text, source information, section, and similarity score.

### answer

This function retrieves relevant chunks, formats them as context, and sends them with the question to `gpt-4o-mini`. The model answers only from the retrieved excerpts and includes source citations.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your private keys:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export QDRANT_URL="your-qdrant-cluster-url"
export QDRANT_API_KEY="your-qdrant-api-key"
```

Load the keys:

```bash
source .env
```

## How to Run the Project

### 1. Ingest the corpus

```bash
python mp2_rag.py ingest
```

This loads five documents, creates chunks, creates the Qdrant collection, and uploads all chunk embeddings.

Expected output is similar to:

```text
5 documents loaded
Total chunks: 87
✓ Done. Try: python mp2_rag.py ask
```

### 2. Ask questions

```bash
python mp2_rag.py ask
```

Example question:

```text
Who was Vincent Spaulding really?
```

The system returns an answer, source sections, and latency. Press Enter on an empty line to exit.

### 3. Run validation

```bash
python mp2_rag.py validate | tee mp2_validation.txt
```

This runs the two predefined questions and the three learner questions. It also saves the output into `mp2_validation.txt`.

## Interpreting Results

A `✓` means the correct story was retrieved.

My validation results were:

- Predefined questions: 2/2 source matches
- Learner questions: 3/3 source matches
- Total: 5/5 source matches

`Facts matched` checks whether the answer contains exact expected words. A lower facts-matched value does not always mean the answer is wrong, because the model may use different but correct wording.

For example, the first predefined question had a 1/5 facts-matched score but retrieved the correct source and gave the correct answer.

## Known Notes

The Qdrant `recreate_collection` warning is only a deprecation warning. It does not stop the project from working.

The retrieval function supports both older Qdrant `search()` and newer Qdrant `query_points()` methods.

## Submission

Submit these files:

- `mp2_rag.py`
- `README.md`
- `data/learner_questions.jsonl`
- `mp2_reflection.md`
- `mp2_validation.txt`
- `corpus/`
- `requirements.txt`

Do not submit `.env`, `.venv`, API keys, or Qdrant credentials.