# Customer Support AI Agent

An AI-powered customer support application built as a hands-on AI engineering project.

## V1 Scope

- Answer company policy and FAQ questions using RAG
- Retrieve relevant document passages and cite sources
- Look up orders using a mock database
- Ask for missing information
- Handle errors and unsupported requests
- Escalate requests requiring human assistance

## Planned Tech Stack

Python, LangChain, LangSmith, Chroma, FastAPI, Streamlit, and pytest.

## Local Setup

**Prerequisites:** Python 3.13 and Git.

1. Clone or download the repository.
2. Create a virtual environment:

   `python3 -m venv .venv`

3. Activate it on macOS/Linux:

   `source .venv/bin/activate`

4. Install dependencies:

   `python -m pip install -r requirements-lock.txt`

5. Create the local environment file:

   `cp .env.example .env`

6. Add your own API credentials to `.env`.

7. Run tests:

   `python -m pytest -v`

## Security

Never commit `.env`, API keys, customer information, or local environment files.

## Project Status

Phase 2 — Environment setup.

The LLM agent, RAG, order lookup, API, and UI will be implemented incrementally.

## Current Limitations

- No live customer integrations
- No refund or cancellation execution
- No deployed application yet
