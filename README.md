# ShopEase — AI Customer Support Agent

A Python-based customer support AI agent that combines **Retrieval-Augmented Generation (RAG)**, secure tool calling, and evaluation to answer customer questions using a knowledge base and mock order data.

## Features

- **Knowledge-grounded answers:** Retrieves relevant shipping, return, refund, and support policy documents.
- **Source citations:** Includes references to retrieved policy information.
- **Order-status lookup:** Uses a tool to retrieve order information from mock data.
- **Authorization checks:** Validates customer ownership before revealing order details.
- **Safe handling:** Supports refusals and human escalation for operations the agent cannot perform.
- **User interface and API:** Streamlit and FastAPI entry points.
- **Observability:** LangSmith tracing of agent, retrieval, model, and tool execution.
- **Evaluation:** pytest tests, a golden dataset, and repeatable offline LangSmith experiments.

## Architecture

The application has two main information paths:

1. **Policy questions:** User question → document retrieval → policy context → LLM → source-grounded answer.
2. **Order questions:** User question → model requests order lookup → deterministic ownership validation → mock order tool → LLM response.

LangSmith traces the execution path to support debugging and evaluation.

## Technology Stack

| Component | Technology |
|---|---|
| Application language | Python |
| LLM integration | LangChain and OpenAI |
| Retrieval | Knowledge base and vector search |
| Vector store | Chroma |
| Backend API | FastAPI |
| UI | Streamlit |
| Testing | pytest |
| Evaluation and tracing | LangSmith |
| Version control | Git and GitHub |

## Project Structure

```text
customer-support-agent/
├── app/
│   ├── combined_agent.py
│   ├── api.py
│   ├── streamlit_ui.py
│   ├── rag.py
│   ├── search_knowledge.py
│   ├── document_processing.py
│   ├── vector_store.py
│   └── tools/
├── data/
│   ├── knowledge_base/
│   │   ├── return_policy.md
│   │   ├── refund_policy.md
│   │   ├── shipping_policy.md
│   │   └── contact_support.md
│   ├── orders.json
│   └── order_owners.json
├── tests/
├── offline_eval.py
├── create_shopease_golden_dataset.py
├── requirements.txt
├── .env.example
└── README.md
```

The `app/` directory also contains earlier learning-stage implementations.

## Local Setup

**Prerequisites:** Python, an OpenAI API key, and optionally a LangSmith API key.

Clone the repository and create a virtual environment:

```bash
git clone <PRIVATE_REPOSITORY_URL>
cd shopease-customer-support-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` based on `.env.example` and add your credentials. Never commit the populated `.env`.

Before running the application, initialize the knowledge-base index using the project's ingestion script or existing local setup procedure.

## Testing

Run the automated tests:

```bash
python -m pytest tests/ -v
```

Run the evaluator's local self-test:

```bash
python offline_eval.py --self-test
```

Run the LangSmith offline evaluation:

```bash
python offline_eval.py
```

The offline evaluation uses the `shopease-support-golden-set` dataset, which includes six policy and safety scenarios covering returns, refunds, digital products, support contact information, cancellation, and undelivered packages.

## Evaluation Strategy

The project uses three complementary evaluation methods:

- **Unit and integration tests:** Verify deterministic application behavior, permissions, and failure handling.
- **Offline regression evaluations:** Compare agent behavior against curated golden cases.
- **Online feedback and traces:** Help investigate real execution behavior, quality, and latency.

Passing deterministic text checks does not prove that an answer is fully correct or grounded. Ambiguous cases require trace inspection and human review.

## Security and Limitations

- Order data and customer identifiers are synthetic and intended for demonstration.
- API keys are managed through environment variables.
- Customer authorization checks are implemented in deterministic application code.
- Mock authentication must not be treated as production-grade identity verification.
- The prototype has not been validated as production-ready.
- Remaining work includes production authentication, privacy controls, more extensive regression coverage, and operational alerting.

## Engineering Decisions

**Why RAG?** To use retrieved policy documents rather than relying on the LLM's general knowledge.

**Why tools?** To access structured order records that cannot be reliably answered from policy documents.

**Why deterministic authorization?** Security decisions must not depend on the model's willingness to follow instructions.

**Why LangSmith?** To inspect execution traces, debug failures, and repeat regression evaluations.

**Why a single-agent design?** The problem can be handled without the additional coordination complexity of a multi-agent architecture.

## Project Status

Local prototype with RAG, tool calling, FastAPI, Streamlit, tracing, and evaluation. No public deployment is claimed.
