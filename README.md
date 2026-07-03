# Agentic RAG Assistant

Agentic RAG Assistant is a multi-agent question-answering system that routes user queries to the right specialist agent. It combines Retrieval-Augmented Generation for HR and technical documentation with a SQL-backed project agent for structured internship/project data.

The application includes a Streamlit chat interface, Gemini 2.5 Flash for response generation and routing, LangChain-based retrieval, ChromaDB vector stores, and SQLite for project records.

## Features

- Intelligent router that classifies questions across HR, technical, and project domains
- HR RAG Agent for answering policy-related questions from HR documents
- Technical RAG Agent for answering engineering and setup questions from technical documents
- Project SQL Agent for querying structured intern, mentor, project, and status data
- ChromaDB vector retrieval with LangChain retrievers
- SQLite-backed project database
- Gemini 2.5 Flash integration
- Streamlit chat UI with persistent session chat history
- Modular source structure for agents, routing, RAG, database access, and utilities

## Architecture Diagram

```text
                  User
                    │
                    ▼
             Streamlit UI
                    │
                    ▼
            process_query()
                    │
                    ▼
             Intelligent Router
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   HR Agent   Technical   Project Agent
        │          │          │
        ▼          ▼          ▼
    ChromaDB   ChromaDB    SQLite
        │          │          │
        └──────────┼──────────┘
                   ▼
          Gemini 2.5 Flash
                   ▼
             Final Response
```

## Folder Structure

```text
Agentic-RAG/
|-- data/
|   |-- hr_docs/
|   |   `-- hr_policy.md
|   |-- sample_db/
|   |   |-- interns.csv
|   |   `-- interns.db
|   `-- technical_docs/
|       `-- technical_guide.md
|-- notebooks/
|-- src/
|   |-- agents/
|   |   |-- hr_agent.py
|   |   |-- project_agent.py
|   |   `-- technical_agent.py
|   |-- database/
|   |   |-- models.py
|   |   |-- sql_db.py
|   |   `-- sql_queries.py
|   |-- rag/
|   |   |-- embeddings.py
|   |   |-- retriever.py
|   |   `-- vector.store.py
|   |-- router/
|   |   `-- router.py
|   |-- utils/
|   `-- app.py
|-- tests/
|-- vector_db/
|-- streamlit_app.py
|-- requirements.txt
|-- .env
`-- README.md
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Agentic-RAG
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The key is used by Gemini 2.5 Flash for routing and agent responses.

## Running the Project

The project can be launched either through the Streamlit web interface or from the command line.

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

The Streamlit app opens a browser-based chat interface where you can ask HR, technical, or project-related questions.

### Command Line Version

```bash
python src/app.py
```

The command line version runs the core application flow directly from the terminal.

## Example Questions

### HR Agent

- How many leaves are allowed?
- What is the work from home policy?
- What are the HR rules for attendance?

### Technical Agent

- How do I install Docker?
- What are the setup steps for the project?
- How should I configure the development environment?

### Project SQL Agent

- Who is working on the Platform project?
- Which interns are assigned to the payment module?
- Show active interns.
- Who is the mentor for this project?
- Which projects are completed?

## Technologies Used

- Python
- Streamlit
- Gemini 2.5 Flash
- LangChain
- ChromaDB
- SQLite
- Hugging Face sentence-transformer embeddings
- pandas
- python-dotenv

## Future Improvements

- Add authentication for production deployments
- Add automated evaluation for retrieval quality
- Add chat export and conversation persistence
- Add admin tools for uploading new HR and technical documents
- Add database migration support for structured project data
- Add observability for routing decisions, retrieval scores, and latency
- Add Docker support for consistent deployment
- Add CI checks for formatting, tests, and security scanning

## Screenshots

Create a folder named:

screenshots/

Add images such as:

- home.png
- hr-query.png
- technical-query.png
- project-query.png

Then reference them like:

![Home](screenshots/home.png)

![HR Query](screenshots/hr-query.png)

![Technical Query](screenshots/technical-query.png)

![Project Query](screenshots/project-query.png)

## Author

**Divanshu Kumawat**

M.Sc. Data Science

Amity University Rajasthan

GitHub: divanshukumawat05-cmd

LinkedIn: https://www.linkedin.com/in/divanshu-kumawat-0b92073b1/
