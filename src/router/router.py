from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


def route_query(user_query: str):
    try:
        prompt = f"""
You are an intelligent routing agent.

Your task is to classify a user's question into exactly ONE category.

Categories:

HR
Questions related to:
- leave
- attendance
- holidays
- stipend
- internship policy
- HR rules
- working hours
- code of conduct
- mentor communication
- performance evaluation

TECHNICAL
Questions related to:
- Python
- Programming
- VS Code
- Git
- Docker
- APIs
- Installation
- Running the project
- Project setup
- Development
- Deployment
- Debugging
- Coding
- Software

PROJECT
Questions related to:
- Project status
- Project progress
- Assigned project
- Completion percentage
- Mentor assigned project
- Database project information

Return ONLY one of:

HR

TECHNICAL

PROJECT

Never explain your answer.

Never return sentences.

Return exactly one word.

User Question:

{user_query}
"""

        response = llm.invoke(prompt)

        if not hasattr(response, "content"):
            return "TECHNICAL"

        category = str(response.content).strip().upper()

        if category not in {"HR", "TECHNICAL", "PROJECT"}:
            return "TECHNICAL"

        return category
    except Exception as exc:
        print("Router Error:", repr(exc))
        return "TECHNICAL"
