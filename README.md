💬 AI Customer Support Agent
An intelligent, multi-step customer support assistant built with LangChain, LangGraph, and Groq LLMs.
It automatically classifies, analyzes, and responds to user queries across billing, technical, and general topics — and escalates only when needed.

📁 Project Structure
plaintext
Copy
Edit
ai-customer-support/
├── src/
│   ├── __init__.py
│   ├── state.py        # Defines shared agent state
│   ├── nodes.py        # Node handlers for classification, sentiment, etc.
│   ├── graph.py        # LangGraph workflow configuration
│   ├── app.py          # Streamlit frontend UI
│   └── test_agent.py   # Optional test driver
├── .env                # Contains GROQ_API_KEY
├── README.md
└── pyproject.toml      # Project dependencies and metadata
🚀 Quick Start
✅ Prerequisites
Python 3.11+

Valid Groq API key from https://console.groq.com

⚙️ Installation
bash
Copy
Edit
git clone https://github.com/yourusername/ai-customer-support.git
cd ai-customer-support
python -m venv .venv
.venv\Scripts\activate
pip install -e .
Create a .env file in the root directory:

env
Copy
Edit
GROQ_API_KEY=your_api_key_here
▶️ Run the App
bash
Copy
Edit
streamlit run src/app.py
Access it at http://localhost:8501.

🧪 Example Support Queries
💳 Billing
“How do I update my payment method?”

“Why was I charged twice?”

“I need help with my subscription.”

🛠 Technical
“The app keeps crashing.”

“Getting error code 404.”

“Can’t log into my account.”

📦 General
“What are your hours?”

“Do you ship internationally?”

“How can I track my order?”

🚨 Escalation Triggers
“This is unacceptable!”

“I want to speak to a human.”

“I demand a refund NOW.”

🧱 Architecture
🧠 State Management
Field	Description
messages	Full chat history
category	Billing, technical, general, etc.
sentiment	Positive, neutral, or negative
agent_scratchpad	Tracks toolchain usage (optional)
error	Captures LLM/tool errors

🧩 Workflow
plaintext
Copy
Edit
User Message ➝ Categorization ➝ Sentiment ➝ Route ➝ Sub-Agent ➝ Response
🔧 Node Overview
Node	Role
categorize	Classify the user’s query
analyze_sentiment	Detect tone (positive/negative)
handle_billing	Billing & subscription queries
handle_technical	Tech help & troubleshooting
handle_general	General inquiries
handle_escalation	Escalates to human if necessary

🧪 Testing
Run the test agent manually:

bash
Copy
Edit
cd ai-customer-support
python src/test_agent.py
⚙️ Development & Debugging
Debug Logging
Enable verbose logs by setting:

env
Copy
Edit
LOGGING_LEVEL=DEBUG
VS Code Settings (Recommended)
json
Copy
Edit
{
  "python.testing.pytestEnabled": true,
  "python.analysis.typeCheckingMode": "basic",
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}
📦 Dependencies
Listed in pyproject.toml. Core packages include:

Package	Version
langchain-core	^0.3.49
langchain-groq	^0.3.2
langgraph	^0.4.8
streamlit	^1.32.0
python-dotenv	^1.0.1

🧑‍💻 Contributing
🍴 Fork this repository

🌿 Create a new feature branch

💬 Make changes with clear commit messages

✅ Open a pull request

📄 License
MIT License — free to use, adapt, and share.

🆘 Support
🐛 GitHub Issues

📧 Email: support@example.com

Built with ❤️ using LangChain, LangGraph, and Groq LLMs.

