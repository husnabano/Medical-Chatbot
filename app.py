from flask import Flask, request, render_template
from dotenv import load_dotenv
import os

# LangChain & Groq imports
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ===============================
# 0. Keep conversation in memory (resets on server restart)
# ===============================
chat_history = []

# ===============================
# 1. Setup Flask
# ===============================
app = Flask(__name__)

# ===============================
# 2. Load Environment & Keys
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===============================
# 3. Load Embeddings + FAISS
# ===============================
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

# ===============================
# 4. Setup Groq LLM + RetrievalQA
# ===============================
llm = ChatGroq(
    model="gemma2-9b-it",  # you can change model here
    temperature=0.0,
    api_key=GROQ_API_KEY
)

retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ===============================
# 5. Flask Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        if query:
            # Append user message
            chat_history.append({"type": "user", "text": query})

            # Get bot response
            response = qa_chain.invoke(query)
            answer = response["result"]
            sources = [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]]

            # Append bot message
            bot_text = f"{answer}\nSources: {', '.join(sources)}"
            chat_history.append({"type": "bot", "text": bot_text})

    return render_template("index.html", chat_history=chat_history)

# ===============================
# Run the App
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
