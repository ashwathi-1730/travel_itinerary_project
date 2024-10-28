from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Initialize the model and prompt
template = """
Answer the question below.
Here is the conversation history: {context}
Question: {question}
Answer:
"""
model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Store conversation context
context = ""

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/chat", methods=["POST"])

def chat():
    global context
    user_input = request.json.get("user_input")
    
    # Generate response from the model
    result = chain.invoke({"context": context, "question": user_input})
    
    # Format the response with basic HTML tags for better display
    formatted_result = result.replace("\n", "<br>")  # Replace line breaks with <br> tags
    
    # Append to conversation context
    context += f"\nUser: {user_input}\nAI: {result}"
    
    # Return the bot's response
    return jsonify({"response": formatted_result})


if __name__ == "__main__":
    app.run(debug=True)
