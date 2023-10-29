import os
import openai
import pickle
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain


load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.form["message"]
    prompt = f"User: {user_input}\nChatbot:"

    messages = [
        {
            "role": "system",
            "content": "You are a chat support for a company that offers that specializes in logistics solution and peer to peer lending",
        }
    ]

    messages.append({"role": "user", "content": prompt})
    
    with open("nasa_docs.txt", "rb") as f:
        docs = pickle.load(f)

    embeddings = OpenAIEmbeddings()

    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(vectorStore_openAI, f)

    with open("faiss_store_openai.pkl", "rb") as f:
        VectorStore = pickle.load(f)

    llm=OpenAI(temperature=0, model_name='text-davinci-003')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=VectorStore.as_retriever())

    if 'peer to peer lending' in user_input.lower():
        ChatGPT_reply = 'A user can make a request to another user for NASA token, once the user approve the request, \
            the token is automatically added to the requester token balance. Once the owner of the token make a request to \
            recover its lend-out NASA token, if the borrower has the token, the token is automatically deducted from his wallet \
            and returned back to the owner.'
    else:
        ChatGPT_reply = qa.run(user_input)

    #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    #ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})

    return render_template(
        "chatbot.html",
        user_input=user_input,
        bot_response=ChatGPT_reply)

if __name__ == "__main__":
    app.run(debug=True)