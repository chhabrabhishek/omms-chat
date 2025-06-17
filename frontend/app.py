import gradio as gr
import requests

API_URL = "http://app:8000/chat"


def chat(message, chat_history):
    response = requests.post(API_URL, json={"query": message})
    if response.status_code == 200:
        data = response.json()
        return data.get("response", "No response from server")
    else:
        return f"Error: {response.status_code} - {response.text}"


gr.ChatInterface(chat, title="OMMS Chat").launch(
    server_name="0.0.0.0", server_port=3000
)
