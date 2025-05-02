import gradio as gr
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chat_models import ChatGroq
import os

def initialize_llm():
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key="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",
        model_name="llama-3-70b-versatile"
    )
    return llm

def create_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory

def setup_conversational_chain(llm, memory):
    system_prompt = """
You are Sthira, a compassionate and calm emotional support companion.

- The user might speak in:
    - English
    - Hinglish (Hindi mixed with English)
    - Marathi written using English alphabets
- Always reply in the **same language style** that the user is using.
- Understand Hinglish, Marathi-English, and respond empathetically.
- Let the user express their emotions fully, listen patiently.
- Use warm, non-judgmental, comforting words.
- Validate their feelings (e.g., "It's okay to feel like this.", "Main samajh sakta hoon aapka dukh.").
- Invite them to share more: ("Kya aap aur batana chahenge?", "Would you like to tell me more?", "Ajun kahi sangaycha ahe ka?").
- Do not correct their language, spelling, or grammar.
- You are not a therapist, just a caring friend.

Start by warmly asking how they are feeling today.
"""
    prompt = PromptTemplate(input_variables=["chat_history", "input"], template=system_prompt + "\n{chat_history}\nHuman: {input}\nSthira:")

    conversational_chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )
    return conversational_chain

# Initialize LLM and memory once
llm = initialize_llm()
memory = create_memory()
conversation_chain = setup_conversational_chain(llm, memory)

def chat_with_sthira(user_message, chat_history):
    response = conversation_chain.predict(input=user_message)
    chat_history.append((user_message, response))
    return chat_history, chat_history

# Launch Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("<h1 style='text-align: center;'>🧘‍♂️ Sthira - Your Calm Support Companion 💬</h1>")

    chatbot = gr.Chatbot([], elem_id="chatbot", height=500)
    user_input = gr.Textbox(placeholder="Type your feelings here...", show_label=False)

    submit_button = gr.Button("Send")

    clear_button = gr.Button("Clear Chat")

    state = gr.State([])

    submit_button.click(chat_with_sthira, [user_input, state], [chatbot, state])
    user_input.submit(chat_with_sthira, [user_input, state], [chatbot, state])  # also allow hitting enter

    clear_button.click(lambda: ([], []), inputs=None, outputs=[chatbot, state])

# Run the app
app.launch()
