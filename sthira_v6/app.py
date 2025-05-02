import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model_name = "tanusrich/Mental_Health_Chatbot"
model, tokenizer, device = load_model_tokenizer(model_name)

# Function to generate a response
def generate_response(user_input, chat_history):
    # Add a system prompt to guide the model behavior
    system_prompt = "You are a supportive mental health chatbot. Keep responses brief and warm. Always end with a friendly follow-up question to continue the conversation."
    
    input_text = system_prompt + "\n\n" + " ".join(chat_history + [user_input])
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=75,  # Reduced to encourage shorter responses
            temperature=0.8,    # Slightly higher for more varied responses
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True     # Add sampling for more natural responses
        )
    response = tokenizer.decode(output[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)
    
    # If the response doesn't end with a question, add a friendly follow-up
    if not any(response.strip().endswith(q) for q in ["?", "!"]):
        friendly_questions = [
            "How are you feeling about that right now?",
            "What would help you feel better today?",
            "Would you like to share more about that?",
            "How has your day been going otherwise?",
            "Is there anything specific on your mind you'd like to talk about?",
            "What do you think might help in this situation?",
            "Have you tried anything that's helped with this before?",
            "What would make you smile right now?",
            "How can I best support you today?"
        ]
        import random
        response += " " + random.choice(friendly_questions)
    
    return response.strip()

# Streamlit app
st.title("Mental Health Chatbot")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "past_messages" not in st.session_state:
    st.session_state["past_messages"] = []
if "generated_responses" not in st.session_state:
    st.session_state["generated_responses"] = []

user_input = st.text_input("You: ")

if user_input:
    st.session_state["past_messages"].append(user_input)
    response = generate_response(user_input, st.session_state["past_messages"])
    st.session_state["generated_responses"].append(response)

if st.session_state["past_messages"]:
    for i in range(len(st.session_state["past_messages"])):
        st.markdown(f'<div style="background-color:#f0f2f6;padding:10px;border-radius:5px;">You: {st.session_state["past_messages"][i]}</div>', unsafe_allow_html=True)
        if i < len(st.session_state["generated_responses"]):
            st.markdown(f'<div style="background-color:#e1f5fe;padding:10px;border-radius:5px;">Chatbot: {st.session_state["generated_responses"][i]}</div>', unsafe_allow_html=True)