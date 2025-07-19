import streamlit as st
import requests
import json
import uuid

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/chat"

# --- Initialisation de session ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Titre ---
st.title("Ella")

# --- Affichage des messages précédents ---
with st.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Formulaire utilisateur ---
with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Posez votre question ici (ou uploadez une image)...", key="prompt_input")
    uploaded_file = st.file_uploader("Uploader une image (optionnel)", type=["png", "jpg", "jpeg", "gif", "tiff", "bmp"], key="file_uploader")
    submitted = st.form_submit_button("Envoyer")

if submitted:
    user_content = prompt.strip() if prompt else ""
    if uploaded_file:
        user_content += f"\n[Image: {uploaded_file.name}]"

    st.session_state.messages.append({"role": "user", "content": user_content})

    # Préparation de la requête
    data = {
        "query": prompt,
        "session_id": st.session_state.session_id
    }

    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)} if uploaded_file else None

    # Requête POST vers l’API FastAPI
    with st.spinner("En attente de la réponse..."):
        try:
            response = requests.post(API_URL, data=data, files=files)

            if response.status_code == 200:
                api_response = response.json()
                bot_answer = api_response.get("answer", "Réponse vide.")
                st.session_state.messages.append({"role": "assistant", "content": bot_answer})
            else:
                try:
                    error_details = response.json().get("detail", "Erreur inconnue.")
                except:
                    error_details = response.text[:100]
                msg = f"Erreur de l'API : {response.status_code} - {error_details}"
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.error(msg)

        except requests.exceptions.ConnectionError:
            msg = f"Erreur de connexion : Impossible de joindre l'API à {API_URL}."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.error(msg)

        except Exception as e:
            msg = f"Erreur inattendue : {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.error(msg)
