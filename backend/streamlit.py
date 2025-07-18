import streamlit as st
import requests
import json
import uuid

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/chat"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
# --- Titre de l'application Streamlit ---
st.title("Ella")

# --- Initialisation de l'historique de conversation ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Conteneur pour l'affichage des messages ---
chat_container = st.container()

# --- Zone de saisie et d'upload dans un formulaire ---
with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Posez votre question ici (ou uploadez une image)...", key="prompt_input")
    uploaded_file = st.file_uploader("Uploader une image (optionnel)", type=["png", "jpg", "jpeg", "gif", "tiff", "bmp"], key="file_uploader")
    submitted = st.form_submit_button("Envoyer")

    if submitted:
        user_content = prompt if prompt else ""
        st.session_state.messages.append({"role": "user", "content": user_content + (f"\n[Image: {uploaded_file.name}]" if uploaded_file else "")})

        # --- Appel à l'API FastAPI ---
        with st.spinner("En attente de la réponse..."):
            try:
                files = None
                data = {"query": user_content, "session_id": st.session_state.session_id}
                headers = {}

                if uploaded_file:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                else:
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}

                response = requests.post(API_URL, files=files, data=data, headers=headers)

                if response.status_code == 200:
                    api_response = response.json()
                    bot_answer = api_response.get("answer", "Erreur: Clé 'answer' non trouvée dans la réponse de l'API.")
                    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
                else:
                    error_message = f"Erreur de l'API : Statut {response.status_code}"
                    try:
                        error_details = response.json()
                        error_message += f" - Détails : {error_details.get('detail', 'N/A')}"
                    except json.JSONDecodeError:
                        error_message += f" - Réponse non-JSON : {response.text[:100]}..."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": f"Erreur: {error_message}"})

            except requests.exceptions.ConnectionError:
                error_message = f"Erreur de connexion : Impossible de joindre l'API à {API_URL}. Assurez-vous que votre application FastAPI tourne."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Erreur: {error_message}"})

            except Exception as e:
                error_message = f"Une erreur inattendue s'est produite : {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": f"Erreur: {error_message}"})

# --- Affichage de l'historique des messages dans le conteneur ---
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])