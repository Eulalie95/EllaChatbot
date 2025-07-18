import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel 
from dotenv import load_dotenv 
from typing import Optional
import speech_recognition as sr
import tempfile 

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI 
from image_processing import extract_text_from_image

# --- Configuration 
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Le même que pour l'indexation
PROMPT_FILE_PATH = "./prompts/rag_prompt.txt"
NOVITA_BASE_URL = "https://api.novita.ai/v3/openai"

load_dotenv()

# Variables globales pour stocker les composants RAG chargés 
db = None
llm = None
qa_chain = None
CUSTOM_RAG_PROMPT = None

# Définir les modèles de données pour l'API (validation et documentation) 
class ChatRequest(BaseModel):
    """Modèle pour la requête de chat"""
    query: str # Le champ attendu dans la requête JSON (la question de l'utilisateur)

class ChatWithImageRequest(BaseModel):
    query: Optional[str] = None
    file: Optional[UploadFile] = None

class ChatResponse(BaseModel):
    """Modèle pour la réponse du chat"""
    answer: str # Le champ dans la réponse JSON (la réponse du chatbot)
    
# Initialisation de l'application FastAPI 
app = FastAPI()

# Configuration CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"], # Exemple: ["http://localhost:8000", "http://votre_frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fonction pour charger les modèles au démarrage 
# Cette fonction sera exécutée une seule fois lorsque l'application démarre
@app.on_event("startup")
async def load_models():
    # Dictionnaire pour stocker l'historique des sessions
    session_histories = {}

    def get_session_history(session_id: str):
        if session_id not in session_histories:
            session_histories[session_id] = InMemoryChatMessageHistory()
        return session_histories[session_id]

    global db, llm, qa_chain, CUSTOM_RAG_PROMPT # Utiliser les variables globales

    print("Chargement de la base de données ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    print("Base de données ChromaDB chargée.")

    print("Initialisation du modèle Novita AI (DeepSeek)...")
    NOVITA_MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b"
    NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")

    try:
        # <<< Initialisation du Chat >>>
        llm = ChatOpenAI(
            api_key=NOVITA_API_KEY,
            base_url=NOVITA_BASE_URL,
            model=NOVITA_MODEL_NAME,
            temperature=0.1, 
            # max_tokens=500 
        )
 

        print(f"Modèle Deepseek AI '{NOVITA_MODEL_NAME}' initialisé (via API).")

    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle Mistral AI : {e}")
        print("Veuillez vérifier votre NOVITA_API_KEY est définie dans le .env et votre connexion internet.")
        llm = None 
    print(f"Chargement du template de prompt depuis {PROMPT_FILE_PATH}...")
    try:
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            prompt_template_string = f.read()
        CUSTOM_RAG_PROMPT = PromptTemplate.from_template(prompt_template_string)
        print("Template de prompt chargé avec succès.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier de prompt est introuvable à {PROMPT_FILE_PATH}. Assurez-vous qu'il existe.")
        CUSTOM_RAG_PROMPT = None # S'assurer qu'il est None en cas d'erreur

    print("Création de la chaîne RAG RetrievalQA avec prompt personnalisé...")
    if CUSTOM_RAG_PROMPT is not None:
        raw_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 8}),
            chain_type_kwargs={"prompt": CUSTOM_RAG_PROMPT},
            # return_source_documents=True
        )

        qa_chain = RunnableWithMessageHistory(
            raw_qa_chain,
            get_session_history,
            input_messages_key="query",
            history_messages_key="chat_history"
        )
        print("Chaîne RAG prête.")
    else:
         print("ATTENTION: Impossible de charger le template de prompt. La chaîne RAG NE SERA PAS INITIALISÉE avec le prompt personnalisé.")
        
    print("\nAPI FastAPI démarrée et tous les modèles/composants chargés (vérifiez les erreurs ci-dessus) !")

# Endpoint HTTP pour le chat
# Cet endpoint répondra aux requêtes POST sur l'URL /chat
def reformulate_question(question):
    # Reformulations simples
    question = question.lower()
    if "payer" in question or "régler mes frais de vote" in question:
        question = question.replace("payer", "moyens de paiement")
        question = question.replace("régler", "moyens de paiement")
    return question

@app.post("/chat", response_model=ChatResponse) 
async def chat_endpoint(query: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    """
    Endpoint pour envoyer une question au chatbot et obtenir une réponse.
    Il route la question vers la chaîne RAG ou l'Agent SQL.
    Il reçoit des images et autres fichiers aussi.
    """
    print(f"Requête reçue : {query}")
    user_query = query if query else ""
    user_query = reformulate_question(user_query)
    print(f"Question réécrite (si applicable) : {user_query}")
    # Logique de routage des messages
    extracted_text = ""
    if file:
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/tiff", "image/bmp"]
        if file.content_type in allowed_types:
            try:
                image_content = await file.read()
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(image_content)
                    temp_file_path = tmp_file.name
                extracted_text = extract_text_from_image(temp_file_path)
                os.unlink(temp_file_path)
                if extracted_text:
                    print(f"Texte extrait de l'image : {extracted_text}")
                    user_query += f"\nContenu de l'image : {extracted_text}"
                else:
                    print("Aucun texte n'a été extrait de l'image.")
            except Exception as e:
                print(f"Erreur lors du traitement de l'image dans l'endpoint /chat : {e}")
                raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de l'image : {e}")
        else:
            raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas un format image supporté.")

    print(f"Requête reçue (combinée avec le texte de l'image) : {user_query}")
    
    keywords_for_sql = ["lien de vote", "lien du candidat"]

    answer = "Désolé, je n'ai pas pu générer de réponse pour votre question."

    if any(keyword in user_query.lower() for keyword in keywords_for_sql):
        print(f"Détection d'une question BDD: '{user_query}' -> Routage vers PandasAI.")
        try:
            response_from_pandasai = query_database_via_chatbot(user_query)
            answer = response_from_pandasai
        except Exception as e:
            print(f"Erreur lors de l'appel à PandasAI : {e}")
            answer = "Désolé, une erreur est survenue lors de l'accès à la base de données via PandasAI."

    else:
        print(f"Question non-BDD: '{user_query}' -> Routage vers la chaîne RAG.")
        if qa_chain is None:
             raise HTTPException(status_code=503, detail="Les modèles RAG ne sont pas encore chargés.")
        try:
            # Appeler la chaîne RAG avec la question de l'utilisateur
            session_id = "eulalie-session"
            response_dict = qa_chain.invoke({"query": user_query},
                                            config={"configurable": {"session_id": session_id}}) 
            answer = response_dict.get("result", "Erreur: Clé 'result' non trouvée dans la réponse de la chaîne RAG.")
        except Exception as e:
            print(f"Erreur lors du traitement de la requête RAG : {e}")
            raise HTTPException(status_code=500, detail=f"Une erreur interne est survenue lors de l'interrogation RAG : {e}")

    print(f"Réponse finale générée : {answer}")
    return ChatResponse(answer=answer) 

@app.post("/transcribe_audio", response_model=ChatResponse)
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    try:
        if audio_file.content_type.startswith("audio/"):
            audio_content = await audio_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_content)
                audio_file_path = tmp_file.name

            r = sr.Recognizer()
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio = r.record(source)
                transcribed_text = r.recognize_google(audio, language="fr-FR")
                os.unlink(audio_file_path)
                return {"answer": transcribed_text}
            except sr.UnknownValueError:
                os.unlink(audio_file_path)
                raise HTTPException(status_code=400, detail="Impossible de comprendre l'audio.")
            except sr.RequestError as e:
                os.unlink(audio_file_path)
                raise HTTPException(status_code=500, detail=f"Erreur service de reconnaissance vocale: {e}")
            except sr.Unsupportedsampletype:
                os.unlink(audio_file_path)
                raise HTTPException(status_code=400, detail="Format audio non supporté. Essayez WAV.")
        else:
            raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas un fichier audio.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement audio: {e}")


# Endpoint racine
@app.get("/")
async def read_root():
    return {"message": "API Chatbot hybride (RAG + BDD + Image Text Handling) opérationnelle. Utilisez l'endpoint /chat pour le texte et les images."}
