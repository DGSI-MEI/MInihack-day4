import chromadb
import requests
from sentence_transformers import SentenceTransformer

# Configuración de DeepSeek API (¡Asegúrate de reemplazar con tu API key!)
DEEPSEEK_API_KEY = "sk-45273889dcbe407480bb5f35931d01f4"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# Conectar a ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="markdown_docs")

# Cargar modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def obtener_embedding(texto):
    """Convierte la consulta del usuario en un embedding."""
    return embedding_model.encode([texto])[0].tolist()

def buscar_en_chromadb(pregunta):
    """Busca en ChromaDB el chunk de texto más relevante."""
    embedding = obtener_embedding(pregunta)
    results = collection.query(
        query_embeddings=[embedding], 
        n_results=1  # Obtener solo el mejor resultado
    )

    if results["documents"] and results["documents"][0]:  
        return results["documents"][0][0]  # Devuelve el chunk más relevante
    else:
        return "No encontré información en la base de datos."

def enviar_a_deepseek(contexto, pregunta):
    """Envía el contexto a DeepSeek API y devuelve la respuesta del modelo."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": f"Contexto: {contexto}\nPregunta: {pregunta}"}
        ],
        "temperature": 0.7
    }
    
    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("❌ ERROR con la API:", response.status_code, response.text)  # Imprimir el error
        return "Hubo un error con la API de DeepSeek."

def chatbot():
    """Ejecuta el chatbot en la consola."""
    print("ChatBot: ¡Hola! Pregunta sobre los documentos Markdown. Escribe 'salir' para terminar.")

    while True:
        user_input = input("Tú: ").strip()
        if user_input.lower() == "salir":
            print("ChatBot: ¡Hasta luego!")
            break

        # 1️⃣ Buscar en ChromaDB
        contexto = buscar_en_chromadb(user_input)
        
        # 2️⃣ Enviar a DeepSeek para mejorar la respuesta
        respuesta = enviar_a_deepseek(contexto, user_input)
        
        print("ChatBot:", respuesta)

if __name__ == "__main__":
    chatbot()
