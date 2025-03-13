import chromadb
from sentence_transformers import SentenceTransformer

def query_chromadb(query_text, model, top_n=3):
    """ Realiza una búsqueda de similitud en ChromaDB y genera un prompt estructurado """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="markdown_docs")
    query_embedding = model.encode([query_text])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_n)

    # Obtener los primeros 3 chunks encontrados
    retrieved_chunks = results["documents"][0]
    
    # Construir el prompt estructurado
    context = "\n".join(retrieved_chunks)
    prompt = f"Contexto:\n{context}\n\nUserInput: {query_text}"
    
    return prompt

def interactive_chat():
    """ Inicia la CLI interactiva tipo chatbot con contexto en el prompt """
    print("🤖 Bot de consultas a ChromaDB con contexto")
    print("Escribe tu pregunta o 'salir' para terminar.\n")
    
    # Cargar el modelo de embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    while True:
        user_input = input("📝 Tú: ")
        
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("👋 Saliendo del chatbot. ¡Hasta luego!")
            break
        
        prompt = query_chromadb(user_input, model)
        print("\n📜 Prompt generado:")
        print(prompt)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    interactive_chat()
