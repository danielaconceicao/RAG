import requests

API_URL = "http://127.0.0.1:8000/chat"

# variável global para armazenar apenas o id da sessão
global_session_id = None 

def send_question(question: str):
    #envia a pergunta e o id da sessão para a api, e atualiza o id.
    global global_session_id 
    
    # O payload AGORA só inclui a pergunta e o ID
    payload = {
        "question": question,
        # vai ser none na primeira vez, e o id nas seguintes
        "session_id": global_session_id 
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        data = response.json()
        
        # atualiza o id da sessão com o que veio da resposta do servidor
        global_session_id = data.get("session_id")
        
        print("Bot:")
        print(data.get("answer"))
        print("-" * 50)
        return True

    except requests.exceptions.RequestException as e:
        print(f"ERRO: {e}")
        return False


def run_chat_cli():
    
    print("=" * 50)
    print("        🚀 RAG Chatbot CLI Iniciado 🚀")
    print("=" * 50)
    print("🤖 Bot: Olá! Como posso te ajudar hoje?") # <--- A MENSAGEM PRINCIPAL
    print("Digite 'sair' ou 'exit' a qualquer momento para terminar.")
    print("-" * 50)

    while True:
        try:
            user_input = input("user: ")

            if user_input.lower() in ['sair', 'exit']:
                #para tudo
                break

            if user_input.strip(): 
                send_question(user_input)
            
        except EOFError:
            #se der erro pare tudo
            break

if __name__ == "__main__":
    run_chat_cli()