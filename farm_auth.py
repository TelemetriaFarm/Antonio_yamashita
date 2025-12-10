# Nome do arquivo: farm_auth.py
# (Versão segura para GitHub)

import requests
import os
import sys
from dotenv import load_dotenv # <-- Biblioteca para ler o arquivo .env

# Carrega as variáveis do arquivo .env (se existir)
# Isso permite rodar localmente sem expor a senha
load_dotenv()

LOGIN_URL = "https://admin.farmcommand.com/login/"

def get_authenticated_session() -> requests.Session | None:
    """
    Executa o login dinâmico no FarmCommand e retorna um 
    objeto requests.Session autenticado.
    
    Ele prioriza Variáveis de Ambiente (GitHub Secrets) ou
    lê de um arquivo .env (para rodar localmente).
    """
    
    # --- 1. Lê as credenciais do ambiente ---
    # (Elas vieram ou do .env local ou dos GitHub Secrets)
    USUARIO = os.environ.get("FARM_USER") 
    SENHA = os.environ.get("FARM_PASS")
    
    if not USUARIO or not SENHA:
        print("❌ ERRO DE AUTENTICAÇÃO: Credenciais não encontradas.")
        print("   Certifique-se de que você criou um arquivo .env")
        print("   ou configurou os GitHub Secrets (FARM_USER, FARM_PASS).")
        return None

    print("\n--- [farm_auth] Iniciando autenticação ---")
    
    s = requests.Session()
    try:
        # 2. Obter o CSRF Token inicial
        login_page = s.get(LOGIN_URL)
        login_page.raise_for_status() 
        
        if 'csrftoken' not in s.cookies:
            print("Erro: [farm_auth] 'csrftoken' não encontrado.")
            return None
        csrftoken_inicial = s.cookies['csrftoken']

        # 3. Montar os dados do formulário de login
        login_data = {
            'username': USUARIO,
            'password': SENHA,
            'csrfmiddlewaretoken': csrftoken_inicial
        }
        login_headers = {'Referer': LOGIN_URL}

        # 4. Enviar o POST de login
        r_login = s.post(LOGIN_URL, data=login_data, headers=login_headers)
        r_login.raise_for_status() 

        # 5. Verificação de Login
        if 'login' in r_login.url:
            print("ERRO: [farm_auth] Falha no login. Verifique as credenciais.")
            return None
        
        print("--- [farm_auth] Autenticação bem-sucedida ---")
        
        # 6. Adiciona o CSRF token aos headers padrão da sessão
        s.headers.update({
            'X-CSRFToken': s.cookies['csrftoken'],
            "accept": "application/json",
            "Content-Type": "application/json",
            "Referer": "https://admin.farmcommand.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        
        return s # Retorna a sessão autenticada

    except requests.exceptions.RequestException as e:
        print(f"ERRO: [farm_auth] Erro HTTP na autenticação: {e}")
        return None
