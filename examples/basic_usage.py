#!/usr/bin/env python
"""
Exemplo básico de uso do detector de credenciais.
"""

import os
import sys
import logging

# Adicionar diretório raiz ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from credential_detector import CredentialDetector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Função principal para demonstrar o uso do detector."""
    # Inicializar o detector
    # Observe que, se o modelo não estiver disponível, ele usará apenas regras
    detector = CredentialDetector()
    
    # Exemplos de texto para testar
    exemplos = [
        "Olá, como vai você?",
        "Minha senha é X#9pL@7!2ZqR e meu usuário é joao123",
        "API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
        "Envie o relatório para joao@example.com até amanhã.",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "AKIA6QOKC2QOKC2EXAMPLE",
        "ID do cliente: 12345"
    ]
    
    # Testar cada exemplo
    for i, texto in enumerate(exemplos):
        resultado = detector.detect(texto)
        
        print(f"\nExemplo {i+1}:")
        print(f"Texto: {texto}")
        print(f"Contém credencial: {'Sim' if resultado.has_credential else 'Não'}")
        print(f"Confiança: {resultado.confidence:.2f}")
        
        if resultado.has_credential and resultado.matches:
            print("Credenciais encontradas:")
            for match in resultado.matches:
                print(f"  - {match}")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 