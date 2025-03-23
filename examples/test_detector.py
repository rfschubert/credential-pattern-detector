#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para testar o detector de credenciais com exemplos de linha de comando.
"""

import sys
import argparse
from src.detector.credential_detector import CredentialDetector

def main():
    """
    Função principal para testar o detector de credenciais.
    """
    parser = argparse.ArgumentParser(description='Testar o detector de credenciais')
    parser.add_argument('texto', nargs='?', default=None, 
                        help='Texto para analisar em busca de credenciais')
    parser.add_argument('--model', default='models/credential_detector_model.pkl',
                        help='Caminho para o modelo treinado')
    parser.add_argument('--use-ml', action='store_true', default=True,
                        help='Usar modelo de ML para detecção')
    
    args = parser.parse_args()
    
    # Se não houver texto, usar exemplos predefinidos
    if args.texto is None:
        exemplos = [
            "Olá, como vai você?",
            "Minha senha é X#9pL@7!2ZqR e meu usuário é joao123",
            "API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            "Envie o relatório para joao@example.com até amanhã.",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "AKIA6QOKC2QOKC2EXAMPLE",
            "ID do cliente: 12345",
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAxzYk86/s0Q==\n-----END RSA PRIVATE KEY-----"
        ]
    else:
        exemplos = [args.texto]
    
    # Inicializar o detector
    try:
        print(f"Inicializando detector de credenciais...")
        if args.use_ml:
            detector = CredentialDetector(model_path=args.model)
            print(f"Detector inicializado com modelo ML de: {args.model}")
        else:
            detector = CredentialDetector(use_ml=False)
            print("Detector inicializado com regras apenas (sem ML)")
    except Exception as e:
        print(f"Erro ao inicializar o detector: {e}")
        return
    
    # Testar os exemplos
    for i, texto in enumerate(exemplos):
        print("\n" + "="*60)
        print(f"Exemplo {i+1}:")
        print(f"Texto: {texto}")
        
        try:
            resultado = detector.detect(texto)
            
            print(f"\nCONTÉM CREDENCIAL: {'SIM' if resultado.has_credential else 'NÃO'}")
            print(f"Confiança: {resultado.confidence:.2f}")
            
            if resultado.has_credential and resultado.matches:
                print("\nCredenciais encontradas:")
                for j, match in enumerate(resultado.matches):
                    print(f"  {j+1}. {match}")
                
                if resultado.match_positions:
                    print("\nPosições de correspondência:")
                    for start, end, match_type in resultado.match_positions:
                        print(f"  - Tipo: {match_type}, Posição: {start}:{end}, Texto: '{texto[start:end]}'")
            
        except Exception as e:
            print(f"Erro ao detectar credenciais: {e}")
        
        print("="*60)

if __name__ == "__main__":
    main() 