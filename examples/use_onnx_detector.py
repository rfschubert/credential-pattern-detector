#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemplo de uso do detector ONNX para detecção de credenciais.
Este script demonstra como carregar e usar um modelo ONNX exportado.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Adicionar diretório raiz ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.utils.onnx_loader import load_detector, ONNXCredentialDetector
except ImportError as e:
    print(f"Erro ao importar o detector ONNX: {e}")
    print("Certifique-se de que o pacote onnxruntime está instalado:")
    print("pip install onnxruntime")
    sys.exit(1)

def detect_credentials(
    text: str, 
    detector: ONNXCredentialDetector
) -> Dict[str, Any]:
    """
    Detecta credenciais em um texto usando o detector ONNX.
    
    Args:
        text: Texto a ser analisado
        detector: Detector ONNX carregado
        
    Returns:
        Resultado da detecção
    """
    result = detector.detect(text)
    return result

def format_result(result: Dict[str, Any], text: str) -> str:
    """
    Formata o resultado da detecção para exibição.
    
    Args:
        result: Resultado da detecção
        text: Texto original
        
    Returns:
        Texto formatado
    """
    has_credential = result.get("has_credential", False)
    confidence = result.get("confidence", 0.0)
    matches = result.get("matches", [])
    positions = result.get("match_positions", [])
    
    formatted = f"Texto analisado: {text}\n"
    formatted += f"Contém credencial: {'Sim' if has_credential else 'Não'}\n"
    formatted += f"Confiança: {confidence:.4f}\n"
    
    if matches:
        formatted += "Credenciais encontradas:\n"
        for idx, match in enumerate(matches):
            formatted += f"  {idx+1}. {match}\n"
    
    if positions:
        formatted += "Posições encontradas:\n"
        for start, end, match_type in positions:
            formatted += f"  - Posição {start}-{end} (Tipo: {match_type}): {text[start:end]}\n"
    
    return formatted

def main():
    """
    Função principal do exemplo.
    """
    parser = argparse.ArgumentParser(description="Detector de credenciais usando modelo ONNX")
    parser.add_argument(
        "--model-dir", 
        default="models/onnx",
        help="Diretório contendo o modelo ONNX"
    )
    parser.add_argument(
        "--model-name", 
        default="credential_detector",
        help="Nome base do modelo ONNX"
    )
    parser.add_argument(
        "--text", 
        default=None,
        help="Texto a ser analisado (se não fornecido, exemplos predefinidos serão usados)"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Retornar resultado em formato JSON"
    )
    
    args = parser.parse_args()
    
    # Exemplos para teste
    examples = [
        "Olá, como vai você? Tudo bem?",
        "Minha senha é X#9pL@7!2ZqR e meu usuário é joao123",
        "API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ",
        "AKIA6QOKC2QOKC2EXAMPLE",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAxzYk86/s0Q==\n-----END RSA PRIVATE KEY-----"
    ]
    
    try:
        # Carregar o detector
        detector = load_detector(
            model_directory=args.model_dir,
            model_name=args.model_name
        )
        
        # Processar texto fornecido ou exemplos
        if args.text:
            texts = [args.text]
        else:
            texts = examples
        
        results = []
        
        for text in texts:
            result = detect_credentials(text, detector)
            results.append(result)
            
            if not args.json:
                print("\n" + "="*50)
                print(format_result(result, text))
        
        if args.json:
            # Adicionar o texto original ao resultado
            for i, result in enumerate(results):
                result["text"] = texts[i]
            
            # Imprimir JSON
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        print(f"Erro ao executar o detector: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 