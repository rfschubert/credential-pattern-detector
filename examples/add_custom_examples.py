#!/usr/bin/env python
"""
Exemplo de como adicionar exemplos personalizados para treinamento.
"""

import os
import sys
import logging

# Adicionar diretório raiz ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.data_processor import DataProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Função principal para demonstrar como adicionar exemplos personalizados."""
    # Inicializar o processador de dados
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    processor = DataProcessor(data_dir=data_dir)
    
    # Carregar exemplos existentes
    credentials, non_credentials = processor.load_data()
    
    logger.info(f"Exemplos existentes: {len(credentials)} credenciais, {len(non_credentials)} não-credenciais")
    
    # Adicionar exemplos de credenciais
    print("\nAdicione exemplos de credenciais (um por linha, linha vazia para terminar):")
    while True:
        example = input("> ").strip()
        if not example:
            break
        processor.add_credential_example(example, source="user_input")
        print("Exemplo adicionado como credencial.")
    
    # Adicionar exemplos de não-credenciais
    print("\nAdicione exemplos de não-credenciais (um por linha, linha vazia para terminar):")
    while True:
        example = input("> ").strip()
        if not example:
            break
        processor.add_non_credential_example(example, source="user_input")
        print("Exemplo adicionado como não-credencial.")
    
    # Recarregar para verificar
    credentials, non_credentials = processor.load_data()
    logger.info(f"Total de exemplos: {len(credentials)} credenciais, {len(non_credentials)} não-credenciais")
    
    # Explicar próximo passo
    print("\nExemplos adicionados com sucesso!")
    print("Para treinar o modelo com esses exemplos, execute:")
    print("python -m src.training.train_model")

if __name__ == "__main__":
    main() 