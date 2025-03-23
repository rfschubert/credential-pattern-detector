import os
import sys
import unittest
import tempfile
import json
import shutil
from typing import List, Dict, Any

# Adicionar diretório raiz ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Testes para o processador de dados."""
    
    def setUp(self):
        """Configuração dos testes."""
        # Criar diretório temporário para testes
        self.temp_dir = tempfile.mkdtemp()
        
        # Inicializar processador com diretório temporário
        self.data_processor = DataProcessor(data_dir=self.temp_dir)
        
        # Exemplos para testes
        self.credential_examples = [
            "api_key=\"AbCdEfGhIjKlMnOpQrStUvWxYz123456\"",
            "password=Password123!",
            "AKIA1234567890ABCDEF",
        ]
        
        self.non_credential_examples = [
            "Olá, como vai você?",
            "Este é um exemplo de texto normal.",
            "O valor do produto é 1500 reais.",
        ]
    
    def tearDown(self):
        """Limpeza após os testes."""
        # Remover diretório temporário
        shutil.rmtree(self.temp_dir)
    
    def test_add_credential_example(self):
        """Testa adição de exemplos de credenciais."""
        for example in self.credential_examples:
            self.data_processor.add_credential_example(example, source="test")
        
        # Verificar se os exemplos foram adicionados corretamente
        self.assertEqual(len(self.data_processor.credentials_data), len(self.credential_examples))
        
        # Verificar se o arquivo foi criado
        self.assertTrue(os.path.exists(self.data_processor.credentials_path))
        
        # Verificar conteúdo do arquivo
        with open(self.data_processor.credentials_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(self.credential_examples))
            
            for i, line in enumerate(lines):
                data = json.loads(line)
                self.assertEqual(data["text"], self.credential_examples[i])
                self.assertTrue(data["is_credential"])
                self.assertEqual(data["source"], "test")
    
    def test_add_non_credential_example(self):
        """Testa adição de exemplos que não são credenciais."""
        for example in self.non_credential_examples:
            self.data_processor.add_non_credential_example(example, source="test")
        
        # Verificar se os exemplos foram adicionados corretamente
        self.assertEqual(len(self.data_processor.non_credentials_data), len(self.non_credential_examples))
        
        # Verificar se o arquivo foi criado
        self.assertTrue(os.path.exists(self.data_processor.non_credentials_path))
        
        # Verificar conteúdo do arquivo
        with open(self.data_processor.non_credentials_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(self.non_credential_examples))
            
            for i, line in enumerate(lines):
                data = json.loads(line)
                self.assertEqual(data["text"], self.non_credential_examples[i])
                self.assertFalse(data["is_credential"])
                self.assertEqual(data["source"], "test")
    
    def test_load_data(self):
        """Testa carregamento de dados."""
        # Adicionar exemplos primeiro
        for example in self.credential_examples:
            self.data_processor.add_credential_example(example, source="test")
            
        for example in self.non_credential_examples:
            self.data_processor.add_non_credential_example(example, source="test")
        
        # Criar novo processador para carregar os dados
        new_processor = DataProcessor(data_dir=self.temp_dir)
        credentials, non_credentials = new_processor.load_data()
        
        # Verificar se os dados foram carregados corretamente
        self.assertEqual(len(credentials), len(self.credential_examples))
        self.assertEqual(len(non_credentials), len(self.non_credential_examples))
        
        for i, example in enumerate(credentials):
            self.assertEqual(example["text"], self.credential_examples[i])
            self.assertTrue(example["is_credential"])
            
        for i, example in enumerate(non_credentials):
            self.assertEqual(example["text"], self.non_credential_examples[i])
            self.assertFalse(example["is_credential"])
    
    def test_generate_train_test_split(self):
        """Testa a divisão entre treino e teste."""
        # Adicionar exemplos
        for example in self.credential_examples:
            self.data_processor.add_credential_example(example, source="test")
            
        for example in self.non_credential_examples:
            self.data_processor.add_non_credential_example(example, source="test")
        
        # Carregar dados
        self.data_processor.load_data()
        
        # Gerar divisão
        X_train, y_train, X_test, y_test = self.data_processor.generate_train_test_split()
        
        # Verificar tamanhos
        total_examples = len(self.credential_examples) + len(self.non_credential_examples)
        test_size = int(total_examples * self.data_processor.test_size)
        train_size = total_examples - test_size
        
        self.assertEqual(len(X_train), train_size)
        self.assertEqual(len(y_train), train_size)
        self.assertEqual(len(X_test), test_size)
        self.assertEqual(len(y_test), test_size)
        
        # Verificar se os dados são do tipo correto
        self.assertTrue(all(isinstance(x, str) for x in X_train))
        self.assertTrue(all(isinstance(y, int) for y in y_train))
        self.assertTrue(all(isinstance(x, str) for x in X_test))
        self.assertTrue(all(isinstance(y, int) for y in y_test))
    
    def test_extract_features_batch(self):
        """Testa extração de características em lote."""
        features = self.data_processor.extract_features_batch(self.credential_examples)
        
        # Verificar se as características foram extraídas
        self.assertEqual(len(features), len(self.credential_examples))
        
        # Verificar se as características são strings não vazias
        for feature in features:
            self.assertIsInstance(feature, str)
            self.assertGreater(len(feature), 0)
    
    def test_load_or_create_examples(self):
        """Testa carregamento ou criação de exemplos."""
        # Definir número mínimo de exemplos
        min_examples = 10
        
        # Adicionar apenas alguns exemplos (menos que o mínimo)
        for example in self.credential_examples:
            self.data_processor.add_credential_example(example, source="test")
            
        for example in self.non_credential_examples:
            self.data_processor.add_non_credential_example(example, source="test")
        
        # Limpar dados em memória para forçar carregamento
        self.data_processor.credentials_data = []
        self.data_processor.non_credentials_data = []
        
        # Carregar ou criar exemplos
        self.data_processor.load_or_create_examples(min_examples=min_examples)
        
        # Verificar se temos pelo menos o número mínimo de exemplos
        self.assertGreaterEqual(len(self.data_processor.credentials_data), min_examples)
        self.assertGreaterEqual(len(self.data_processor.non_credentials_data), min_examples)


if __name__ == "__main__":
    unittest.main() 