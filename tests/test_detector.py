import os
import sys
import unittest
from typing import List, Dict, Any

# Adicionar diretório raiz ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector.credential_detector import CredentialDetector, DetectionResult

class TestCredentialDetector(unittest.TestCase):
    """Testes para o detector de credenciais."""
    
    def setUp(self):
        """Configuração dos testes."""
        # Inicializar o detector apenas com regras (sem ML)
        self.detector = CredentialDetector(use_ml=False)
        
        # Exemplos para testes
        self.credential_examples = [
            "api_key=\"AbCdEfGhIjKlMnOpQrStUvWxYz123456\"",
            "password=Password123!",
            "AKIA1234567890ABCDEF",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "access_token=\"1234567890abcdefghijklmnopqrstuvwxyz1234\"",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "client_secret=1234567890abcdefghijklmnopqrstuvwxyz",
            "senha=\"P@ssw0rd123\"",
        ]
        
        self.non_credential_examples = [
            "Olá, como vai você?",
            "Este é um exemplo de texto normal.",
            "O valor do produto é 1500 reais.",
            "Meu email é usuario@example.com",
            "ID do cliente: CLIENT_12345",
            "Código do produto: PROD-9876",
        ]
    
    def test_detect_credentials(self):
        """Testa a detecção de credenciais."""
        for example in self.credential_examples:
            with self.subTest(example=example):
                result = self.detector.detect(example)
                self.assertTrue(
                    result.has_credential, 
                    f"Falha ao detectar credencial em: {example}"
                )
                self.assertGreaterEqual(
                    result.confidence, 
                    self.detector.confidence_threshold,
                    f"Confiança abaixo do limiar para: {example}"
                )
                self.assertGreater(
                    len(result.matches), 
                    0, 
                    f"Não foram encontradas correspondências em: {example}"
                )
    
    def test_detect_non_credentials(self):
        """Testa a não detecção em textos normais."""
        for example in self.non_credential_examples:
            with self.subTest(example=example):
                result = self.detector.detect(example)
                self.assertFalse(
                    result.has_credential, 
                    f"Detectou falsamente uma credencial em: {example}"
                )
    
    def test_empty_input(self):
        """Testa entrada vazia."""
        result = self.detector.detect("")
        self.assertFalse(result.has_credential)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.matches), 0)
    
    def test_detector_initialization(self):
        """Testa inicialização do detector."""
        # Testa padrões personalizados
        custom_patterns = {
            'test_pattern': r'test_secret=(\w+)'
        }
        
        detector = CredentialDetector(patterns=custom_patterns, use_ml=False)
        
        # Deve detectar padrão personalizado
        result = detector.detect("test_secret=abc123")
        self.assertTrue(result.has_credential)
        
        # Não deve detectar padrões padrão
        result = detector.detect("api_key=\"AbCdEfGhIjKlMnOpQrStUvWxYz123456\"")
        self.assertFalse(result.has_credential)
    
    def test_detection_result_initialization(self):
        """Testa inicialização do resultado da detecção."""
        # Inicialização sem listas
        result = DetectionResult(has_credential=True, confidence=0.9)
        self.assertEqual(result.matches, [])
        self.assertEqual(result.match_positions, [])
        
        # Inicialização com listas
        result = DetectionResult(
            has_credential=True, 
            confidence=0.9,
            matches=["abc123"],
            match_positions=[(0, 6, "test")]
        )
        self.assertEqual(result.matches, ["abc123"])
        self.assertEqual(result.match_positions, [(0, 6, "test")])


if __name__ == "__main__":
    unittest.main() 