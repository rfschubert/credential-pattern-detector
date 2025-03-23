#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testes para a exportação de modelos para ONNX.
"""

import os
import sys
import shutil
import tempfile
import unittest
import pickle
from unittest import mock

# Garantir que os módulos do projeto estão no path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

try:
    import onnx
    import onnxruntime as ort
    has_onnx = True
except ImportError:
    has_onnx = False
    
# Skip decorator para pular testes quando ONNX não está disponível
skip_if_no_onnx = unittest.skipIf(not has_onnx, "ONNX não está disponível")

class TestONNXExport(unittest.TestCase):
    """Testes para exportação ONNX."""
    
    def setUp(self):
        """Configuração para os testes."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Criar um diretório temporário para os testes
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pkl")
        self.output_dir = os.path.join(self.test_dir, "onnx")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Criar um modelo simples para teste
        vectorizer = TfidfVectorizer(max_features=10)
        X = ["teste1", "teste2", "credencial1", "credencial2"]
        vectorizer.fit(X)
        
        X_vec = vectorizer.transform(X)
        y = [0, 0, 1, 1]  # 0 = não credencial, 1 = credencial
        
        # Treinar um classificador
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_vec, y)
        
        # Salvar o modelo
        model_data = {
            "classifier": clf,
            "vectorizer": vectorizer,
            "feature_extractor": lambda x: {"text": x},
            "config": {"test": True},
            "confidence_threshold": 0.7,
            "patterns": {"api_key": r"API_KEY=\w+"}
        }
        
        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)
    
    def tearDown(self):
        """Limpeza após os testes."""
        # Remover diretório temporário
        shutil.rmtree(self.test_dir)
    
    @skip_if_no_onnx
    def test_convert_sklearn_model_to_onnx(self):
        """Testa a conversão de um modelo scikit-learn para ONNX."""
        from src.utils.onnx_utils import convert_sklearn_model_to_onnx
        
        # Carregar o modelo
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        classifier = model_data["classifier"]
        
        # Converter o modelo
        onnx_path = os.path.join(self.output_dir, "test_model.onnx")
        onnx_bytes = convert_sklearn_model_to_onnx(
            model=classifier,
            model_name="test_model",
            input_dim=10,  # Dimensão do vectorizer
            output_path=onnx_path
        )
        
        # Verificar se o arquivo foi criado
        self.assertTrue(os.path.exists(onnx_path))
        
        # Verificar se o modelo é válido
        onnx_model = onnx.load(onnx_path)
        self.assertIsNotNone(onnx_model)
    
    @skip_if_no_onnx
    def test_convert_vectorizer_to_onnx(self):
        """Testa a exportação dos dados do vectorizer."""
        from src.utils.onnx_utils import convert_vectorizer_to_onnx
        
        # Carregar o modelo
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        vectorizer = model_data["vectorizer"]
        
        # Converter o vectorizer
        vectorizer_path = os.path.join(self.output_dir, "test_vectorizer.json")
        vectorizer_data, output_path = convert_vectorizer_to_onnx(
            vectorizer=vectorizer,
            model_name="test_vectorizer",
            output_path=vectorizer_path
        )
        
        # Verificar se o arquivo foi criado
        self.assertTrue(os.path.exists(vectorizer_path))
        
        # Verificar se os dados são válidos
        self.assertIn("vocabulary", vectorizer_data)
        self.assertIn("idf", vectorizer_data)
    
    @skip_if_no_onnx
    def test_convert_credential_detector_to_onnx(self):
        """Testa a conversão completa do detector de credenciais para ONNX."""
        from src.utils.onnx_utils import convert_credential_detector_to_onnx
        
        # Converter o modelo
        export_paths = convert_credential_detector_to_onnx(
            model_path=self.model_path,
            output_dir=self.output_dir,
            model_name="test_detector"
        )
        
        # Verificar se os arquivos foram criados
        self.assertTrue(os.path.exists(export_paths["onnx_model"]))
        self.assertTrue(os.path.exists(export_paths["vectorizer"]))
        self.assertTrue(os.path.exists(export_paths["config"]))
    
    @skip_if_no_onnx
    def test_export_onnx_script(self):
        """Testa o script de exportação completo."""
        # Importar apenas se ONNX estiver disponível
        from src.export.export_onnx import export_model
        
        # Exportar o modelo
        test_examples = ["teste", "API_KEY=12345"]
        export_results = export_model(
            model_path=self.model_path,
            output_dir=self.output_dir,
            model_name="test_export",
            test_examples=test_examples
        )
        
        # Verificar se os arquivos foram criados
        self.assertTrue(os.path.exists(export_results["onnx_model"]))
        self.assertTrue(os.path.exists(export_results["vectorizer"]))
        self.assertTrue(os.path.exists(export_results["config"]))
        self.assertTrue(os.path.exists(export_results["patterns"]))
        
        # Verificar se os exemplos de implementação foram gerados
        php_example = os.path.join(self.output_dir, "test_export_php_example.php")
        js_example = os.path.join(self.output_dir, "test_export_js_example.js")
        self.assertTrue(os.path.exists(php_example))
        self.assertTrue(os.path.exists(js_example))
    
    @skip_if_no_onnx
    def test_onnx_loader(self):
        """Testa o carregador de modelos ONNX."""
        # Primeiro exportar o modelo
        from src.export.export_onnx import export_model
        export_model(
            model_path=self.model_path,
            output_dir=self.output_dir,
            model_name="test_loader"
        )
        
        # Depois carregar o modelo exportado
        from src.utils.onnx_loader import load_detector
        
        detector = load_detector(
            model_directory=self.output_dir,
            model_name="test_loader"
        )
        
        # Testar detecção
        result = detector.detect("API_KEY=12345")
        self.assertTrue(result["has_credential"])
        
        result = detector.detect("texto normal sem credenciais")
        self.assertFalse(result["has_credential"])

if __name__ == "__main__":
    unittest.main() 