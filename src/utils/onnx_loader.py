#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitários para carregamento e uso de modelos ONNX.
Este módulo facilita o uso de modelos ONNX em diferentes ambientes.
"""

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import regex as re
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime não disponível. Instale com 'pip install onnxruntime'")

logger = logging.getLogger(__name__)

class ONNXCredentialDetector:
    """
    Detector de credenciais usando modelo ONNX.
    Permite usar o modelo CredentialDetector exportado para ONNX.
    """
    
    def __init__(
        self, 
        model_path: str, 
        config_path: Optional[str] = None,
        vectorizer_path: Optional[str] = None,
        patterns_path: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Inicializa o detector de credenciais com um modelo ONNX.
        
        Args:
            model_path: Caminho para o modelo ONNX
            config_path: Caminho para a configuração do modelo (opcional)
            vectorizer_path: Caminho para os dados do vectorizer (opcional)
            patterns_path: Caminho para os padrões regex (opcional)
            confidence_threshold: Limiar de confiança para classificação
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime não está disponível. Instale com 'pip install onnxruntime'")
        
        # Carregar o modelo ONNX
        self.session = ort.InferenceSession(model_path)
        
        # Definir valores padrão
        self.config = {}
        self.vectorizer_data = {}
        self.patterns = {}
        self.confidence_threshold = confidence_threshold
        
        # Se o caminho da configuração não foi fornecido, tentar inferir
        if config_path is None:
            config_path = model_path.replace(".onnx", "_config.json")
        
        # Carregar configuração
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
                if "confidence_threshold" in self.config:
                    self.confidence_threshold = self.config["confidence_threshold"]
        
        # Se o caminho do vectorizer não foi fornecido, tentar inferir
        if vectorizer_path is None:
            vectorizer_path = model_path.replace(".onnx", "_vectorizer.json")
        
        # Carregar dados do vectorizer
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, "r", encoding="utf-8") as f:
                self.vectorizer_data = json.load(f)
        
        # Se o caminho dos padrões não foi fornecido, tentar inferir
        if patterns_path is None:
            patterns_path = model_path.replace(".onnx", "_patterns.json")
        
        # Carregar padrões
        if os.path.exists(patterns_path):
            with open(patterns_path, "r", encoding="utf-8") as f:
                patterns_data = json.load(f)
                
                # Compilar padrões regex
                self.patterns = {}
                for pattern_name, pattern_info in patterns_data.items():
                    if isinstance(pattern_info, dict) and "pattern" in pattern_info:
                        flags = pattern_info.get("flags", 0)
                        self.patterns[pattern_name] = re.compile(pattern_info["pattern"], flags)
                    elif isinstance(pattern_info, str):
                        self.patterns[pattern_name] = re.compile(pattern_info)
    
    def vectorize_text(self, text: str) -> np.ndarray:
        """
        Vetoriza o texto usando o vocabulário do TF-IDF.
        
        Args:
            text: Texto a ser vetorizado
            
        Returns:
            Array NumPy com o vetor TF-IDF
        """
        if not self.vectorizer_data or "vocabulary" not in self.vectorizer_data:
            raise ValueError("Dados do vectorizer não foram carregados corretamente")
        
        vocabulary = self.vectorizer_data["vocabulary"]
        
        # Criar vetor esparso
        vector = np.zeros(len(vocabulary))
        
        # Implementação simplificada de vetorização TF-IDF
        # (Uma implementação completa replicaria exatamente o comportamento do TfidfVectorizer)
        ngram_min, ngram_max = self.vectorizer_data.get("ngram_range", (1, 1))
        
        # Considerar analyzer char_wb (word-boundary) por padrão
        text_tokens = []
        for n in range(ngram_min, ngram_max + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                text_tokens.append(ngram)
        
        # Contar frequência dos tokens
        for token in text_tokens:
            if token in vocabulary:
                idx = vocabulary[token]
                vector[idx] += 1
        
        # Normalizar (simplificado)
        if np.sum(vector) > 0:
            vector = vector / np.sqrt(np.sum(vector**2))
        
        # Aplicar IDF (se disponível)
        if "idf" in self.vectorizer_data and self.vectorizer_data["idf"]:
            idf = np.array(self.vectorizer_data["idf"])
            vector = vector * idf
        
        return vector
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extrai características do texto para uso na detecção.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Características extraídas do texto
        """
        # Implementação simplificada - na versão completa, deve replicar extract_features do modelo original
        return {"text": text}
    
    def detect_with_regex(self, text: str) -> Tuple[List[str], List[Tuple[int, int, str]]]:
        """
        Detecta credenciais usando padrões regex.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Tupla contendo a lista de credenciais encontradas e suas posições
        """
        matches = []
        positions = []
        
        if not self.patterns:
            return matches, positions
        
        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                credential = text[start:end]
                matches.append(credential)
                positions.append((start, end, pattern_name))
        
        return matches, positions
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detecta se o texto contém credenciais.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Resultado da detecção com confiança e matches
        """
        # Verificar primeiro com regex para casos simples
        matches, positions = self.detect_with_regex(text)
        
        if matches:
            return {
                "has_credential": True,
                "confidence": 1.0,
                "matches": matches,
                "match_positions": positions
            }
        
        # Extrair características e vetorizar
        features = self.extract_features(text)
        vector = self.vectorize_text(text)
        
        # Preparar para inferência
        input_name = self.session.get_inputs()[0].name
        vector_reshaped = vector.reshape(1, -1).astype(np.float32)
        
        # Fazer a inferência
        outputs = self.session.run(None, {input_name: vector_reshaped})
        
        # Interpretar resultados
        # O primeiro output geralmente é a classe ou probabilidade
        probabilities = outputs[0][0]
        
        # Classe 1 geralmente é a classe positiva (contém credencial)
        confidence = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        has_credential = confidence >= self.confidence_threshold
        
        return {
            "has_credential": has_credential,
            "confidence": float(confidence),
            "matches": [text] if has_credential else [],
            "match_positions": []
        }

def load_detector(model_directory: str, model_name: str = "credential_detector") -> ONNXCredentialDetector:
    """
    Carrega um detector de credenciais ONNX a partir de um diretório.
    
    Args:
        model_directory: Diretório contendo os arquivos do modelo
        model_name: Nome base do modelo
        
    Returns:
        Detector de credenciais ONNX carregado
    """
    model_path = os.path.join(model_directory, f"{model_name}.onnx")
    config_path = os.path.join(model_directory, f"{model_name}_config.json")
    vectorizer_path = os.path.join(model_directory, f"{model_name}_vectorizer.json")
    patterns_path = os.path.join(model_directory, f"{model_name}_patterns.json")
    
    return ONNXCredentialDetector(
        model_path=model_path,
        config_path=config_path,
        vectorizer_path=vectorizer_path,
        patterns_path=patterns_path
    ) 