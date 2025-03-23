import re
import os
import pickle
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Pattern, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.text_utils import extract_features

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Resultado da detecção de credenciais."""
    
    has_credential: bool
    confidence: float
    matches: List[str] = None
    match_positions: List[tuple] = None
    
    def __post_init__(self):
        if self.matches is None:
            self.matches = []
        if self.match_positions is None:
            self.match_positions = []


class CredentialDetector:
    """
    Detector de padrões de credenciais em texto.
    
    Esta classe utiliza uma combinação de regras baseadas em expressões regulares
    e um modelo de machine learning para identificar credenciais em texto.
    """
    
    # Padrões comuns de credenciais (regex)
    DEFAULT_PATTERNS = {
        'api_key': r'(?i)(?:api[_-]?key|apikey)[\'"]?\s*(?::|=|=>)\s*[\'"]?([a-zA-Z0-9]{16,64})[\'"]?',
        'aws_key': r'(?:AKIA|A3T|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}',
        'password': r'(?i)(?:password|senha|pwd)[\'"]?\s*(?::|=|=>)\s*[\'"]?([^\s]{8,64})[\'"]?',
        'private_key': r'-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
        'auth_token': r'(?i)(?:authorization|authentication|auth[_-]?token|bearer)[\'"]?\s*(?::|=|=>)\s*[\'"]?([^\s]{8,})[\'"]?',
        'jwt': r'eyJ[a-zA-Z0-9_-]{5,}\.eyJ[a-zA-Z0-9_-]{5,}\.[a-zA-Z0-9_-]{5,}',
        'secret': r'(?i)(?:secret|secretkey|client[_-]secret)[\'"]?\s*(?::|=|=>)\s*[\'"]?([^\s]{8,})[\'"]?',
    }
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        patterns: Optional[Dict[str, str]] = None,
        confidence_threshold: float = 0.7,
        use_ml: bool = True
    ):
        """
        Inicializa o detector de credenciais.
        
        Args:
            model_path: Caminho para o modelo treinado.
            patterns: Dicionário de padrões regex para identificar credenciais.
            confidence_threshold: Limiar de confiança para considerar uma detecção positiva.
            use_ml: Se deve usar o modelo de machine learning ou apenas regras.
        """
        self.confidence_threshold = confidence_threshold
        self.use_ml = use_ml
        self.model = None
        self.vectorizer = None
        
        # Usar padrões fornecidos ou padrões padrão
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.compiled_patterns = {
            name: re.compile(pattern) for name, pattern in self.patterns.items()
        }
        
        # Carregar modelo se o caminho for fornecido e use_ml for True
        if model_path and use_ml:
            self._load_model(model_path)
        elif use_ml:
            # Usar modelo padrão se não for fornecido um caminho e use_ml for True
            default_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models",
                "credential_detector_model.pkl"
            )
            if os.path.exists(default_model_path):
                self._load_model(default_model_path)
            else:
                logger.warning(
                    f"Modelo não encontrado em {default_model_path}. "
                    "Usando apenas detecção baseada em regras."
                )
                self.use_ml = False
    
    def _load_model(self, model_path: str) -> None:
        """
        Carrega o modelo treinado a partir do caminho fornecido.
        
        Args:
            model_path: Caminho para o arquivo do modelo.
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data.get('model')
            self.vectorizer = model_data.get('vectorizer')
            
            if not self.model or not self.vectorizer:
                logger.error("Formato de modelo inválido")
                self.use_ml = False
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            self.use_ml = False
    
    def _find_pattern_matches(self, text: str) -> tuple:
        """
        Encontra correspondências usando expressões regulares.
        
        Args:
            text: Texto a ser analisado.
            
        Returns:
            Tupla com correspondências encontradas e suas posições.
        """
        matches = []
        positions = []
        
        for pattern_name, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                # Se o padrão tem grupo de captura, use o primeiro grupo
                if match.groups():
                    start, end = match.span(1)
                    match_value = match.group(1)
                else:
                    start, end = match.span()
                    match_value = match.group()
                
                matches.append(match_value)
                positions.append((start, end, pattern_name))
        
        return matches, positions
    
    def _predict_with_ml(self, text: str) -> float:
        """
        Faz predição usando o modelo de machine learning.
        
        Args:
            text: Texto a ser analisado.
            
        Returns:
            Probabilidade estimada de o texto conter uma credencial.
        """
        if not self.model or not self.vectorizer:
            return 0.0
        
        # Extrair características do texto
        features = extract_features(text)
        
        # Transformar características em vetor usando o vectorizador treinado
        X = self.vectorizer.transform([features])
        
        # Obter a probabilidade da classe positiva (contém credencial)
        prob = self.model.predict_proba(X)[0, 1]
        
        return float(prob)
    
    def detect(self, text: str) -> DetectionResult:
        """
        Detecta credenciais no texto fornecido.
        
        Args:
            text: Texto a ser analisado.
            
        Returns:
            Objeto DetectionResult com o resultado da detecção.
        """
        if not text:
            return DetectionResult(has_credential=False, confidence=0.0)
        
        # Encontrar correspondências de padrões regex
        matches, positions = self._find_pattern_matches(text)
        
        # Inicializar confiança
        confidence = 0.0
        
        # Se houver correspondências de padrões, calcular confiança
        if matches:
            confidence = 0.9  # Alta confiança para correspondências de regex
        
        # Se estiver usando ML, ajustar confiança
        if self.use_ml:
            ml_confidence = self._predict_with_ml(text)
            
            # Combinar confiança de regras e ML (dando mais peso para regras)
            if matches:
                # Se há correspondências de regex, manter alta confiança
                confidence = max(confidence, ml_confidence)
            else:
                # Se não há correspondências de regex, usar ML com cautela
                confidence = ml_confidence * 0.8
        
        # Determinar se há credencial com base no limiar de confiança
        has_credential = confidence >= self.confidence_threshold
        
        return DetectionResult(
            has_credential=has_credential,
            confidence=confidence,
            matches=matches,
            match_positions=positions
        ) 