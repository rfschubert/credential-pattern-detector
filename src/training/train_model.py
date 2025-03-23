#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de treinamento para o detector de padrões de credenciais.
Treina um modelo de classificação para identificar se um texto contém credenciais.
Também exporta automaticamente o modelo para formato ONNX.
"""

import os
import sys
import logging
import yaml
import pickle
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Adicionar diretório raiz ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.data_processor import DataProcessor
from src.utils.text_utils import is_binary_text

# Verificar se temos o módulo de exportação
try:
    from src.export.export_onnx import export_model as export_to_onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("Módulo ONNX não disponível. O modelo não será exportado para ONNX.")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def extract_features(text: str) -> Dict[str, Any]:
    """
    Extrai características do texto para uso no treinamento.
    
    Args:
        text: Texto a ser analisado
        
    Returns:
        Dicionário de características extraídas do texto
    """
    features = {
        "length": len(text),
        "has_special_chars": any(c in "!@#$%^&*()_+-=[]{}|;:'\"<>,.?/~`" for c in text),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "digit_ratio": sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        "is_binary": is_binary_text(text),
        "text": text  # Mantendo o texto original para uso na vetorização
    }
    return features

def load_config(config_path: str = "config/training_config.yaml") -> Dict[str, Any]:
    """
    Carrega a configuração de treinamento do arquivo YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com as configurações de treinamento
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Erro ao carregar configuração: {e}")
        # Configurações padrão
        return {
            "model_type": "random_forest",
            "rf_n_estimators": 200,
            "rf_max_depth": 20,
            "rf_min_samples_split": 2,
            "rf_min_samples_leaf": 1,
            "vectorizer_max_features": 1500,
            "vectorizer_min_df": 2,
            "vectorizer_max_df": 0.95,
            "vectorizer_ngram_max": 3,
            "use_grid_search": False
        }

def train_model(config: Dict[str, Any], train_dir: str = "data/processed", test_size: float = 0.2) -> Dict[str, Any]:
    """
    Treina o modelo usando os dados fornecidos.
    
    Args:
        config: Configurações de treinamento
        train_dir: Diretório contendo os dados de treinamento
        test_size: Proporção dos dados a serem usados para teste
        
    Returns:
        Dicionário contendo o modelo treinado e métricas
    """
    # Criar processador de dados
    data_processor = DataProcessor(data_dir=train_dir, test_size=test_size)
    
    # Carregar dados de treinamento
    data_processor.load_data()
    
    # Gerar conjuntos de dados balanceados
    X_train, y_train, X_test, y_test = data_processor.generate_train_test_split()
    
    # Extrair características
    logger.info("Extraindo características dos dados de treinamento...")
    X_train_features = []
    
    for text in tqdm(X_train, desc="Extraindo características de treino"):
        features = extract_features(text)
        X_train_features.append(features)
    
    logger.info("Extraindo características dos dados de teste...")
    X_test_features = []
    
    for text in tqdm(X_test, desc="Extraindo características de teste"):
        features = extract_features(text)
        X_test_features.append(features)
    
    # Vetorização de texto
    logger.info("Treinando o vectorizer TF-IDF...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=config["vectorizer_max_features"],
        min_df=config["vectorizer_min_df"],
        max_df=config["vectorizer_max_df"],
        ngram_range=(1, config["vectorizer_ngram_max"]),
        strip_accents="unicode",
        analyzer="char_wb"
    )
    
    X_train_text = [f["text"] for f in X_train_features]
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    
    X_test_text = [f["text"] for f in X_test_features]
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Treinamento do modelo
    logger.info("Treinando o modelo de classificação...")
    if config["model_type"] == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            min_samples_split=config["rf_min_samples_split"],
            min_samples_leaf=config["rf_min_samples_leaf"],
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Tipo de modelo não suportado: {config['model_type']}")
    
    # Treinamento
    model.fit(X_train_tfidf, y_train)
    
    # Avaliação
    logger.info("Avaliando o modelo...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = (y_pred == y_test).mean()
    logger.info(f"Acurácia: {accuracy}")
    
    classification_rep = classification_report(
        y_test, 
        y_pred, 
        target_names=["Não Credencial", "Credencial"]
    )
    logger.info(f"Relatório de classificação:\n{classification_rep}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info(f"Matriz de confusão:\n{conf_matrix}")
    
    # Salvar relatório de treinamento
    report_path = os.path.join("models", "training_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Acurácia: {accuracy}\n\n")
        f.write(f"Relatório de classificação:\n{classification_rep}\n\n")
        f.write(f"Matriz de confusão:\n{conf_matrix}\n\n")
        f.write("Configuração de treinamento:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Retornar o modelo treinado e métricas
    return {
        "classifier": model,
        "vectorizer": vectorizer,
        "feature_extractor": extract_features,
        "config": config,
        "confidence_threshold": 0.7,
        "accuracy": accuracy,
        "classification_report": classification_rep,
        "confusion_matrix": conf_matrix
    }

def save_model(model_data: Dict[str, Any], output_path: str = "models/credential_detector_model.pkl") -> str:
    """
    Salva o modelo treinado em um arquivo.
    
    Args:
        model_data: Dicionário contendo o modelo treinado e métricas
        output_path: Caminho para salvar o modelo
        
    Returns:
        Caminho do arquivo salvo
    """
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar o modelo
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Modelo salvo em {output_path}")
    return output_path

def main() -> None:
    """
    Função principal para treinamento do modelo.
    """
    # Carregar configuração
    config_path = "config/training_config.yaml"
    config = load_config(config_path)
    
    logger.info(f"Configuração de treinamento: {config}")
    
    # Treinar o modelo
    model_data = train_model(config)
    
    # Salvar o modelo
    model_path = save_model(model_data)
    
    # Exportar para ONNX se disponível
    if ONNX_AVAILABLE:
        logger.info("Exportando modelo para ONNX...")
        onnx_output_dir = "models/onnx"
        os.makedirs(onnx_output_dir, exist_ok=True)
        
        export_to_onnx(
            model_path=model_path,
            output_dir=onnx_output_dir
        )
        logger.info(f"Modelo exportado para ONNX em {onnx_output_dir}")
    else:
        logger.warning("Exportação para ONNX não disponível. Instale os pacotes necessários.")

if __name__ == "__main__":
    main() 