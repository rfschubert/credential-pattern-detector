#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilitários para converter modelos do projeto para o formato ONNX.
ONNX (Open Neural Network Exchange) é um formato aberto para modelos de machine learning
que permite compartilhar modelos entre diferentes frameworks.
"""

import os
import pickle
import logging
import json
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logger = logging.getLogger(__name__)

# Classe auxiliar para serializar objetos NumPy
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_sklearn_model_to_onnx(
    model: Any,
    model_name: str,
    input_dim: int,
    output_path: Optional[str] = None
) -> bytes:
    """
    Converte um modelo scikit-learn para o formato ONNX.
    
    Args:
        model: O modelo scikit-learn a ser convertido
        model_name: Nome do modelo
        input_dim: Dimensão dos dados de entrada (número de features)
        output_path: Caminho para salvar o modelo convertido. Se None, apenas retorna os bytes.
        
    Returns:
        Bytes do modelo ONNX.
    """
    # Definir o tipo de entrada
    initial_type = [('input', FloatTensorType([None, input_dim]))]
    
    # Converter o modelo para ONNX
    onnx_model = convert_sklearn(model, model_name, initial_type, target_opset=15)
    
    # Verificar o modelo ONNX
    onnx.checker.check_model(onnx_model)
    
    # Serializar para bytes
    onnx_bytes = onnx_model.SerializeToString()
    
    # Salvar o modelo se um caminho foi fornecido
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_bytes)
        logger.info(f"Modelo ONNX salvo em: {output_path}")
    
    return onnx_bytes

def convert_vectorizer_to_onnx(
    vectorizer: Any, 
    model_name: str,
    output_path: Optional[str] = None
) -> Tuple[Dict, str]:
    """
    Como os vectorizers TF-IDF não são diretamente convertíveis para ONNX,
    extraímos seus parâmetros e vocabulário para reimplementação em outras linguagens.
    
    Args:
        vectorizer: O vectorizer TF-IDF do scikit-learn
        model_name: Nome do modelo
        output_path: Caminho para salvar os dados do vectorizer em formato JSON
        
    Returns:
        Dicionário com os parâmetros do vectorizer e caminho para o arquivo salvo
    """
    # Extrair parâmetros do vectorizer
    vocab_dict = {}
    for word, idx in vectorizer.vocabulary_.items():
        # Converter chaves para strings se forem bytes
        word_key = word.decode('utf-8') if isinstance(word, bytes) else word
        vocab_dict[word_key] = int(idx)
    
    vectorizer_data = {
        "vocabulary": vocab_dict,
        "idf": vectorizer.idf_.tolist() if hasattr(vectorizer, 'idf_') else None,
        "stop_words": list(vectorizer.stop_words_) if hasattr(vectorizer, 'stop_words_') and vectorizer.stop_words_ else None,
        "ngram_range": vectorizer.ngram_range,
        "norm": vectorizer.norm,
        "max_df": float(vectorizer.max_df) if isinstance(vectorizer.max_df, (np.floating, np.integer)) else vectorizer.max_df,
        "min_df": float(vectorizer.min_df) if isinstance(vectorizer.min_df, (np.floating, np.integer)) else vectorizer.min_df,
        "max_features": int(vectorizer.max_features) if vectorizer.max_features is not None else None,
        "model_name": model_name
    }
    
    # Salvar como JSON se um caminho foi fornecido
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vectorizer_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        logger.info(f"Dados do vectorizer salvos em: {output_path}")
    
    return vectorizer_data, output_path

def convert_credential_detector_to_onnx(
    model_path: str,
    output_dir: str,
    model_name: str = "credential_detector"
) -> Dict[str, str]:
    """
    Converte o modelo CredentialDetector para formato ONNX.
    
    Args:
        model_path: Caminho para o arquivo .pkl do modelo treinado
        output_dir: Diretório onde os arquivos convertidos serão salvos
        model_name: Nome base para os arquivos gerados
    
    Returns:
        Dicionário com os caminhos para os arquivos gerados
    """
    # Criar o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar o modelo
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    classifier = model_data["classifier"]
    vectorizer = model_data["vectorizer"]
    feature_extractor = model_data.get("feature_extractor")
    config = model_data.get("config", {})
    
    # Caminhos dos arquivos de saída
    onnx_model_path = os.path.join(output_dir, f"{model_name}.onnx")
    vectorizer_path = os.path.join(output_dir, f"{model_name}_vectorizer.json")
    config_path = os.path.join(output_dir, f"{model_name}_config.json")
    
    # Converter o classificador para ONNX
    # Precisamos determinar a dimensão de entrada
    input_dim = len(vectorizer.vocabulary_)
    onnx_bytes = convert_sklearn_model_to_onnx(
        classifier, 
        model_name, 
        input_dim,
        onnx_model_path
    )
    
    # Converter o vectorizer (extrair parâmetros)
    vectorizer_data, _ = convert_vectorizer_to_onnx(
        vectorizer,
        f"{model_name}_vectorizer",
        vectorizer_path
    )
    
    # Salvar a configuração
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "onnx_version": onnx.__version__,
            "input_dim": input_dim,
            "config": config,
            "feature_names": list(vectorizer.get_feature_names_out()) if hasattr(vectorizer, 'get_feature_names_out') else [],
            "patterns": model_data.get("patterns", {}),
            "confidence_threshold": model_data.get("confidence_threshold", 0.7)
        }, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Conversão ONNX concluída. Arquivos salvos em: {output_dir}")
    
    return {
        "onnx_model": onnx_model_path,
        "vectorizer": vectorizer_path,
        "config": config_path
    }

def test_onnx_inference(
    onnx_model_path: str,
    input_data: np.ndarray
) -> np.ndarray:
    """
    Testa a inferência usando o modelo ONNX.
    
    Args:
        onnx_model_path: Caminho para o modelo ONNX
        input_data: Array NumPy com os dados de entrada
        
    Returns:
        Resultado da inferência
    """
    # Carregar o modelo ONNX
    session = ort.InferenceSession(onnx_model_path)
    
    # Obter o nome da entrada
    input_name = session.get_inputs()[0].name
    
    # Fazer a inferência
    result = session.run(None, {input_name: input_data.astype(np.float32)})
    
    return result 