import os
import pickle
import logging
import argparse
import yaml
from typing import Dict, Any, List, Tuple
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

from .data_processor import DataProcessor
from ..utils.text_utils import extract_features

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(
    data_processor: DataProcessor, 
    config: Dict[str, Any],
    output_dir: str
) -> Tuple[Dict[str, Any], float]:
    """
    Treina um modelo de detecção de credenciais.
    
    Args:
        data_processor: Processador de dados
        config: Configuração de treinamento
        output_dir: Diretório para salvar o modelo
        
    Returns:
        Dicionário com o modelo e vectorizer, e a acurácia do modelo
    """
    # Garantir que temos dados processados
    if data_processor.X_train is None:
        data_processor.generate_train_test_split()
        
    # Extrair características
    logger.info("Extraindo características dos dados de treinamento...")
    X_train_features = data_processor.extract_features_batch(data_processor.X_train)
    logger.info("Extraindo características dos dados de teste...")
    X_test_features = data_processor.extract_features_batch(data_processor.X_test)
    
    # Criar e treinar o vectorizer
    logger.info("Treinando o vectorizer TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=config.get('vectorizer_max_features', 1000),
        min_df=config.get('vectorizer_min_df', 2),
        max_df=config.get('vectorizer_max_df', 0.95),
        ngram_range=(1, config.get('vectorizer_ngram_max', 2))
    )
    
    # Transformar os dados de treinamento
    X_train_vec = vectorizer.fit_transform(X_train_features)
    
    # Criar e treinar o modelo
    logger.info("Treinando o modelo de classificação...")
    model_type = config.get('model_type', 'random_forest')
    
    if model_type == 'random_forest':
        if config.get('use_grid_search', False):
            # Usar GridSearchCV para encontrar os melhores parâmetros
            logger.info("Realizando busca por hiperparâmetros com GridSearchCV...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            model.fit(X_train_vec, data_processor.y_train)
            logger.info(f"Melhores parâmetros: {model.best_params_}")
            model = model.best_estimator_
        else:
            # Usar parâmetros definidos na configuração
            model = RandomForestClassifier(
                n_estimators=config.get('rf_n_estimators', 100),
                max_depth=config.get('rf_max_depth', None),
                min_samples_split=config.get('rf_min_samples_split', 2),
                min_samples_leaf=config.get('rf_min_samples_leaf', 1),
                random_state=42
            )
            
            model.fit(X_train_vec, data_processor.y_train)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    # Avaliar o modelo
    logger.info("Avaliando o modelo...")
    X_test_vec = vectorizer.transform(X_test_features)
    
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    
    # Métricas de avaliação
    accuracy = accuracy_score(data_processor.y_test, y_pred)
    logger.info(f"Acurácia: {accuracy:.4f}")
    
    # Relatório de classificação
    report = classification_report(data_processor.y_test, y_pred, target_names=['Não Credencial', 'Credencial'])
    logger.info(f"Relatório de classificação:\n{report}")
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(data_processor.y_test, y_pred)
    logger.info(f"Matriz de confusão:\n{conf_matrix}")
    
    # Salvar o modelo
    os.makedirs(output_dir, exist_ok=True)
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'config': config,
        'metrics': {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report
        },
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    model_path = os.path.join(output_dir, 'credential_detector_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    logger.info(f"Modelo salvo em {model_path}")
    
    # Salvar relatório em formato de texto
    report_path = os.path.join(output_dir, 'training_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Acurácia: {accuracy:.4f}\n\n")
        f.write(f"Relatório de classificação:\n{report}\n\n")
        f.write(f"Matriz de confusão:\n{conf_matrix}\n\n")
        f.write(f"Configuração de treinamento:\n{yaml.dump(config)}\n")
        
    return model_data, accuracy

def main():
    """Função principal para treinamento do modelo."""
    parser = argparse.ArgumentParser(description='Treina um modelo de detecção de credenciais')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Caminho para o arquivo de configuração de treinamento')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Diretório onde os dados estão armazenados')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Diretório onde o modelo treinado será salvo')
    parser.add_argument('--min-examples', type=int, default=200,
                        help='Número mínimo de exemplos para cada classe')
    args = parser.parse_args()
    
    # Configuração padrão
    config = {
        'model_type': 'random_forest',
        'rf_n_estimators': 100,
        'rf_max_depth': None,
        'rf_min_samples_split': 2,
        'rf_min_samples_leaf': 1,
        'vectorizer_max_features': 1000,
        'vectorizer_min_df': 2,
        'vectorizer_max_df': 0.95,
        'vectorizer_ngram_max': 2,
        'use_grid_search': False
    }
    
    # Carregar configuração de arquivo, se disponível
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            loaded_config = yaml.safe_load(f)
            config.update(loaded_config)
    else:
        logger.warning(f"Arquivo de configuração não encontrado: {args.config}")
        logger.info("Usando configuração padrão")
        
        # Criar diretório de configuração e salvar configuração padrão
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    # Definir diretório de saída padrão se não for fornecido
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    logger.info(f"Configuração de treinamento: {config}")
    
    # Inicializar processador de dados
    data_processor = DataProcessor(data_dir=args.data_dir)
    
    # Carregar ou criar exemplos
    data_processor.load_or_create_examples(min_examples=args.min_examples)
    
    # Gerar divisão de treinamento e teste
    data_processor.generate_train_test_split()
    
    # Treinar modelo
    train_model(data_processor, config, args.output_dir)
    
if __name__ == "__main__":
    main() 