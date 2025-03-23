import os
import sys
import pickle
import argparse
import logging
import json
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc, f1_score
)
import matplotlib.pyplot as plt

from ..detector.credential_detector import CredentialDetector
from ..training.data_processor import DataProcessor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(
    model_path: str,
    data_processor: DataProcessor = None,
    test_file: str = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Avalia um modelo de detecção de credenciais.
    
    Args:
        model_path: Caminho para o modelo treinado
        data_processor: Processador de dados (opcional)
        test_file: Arquivo de teste (opcional)
        output_dir: Diretório para salvar resultados
        
    Returns:
        Dicionário com métricas de avaliação
    """
    # Carregar o modelo
    detector = CredentialDetector(model_path=model_path)
    
    # Se não tiver processador de dados, crie um
    if data_processor is None:
        data_processor = DataProcessor()
    
    # Se tiver arquivo de teste, carregue-o
    if test_file and os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        texts = test_data.get('texts', [])
        labels = test_data.get('labels', [])
    # Senão, use os dados de teste do processador
    elif data_processor.X_test and data_processor.y_test:
        texts = data_processor.X_test
        labels = data_processor.y_test
    else:
        logger.error("Nenhum dado de teste disponível.")
        return {}
    
    # Fazer predições
    predictions = []
    confidences = []
    
    for text in texts:
        result = detector.detect(text)
        predictions.append(1 if result.has_credential else 0)
        confidences.append(result.confidence)
    
    # Calcular métricas
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, target_names=['Não Credencial', 'Credencial'])
    
    logger.info(f"Acurácia: {accuracy:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Matriz de confusão:\n{conf_matrix}")
    logger.info(f"Relatório de classificação:\n{report}")
    
    # Métricas adicionais
    precision, recall, pr_thresholds = precision_recall_curve(labels, confidences)
    fpr, tpr, roc_thresholds = roc_curve(labels, confidences)
    roc_auc = auc(fpr, tpr)
    
    logger.info(f"Área sob a curva ROC: {roc_auc:.4f}")
    
    # Salvar gráficos e relatório
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Criar e salvar gráfico de curva ROC
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva Característica de Operação do Receptor (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        
        # Criar e salvar gráfico de Precisão-Recall
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precisão')
        plt.title('Curva de Precisão-Recall')
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
        
        # Salvar relatório
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Acurácia: {accuracy:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write(f"Área sob a curva ROC: {roc_auc:.4f}\n\n")
            f.write(f"Matriz de confusão:\n{conf_matrix}\n\n")
            f.write(f"Relatório de classificação:\n{report}\n")
    
    # Retornar métricas
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'thresholds': {
            'pr_thresholds': pr_thresholds.tolist(),
            'roc_thresholds': roc_thresholds.tolist()
        },
        'curves': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    return metrics

def main():
    """Função principal para avaliação do modelo."""
    parser = argparse.ArgumentParser(description='Avalia um modelo de detecção de credenciais')
    parser.add_argument('--model', type=str, required=True,
                        help='Caminho para o modelo treinado')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Arquivo com dados de teste (opcional)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Diretório onde os dados estão armazenados (opcional)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Diretório para salvar resultados da avaliação')
    args = parser.parse_args()
    
    # Verificar se o modelo existe
    if not os.path.exists(args.model):
        logger.error(f"Modelo não encontrado: {args.model}")
        return 1
    
    # Definir diretório de saída padrão se não for fornecido
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "evaluation")
    
    # Inicializar processador de dados
    data_processor = None
    if args.data_dir:
        data_processor = DataProcessor(data_dir=args.data_dir)
        data_processor.load_data()
        data_processor.generate_train_test_split()
    
    # Avaliar modelo
    evaluate_model(args.model, data_processor, args.test_file, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 