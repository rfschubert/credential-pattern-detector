#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para exportar o modelo de detecção de credenciais para o formato ONNX.
ONNX permite que o modelo seja carregado em diferentes ambientes como PHP e JavaScript.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import json

# Adicionar diretório raiz ao path para importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.onnx_utils import convert_credential_detector_to_onnx
from src.detector.credential_detector import CredentialDetector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def export_model(
    model_path: str,
    output_dir: str,
    model_name: str = "credential_detector",
    test_examples: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Exporta o modelo para ONNX e gera arquivos auxiliares para uso em outras linguagens.
    
    Args:
        model_path: Caminho para o arquivo .pkl do modelo treinado
        output_dir: Diretório onde os arquivos convertidos serão salvos
        model_name: Nome base para os arquivos gerados
        test_examples: Lista de exemplos para testar o modelo após exportação
        
    Returns:
        Dicionário com os caminhos para os arquivos gerados
    """
    logger.info(f"Exportando modelo de {model_path} para ONNX")
    
    # Carregar o detector para extrair os padrões regex também
    detector = CredentialDetector(model_path=model_path)
    
    # Converter o modelo para ONNX
    export_paths = convert_credential_detector_to_onnx(
        model_path=model_path,
        output_dir=output_dir,
        model_name=model_name
    )
    
    # Exportar os padrões regex para uso em outras linguagens
    patterns_path = os.path.join(output_dir, f"{model_name}_patterns.json")
    
    # Convertendo os padrões regex para strings serialáveis
    serializable_patterns = {}
    for pattern_type, pattern in detector.patterns.items():
        if hasattr(pattern, "pattern"):
            serializable_patterns[pattern_type] = {
                "pattern": pattern.pattern,
                "flags": pattern.flags
            }
        else:
            serializable_patterns[pattern_type] = str(pattern)
    
    with open(patterns_path, "w", encoding="utf-8") as f:
        json.dump(serializable_patterns, f, ensure_ascii=False, indent=2)
    
    export_paths["patterns"] = patterns_path
    
    # Gerar exemplos de implementação para PHP e JavaScript
    generate_implementation_examples(output_dir, model_name)
    
    # Testar com exemplos, se fornecidos
    if test_examples:
        test_results = []
        for example in test_examples:
            result = detector.detect(example)
            test_results.append({
                "text": example,
                "has_credential": result.has_credential,
                "confidence": float(result.confidence),
                "matches": result.matches,
                "match_positions": [
                    {"start": start, "end": end, "type": match_type}
                    for start, end, match_type in (result.match_positions or [])
                ]
            })
            
        # Salvar resultados de teste para verificação
        test_results_path = os.path.join(output_dir, f"{model_name}_test_results.json")
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
            
        export_paths["test_results"] = test_results_path
        
    logger.info(f"Exportação concluída. Arquivos gerados em {output_dir}")
    
    # Listar todos os arquivos gerados
    for name, path in export_paths.items():
        logger.info(f"- {name}: {path}")
    
    return export_paths

def generate_implementation_examples(output_dir: str, model_name: str) -> None:
    """
    Gera exemplos de código para implementação em PHP e JavaScript.
    
    Args:
        output_dir: Diretório onde os exemplos serão salvos
        model_name: Nome base do modelo
    """
    # Exemplo de implementação em PHP
    php_example = """<?php
/**
 * Exemplo de implementação do detector de credenciais em PHP.
 * Requer a extensão PHP ORT (ONNX Runtime)
 * https://github.com/microsoft/onnxruntime-php
 */

namespace RfSchubert\\CredentialDetector;

class Detector {
    private $onnxModel;
    private $vectorizer;
    private $patterns;
    private $config;
    private $confidenceThreshold;

    public function __construct(string $modelPath, string $vectorizerPath, string $patternsPath, string $configPath) {
        // Carregar o modelo ONNX
        $this->onnxModel = new \\ORT\\Session($modelPath);
        
        // Carregar o vectorizer
        $this->vectorizer = json_decode(file_get_contents($vectorizerPath), true);
        
        // Carregar os padrões regex
        $this->patterns = json_decode(file_get_contents($patternsPath), true);
        
        // Carregar a configuração
        $this->config = json_decode(file_get_contents($configPath), true);
        
        // Definir o limite de confiança
        $this->confidenceThreshold = $this->config['confidence_threshold'] ?? 0.7;
    }
    
    /**
     * Detecta credenciais em uma string de texto.
     */
    public function detect(string $text): DetectionResult {
        // Detectar usando regex primeiro
        $regexMatches = $this->detectWithRegex($text);
        
        // Se encontrou correspondências por regex, retornar diretamente
        if (!empty($regexMatches['matches'])) {
            return new DetectionResult(
                true,
                1.0,
                $regexMatches['matches'],
                $regexMatches['positions']
            );
        }
        
        // Extrair features
        $features = $this->extractFeatures($text);
        
        // Vectorizar o texto
        $vector = $this->vectorizeText($features);
        
        // Fazer a previsão com o modelo ONNX
        $input = ['input' => $vector];
        $output = $this->onnxModel->run($input);
        
        // O output contém as probabilidades para cada classe [não_credencial, credencial]
        $probabilities = $output[0][0];
        $confidence = $probabilities[1]; // Probabilidade da classe "credencial"
        
        $hasCredential = $confidence >= $this->confidenceThreshold;
        
        return new DetectionResult(
            $hasCredential,
            $confidence,
            $hasCredential ? [$text] : [],
            []
        );
    }
    
    // Implementações dos métodos auxiliares
    // ...
}

class DetectionResult {
    public $hasCredential;
    public $confidence;
    public $matches;
    public $matchPositions;
    
    public function __construct(bool $hasCredential, float $confidence, array $matches, array $matchPositions) {
        $this->hasCredential = $hasCredential;
        $this->confidence = $confidence;
        $this->matches = $matches;
        $this->matchPositions = $matchPositions;
    }
}
"""

    # Exemplo de implementação em JavaScript
    js_example = """/**
 * Exemplo de implementação do detector de credenciais em JavaScript.
 * Requer a biblioteca ONNX.js
 * https://github.com/microsoft/onnxjs
 */

class CredentialDetector {
  constructor(modelPath, vectorizerData, patternsData, configData) {
    this.session = null;
    this.vectorizer = vectorizerData;
    this.patterns = patternsData;
    this.config = configData;
    this.confidenceThreshold = configData.confidence_threshold || 0.7;
    
    // Carregar o modelo ONNX
    this.loadModel(modelPath);
  }
  
  async loadModel(modelPath) {
    // Importar ONNX.js
    const onnx = require('onnxjs');
    
    // Criar uma sessão
    this.session = new onnx.InferenceSession();
    
    // Carregar o modelo
    await this.session.loadModel(modelPath);
    
    console.log('Modelo ONNX carregado com sucesso!');
  }
  
  async detect(text) {
    // Verificar se o modelo foi carregado
    if (!this.session) {
      throw new Error('Modelo ONNX não carregado');
    }
    
    // Detectar usando regex primeiro
    const regexResult = this.detectWithRegex(text);
    
    if (regexResult.matches.length > 0) {
      return {
        hasCredential: true,
        confidence: 1.0,
        matches: regexResult.matches,
        matchPositions: regexResult.positions
      };
    }
    
    // Extrair features
    const features = this.extractFeatures(text);
    
    // Vectorizar o texto
    const vector = this.vectorizeText(features);
    
    // Preparar o tensor de entrada
    const inputTensor = new onnx.Tensor(new Float32Array(vector), 'float32', [1, vector.length]);
    
    // Fazer a inferência
    const outputMap = await this.session.run([inputTensor]);
    const outputTensor = outputMap.values().next().value;
    
    // Obter as probabilidades
    const probabilities = outputTensor.data;
    const confidence = probabilities[1]; // Classe 1 = credencial
    
    const hasCredential = confidence >= this.confidenceThreshold;
    
    return {
      hasCredential,
      confidence,
      matches: hasCredential ? [text] : [],
      matchPositions: []
    };
  }
  
  // Implementações dos métodos auxiliares
  // ...
}

// Exemplo de uso
async function example() {
  // Carregar os dados
  const fs = require('fs');
  const path = require('path');
  
  const modelPath = path.join(__dirname, 'credential_detector.onnx');
  const vectorizerData = JSON.parse(fs.readFileSync(
    path.join(__dirname, 'credential_detector_vectorizer.json'), 'utf8'
  ));
  const patternsData = JSON.parse(fs.readFileSync(
    path.join(__dirname, 'credential_detector_patterns.json'), 'utf8'
  ));
  const configData = JSON.parse(fs.readFileSync(
    path.join(__dirname, 'credential_detector_config.json'), 'utf8'
  ));
  
  // Inicializar o detector
  const detector = new CredentialDetector(
    modelPath, vectorizerData, patternsData, configData
  );
  
  // Detectar credenciais
  const text = 'API_KEY=a1b2c3d4e5f6g7h8i9j0';
  const result = await detector.detect(text);
  
  console.log('Texto:', text);
  console.log('Resultado:', result);
}
"""

    # Salvar os exemplos
    php_example_path = os.path.join(output_dir, f"{model_name}_php_example.php")
    with open(php_example_path, "w", encoding="utf-8") as f:
        f.write(php_example)
    
    js_example_path = os.path.join(output_dir, f"{model_name}_js_example.js")
    with open(js_example_path, "w", encoding="utf-8") as f:
        f.write(js_example)
    
    logger.info(f"Exemplos de implementação gerados em:")
    logger.info(f"- PHP: {php_example_path}")
    logger.info(f"- JavaScript: {js_example_path}")

def main():
    """
    Função principal para execução do script.
    """
    parser = argparse.ArgumentParser(description="Exportar modelo para ONNX")
    parser.add_argument(
        "--model", 
        default="models/credential_detector_model.pkl",
        help="Caminho para o modelo treinado (.pkl)"
    )
    parser.add_argument(
        "--output", 
        default="models/onnx",
        help="Diretório de saída para os arquivos exportados"
    )
    parser.add_argument(
        "--name", 
        default="credential_detector",
        help="Nome base para os arquivos gerados"
    )
    
    args = parser.parse_args()
    
    # Exemplos de teste
    test_examples = [
        "Olá, como vai você?",
        "Minha senha é X#9pL@7!2ZqR e meu usuário é joao123",
        "API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ",
        "AKIA6QOKC2QOKC2EXAMPLE",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAxzYk86/s0Q==\n-----END RSA PRIVATE KEY-----"
    ]
    
    # Exportar o modelo
    export_paths = export_model(
        model_path=args.model,
        output_dir=args.output,
        model_name=args.name,
        test_examples=test_examples
    )
    
    logger.info("Exportação concluída com sucesso!")

if __name__ == "__main__":
    main() 