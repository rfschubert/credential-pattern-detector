<?php
/**
 * Exemplo de implementação do detector de credenciais em PHP.
 * Requer a extensão PHP ORT (ONNX Runtime)
 * https://github.com/microsoft/onnxruntime-php
 */

namespace RfSchubert\CredentialDetector;

class Detector {
    private $onnxModel;
    private $vectorizer;
    private $patterns;
    private $config;
    private $confidenceThreshold;

    public function __construct(string $modelPath, string $vectorizerPath, string $patternsPath, string $configPath) {
        // Carregar o modelo ONNX
        $this->onnxModel = new \ORT\Session($modelPath);
        
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
