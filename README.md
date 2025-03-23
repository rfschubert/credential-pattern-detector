# Detector de Padrões de Credenciais

Uma inteligência artificial projetada para identificar padrões de credenciais em strings de texto, ajudando a prevenir vazamentos de dados sensíveis em sistemas de chat e outras aplicações de texto.

## Objetivo

- Receber uma string (que pode ser multi-linha) e identificar se contém uma credencial potencial
- Não é necessário compreender o significado do texto, apenas detectar se representa uma credencial
- Permitir implementação em sistemas de chat para criptografar automaticamente mensagens contendo informações sensíveis

## Características

- Treinado para reconhecer diversos formatos de credenciais (senhas, tokens, chaves de API, etc.)
- Baixa taxa de falsos positivos
- Rápido o suficiente para uso em aplicações em tempo real
- Fácil integração com diferentes linguagens e plataformas
- **Novo**: Exportação para ONNX, permitindo uso em PHP, JavaScript e outras linguagens

## Estrutura do Projeto

```
credential-pattern-detector/
├── data/                      # Dados para treinamento e testes
│   ├── raw/                   # Dados brutos
│   ├── processed/             # Dados processados para treinamento
│   └── test/                  # Conjunto de testes
├── models/                    # Modelos treinados
│   └── onnx/                  # Modelos exportados para ONNX
├── notebooks/                 # Jupyter notebooks para experimentação
├── src/                       # Código fonte
│   ├── detector/              # Módulo principal de detecção
│   ├── training/              # Scripts de treinamento
│   ├── evaluation/            # Scripts de avaliação
│   ├── export/                # Scripts para exportação de modelos
│   └── utils/                 # Utilitários
├── tests/                     # Testes unitários e de integração
├── examples/                  # Exemplos de uso
├── requirements.txt           # Dependências Python
└── setup.py                   # Script de instalação
```

## Instalação

### Usando Docker (Recomendado)

Este projeto utiliza Docker para facilitar o desenvolvimento e garantir reprodutibilidade. 
Certifique-se de ter o Docker e o Docker Compose instalados no seu sistema.

```bash
# Clone o repositório
git clone https://github.com/rfschubert/credential-pattern-detector.git
cd credential-pattern-detector

# Construir as imagens
make build

# Executar os testes
make test
```

Para usuários com GPU AMD Radeon (ROCm):
```bash
# Construir as imagens com suporte a ROCm
make build-rocm

# Executar os testes com ROCm
make test-rocm
```

### Instalação Local (Alternativa)

```bash
# Clone o repositório
git clone https://github.com/rfschubert/credential-pattern-detector.git
cd credential-pattern-detector

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Instale o pacote em modo de desenvolvimento
pip install -e .
```

## Uso Básico

```python
from credential_detector import CredentialDetector

detector = CredentialDetector()

texto = "Minha senha é X#9pL@7!2ZqR e meu usuário é joao123"
resultado = detector.detect(texto)

if resultado.has_credential:
    print(f"Credencial detectada com confiança: {resultado.confidence}")
    print(f"Credenciais encontradas: {resultado.matches}")
else:
    print("Nenhuma credencial detectada")
```

## Treinamento do Modelo

### Adicionando exemplos personalizados para treinamento

Você pode adicionar seus próprios exemplos para treinamento:

1. Manualmente, usando o script de exemplo:

```bash
# Com Docker
docker-compose run --rm app python examples/add_custom_examples.py

# Localmente
python examples/add_custom_examples.py
```

2. Através de arquivos:
   - Adicione exemplos de credenciais em `data/processed/credentials.jsonl`
   - Adicione exemplos de não-credenciais em `data/processed/non_credentials.jsonl`

### Executando o treinamento

```bash
# Com Docker
make train

# Com Docker e GPU AMD (ROCm)
make train-rocm

# Localmente
python -m src.training.train_model
```

Para personalizar o treinamento, edite o arquivo `config/training_config.yaml`.

## Avaliação do Modelo

```bash
# Com Docker
docker-compose run --rm app python -m src.evaluation.evaluate --model models/credential_detector_model.pkl

# Localmente
python -m src.evaluation.evaluate --model models/credential_detector_model.pkl
```

## Exportação para Outras Linguagens (ONNX)

O detector pode ser exportado para o formato ONNX (Open Neural Network Exchange), permitindo seu uso em outras linguagens como PHP, JavaScript, C#, Java, etc.

### Exportando o modelo para ONNX

```bash
# Com Docker
docker-compose run --rm app python -m src.export.export_onnx --model models/credential_detector_model.pkl --output models/onnx

# Localmente
python -m src.export.export_onnx --model models/credential_detector_model.pkl --output models/onnx
```

Após a exportação, serão gerados os seguintes arquivos:
- `credential_detector.onnx`: Modelo em formato ONNX
- `credential_detector_vectorizer.json`: Dados do vectorizer para processamento de texto
- `credential_detector_patterns.json`: Padrões de expressões regulares para detecção
- `credential_detector_config.json`: Configurações do modelo
- `credential_detector_php_example.php`: Exemplo de implementação em PHP
- `credential_detector_js_example.js`: Exemplo de implementação em JavaScript

### Usando o modelo em PHP

Para utilizar o modelo em PHP, consulte o [Cliente PHP para Detector de Credenciais](https://github.com/rfschubert/php-credential-detector).

### Usando o modelo em JavaScript

Um exemplo básico de uso do modelo em JavaScript está disponível no arquivo gerado na exportação.

## Explorando com Jupyter Notebook

```bash
# Iniciar o Jupyter Notebook
make notebook

# Para GPU AMD (ROCm)
make notebook-rocm

# Acesse http://localhost:8888 (ou http://localhost:8889 para ROCm)
```

## Contribuição

Contribuições são bem-vindas! Por favor, leia o arquivo CONTRIBUTING.md para mais detalhes.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.