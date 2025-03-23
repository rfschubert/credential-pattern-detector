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

## Estrutura do Projeto

```
credential-pattern-detector/
├── data/                      # Dados para treinamento e testes
│   ├── raw/                   # Dados brutos
│   ├── processed/             # Dados processados para treinamento
│   └── test/                  # Conjunto de testes
├── models/                    # Modelos treinados
├── notebooks/                 # Jupyter notebooks para experimentação
├── src/                       # Código fonte
│   ├── detector/              # Módulo principal de detecção
│   ├── training/              # Scripts de treinamento
│   ├── evaluation/            # Scripts de avaliação
│   └── utils/                 # Utilitários
├── tests/                     # Testes unitários e de integração
├── examples/                  # Exemplos de uso
├── requirements.txt           # Dependências Python
└── setup.py                   # Script de instalação
```

## Instalação

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
else:
    print("Nenhuma credencial detectada")
```

## Contribuição

Contribuições são bem-vindas! Por favor, leia o arquivo CONTRIBUTING.md para mais detalhes.

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes.