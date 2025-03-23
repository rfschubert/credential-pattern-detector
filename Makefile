.PHONY: build up down test train shell notebook format lint clean help build-rocm train-rocm notebook-rocm up-rocm down-rocm

# Variáveis
PROJECT_NAME = credential-pattern-detector

help: ## Mostra esta mensagem de ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Constrói as imagens Docker
	docker-compose build

build-rocm: ## Constrói as imagens Docker com suporte a AMD ROCm
	docker-compose -f docker-compose.rocm.yml build

up: ## Inicia os contêineres em segundo plano
	docker-compose up -d

up-rocm: ## Inicia os contêineres ROCm em segundo plano
	docker-compose -f docker-compose.rocm.yml up -d

down: ## Para e remove os contêineres
	docker-compose down

down-rocm: ## Para e remove os contêineres ROCm
	docker-compose -f docker-compose.rocm.yml down

test: ## Executa testes
	docker-compose run --rm app pytest -xvs tests/

train: ## Treina o modelo
	docker-compose run --rm training

train-rocm: ## Treina o modelo com suporte a AMD ROCm
	docker-compose -f docker-compose.rocm.yml run --rm training-rocm

shell: ## Acessa o shell do contêiner
	docker-compose run --rm app bash

shell-rocm: ## Acessa o shell do contêiner ROCm
	docker-compose -f docker-compose.rocm.yml run --rm training-rocm bash

notebook: ## Inicia o Jupyter Notebook
	docker-compose up notebook

notebook-rocm: ## Inicia o Jupyter Notebook com suporte a AMD ROCm
	docker-compose -f docker-compose.rocm.yml up notebook-rocm

format: ## Formata o código (black e isort)
	docker-compose run --rm app bash -c "black src/ tests/ && isort src/ tests/"

lint: ## Executa análise estática de código
	docker-compose run --rm app bash -c "flake8 src/ tests/ && mypy src/ tests/"

clean: ## Remove arquivos temporários e caches
	docker-compose run --rm app bash -c "find . -type d -name __pycache__ -exec rm -rf {} +; find . -type f -name '*.pyc' -delete; find . -type d -name '.pytest_cache' -exec rm -rf {} +; find . -type d -name '.mypy_cache' -exec rm -rf {} +;"

# Valor padrão se nenhum alvo for especificado
.DEFAULT_GOAL := help 