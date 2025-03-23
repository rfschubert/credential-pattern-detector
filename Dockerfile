FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de requisitos primeiro (para melhor uso de cache)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código do projeto
COPY . .

# Instalar o pacote em modo de desenvolvimento
RUN pip install -e .

# Definir variáveis de ambiente
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Comando padrão
CMD ["python", "-m", "pytest", "-xvs", "tests/"] 