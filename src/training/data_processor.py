import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
import csv
import random
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..utils.text_utils import extract_features

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processador de dados para treinar o detector de credenciais.
    
    Esta classe gerencia o carregamento, processamento e divisão de
    exemplos para treinamento e validação.
    """
    
    def __init__(
        self,
        data_dir: str = None,
        credentials_file: str = "credentials.jsonl",
        non_credentials_file: str = "non_credentials.jsonl",
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        """
        Inicializa o processador de dados.
        
        Args:
            data_dir: Diretório onde os dados estão armazenados
            credentials_file: Nome do arquivo com exemplos de credenciais
            non_credentials_file: Nome do arquivo com exemplos que não são credenciais
            test_size: Proporção dos dados a ser usada para teste
            random_seed: Semente para aleatoriedade
        """
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Define diretório de dados padrão se não for fornecido
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.data_dir = os.path.join(base_dir, "data", "processed")
        else:
            self.data_dir = data_dir
            
        # Caminhos para arquivos
        self.credentials_path = os.path.join(self.data_dir, credentials_file)
        self.non_credentials_path = os.path.join(self.data_dir, non_credentials_file)
        
        # Garantir que o diretório de dados existe
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Conjuntos de dados
        self.credentials_data = []
        self.non_credentials_data = []
        
        # Dados de treinamento e teste
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def add_credential_example(self, text: str, source: str = "manual") -> None:
        """
        Adiciona um exemplo de credencial ao conjunto de dados.
        
        Args:
            text: Texto contendo credencial
            source: Origem do exemplo (manual, gerado, etc.)
        """
        example = {
            "text": text,
            "is_credential": True,
            "source": source
        }
        
        self.credentials_data.append(example)
        self._append_to_file(example, self.credentials_path)
        
    def add_non_credential_example(self, text: str, source: str = "manual") -> None:
        """
        Adiciona um exemplo que não é credencial ao conjunto de dados.
        
        Args:
            text: Texto que não contém credencial
            source: Origem do exemplo (manual, gerado, etc.)
        """
        example = {
            "text": text,
            "is_credential": False,
            "source": source
        }
        
        self.non_credentials_data.append(example)
        self._append_to_file(example, self.non_credentials_path)
        
    def _append_to_file(self, example: Dict[str, Any], file_path: str) -> None:
        """
        Adiciona um exemplo ao arquivo JSONL.
        
        Args:
            example: Dicionário com dados do exemplo
            file_path: Caminho para o arquivo
        """
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Carrega dados dos arquivos JSONL.
        
        Returns:
            Tupla com exemplos de credenciais e não-credenciais
        """
        # Carrega exemplos de credenciais
        if os.path.exists(self.credentials_path):
            self.credentials_data = self._load_jsonl(self.credentials_path)
            logger.info(f"Carregados {len(self.credentials_data)} exemplos de credenciais")
        else:
            logger.warning(f"Arquivo de credenciais não encontrado: {self.credentials_path}")
            
        # Carrega exemplos de não-credenciais
        if os.path.exists(self.non_credentials_path):
            self.non_credentials_data = self._load_jsonl(self.non_credentials_path)
            logger.info(f"Carregados {len(self.non_credentials_data)} exemplos de não-credenciais")
        else:
            logger.warning(f"Arquivo de não-credenciais não encontrado: {self.non_credentials_path}")
            
        return self.credentials_data, self.non_credentials_data
    
    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Carrega dados de um arquivo JSONL.
        
        Args:
            file_path: Caminho para o arquivo
            
        Returns:
            Lista de dicionários com os dados
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Erro ao analisar JSON na linha {line_num} do arquivo {file_path}: {e}")
                        logger.warning(f"Conteúdo da linha: {line[:100]}...")
                        # Tenta reparar linhas com múltiplos objetos JSON
                        if '}{' in line:
                            logger.info(f"Tentando reparar linha {line_num} com múltiplos objetos JSON")
                            # Divida a linha em múltiplos objetos JSON
                            parts = line.replace('}{', '}|{').split('|')
                            for part in parts:
                                try:
                                    if part.strip():
                                        data.append(json.loads(part))
                                except json.JSONDecodeError:
                                    logger.warning(f"Não foi possível reparar parte da linha: {part[:50]}...")
                        continue
        return data
    
    def load_from_csv(self, file_path: str, has_header: bool = True, 
                     text_col: int = 0, label_col: int = 1) -> None:
        """
        Carrega dados de um arquivo CSV.
        
        Args:
            file_path: Caminho para o arquivo CSV
            has_header: Se o arquivo tem uma linha de cabeçalho
            text_col: Índice da coluna com o texto
            label_col: Índice da coluna com o rótulo (1 para credencial, 0 para não)
        """
        if not os.path.exists(file_path):
            logger.error(f"Arquivo CSV não encontrado: {file_path}")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            # Pular cabeçalho se necessário
            if has_header:
                next(reader)
                
            for row in reader:
                if len(row) > max(text_col, label_col):
                    text = row[text_col].strip()
                    is_credential = bool(int(row[label_col]))
                    
                    if is_credential:
                        self.add_credential_example(text, source="csv_import")
                    else:
                        self.add_non_credential_example(text, source="csv_import")
    
    def generate_train_test_split(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Gera divisão de treinamento e teste a partir dos dados carregados.
        
        Returns:
            Tupla com (X_train, y_train, X_test, y_test)
        """
        if not self.credentials_data and not self.non_credentials_data:
            logger.warning("Nenhum dado carregado. Carregue os dados primeiro com load_data().")
            return [], [], [], []
            
        # Combinar os dados e criar rótulos
        texts = []
        labels = []
        
        for example in self.credentials_data:
            texts.append(example["text"])
            labels.append(1)  # 1 para credencial
            
        for example in self.non_credentials_data:
            texts.append(example["text"])
            labels.append(0)  # 0 para não-credencial
            
        # Fazer a divisão
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels, test_size=self.test_size, random_state=self.random_seed,
            stratify=labels  # Mantém a proporção de classes
        )
        
        logger.info(f"Divisão de dados: {len(self.X_train)} para treino, {len(self.X_test)} para teste")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def extract_features_batch(self, texts: List[str]) -> List[str]:
        """
        Extrai características de uma lista de textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de strings com características extraídas
        """
        features = []
        for text in tqdm(texts, desc="Extraindo características"):
            features.append(extract_features(text))
        return features
    
    def load_or_create_examples(self, min_examples: int = 100) -> None:
        """
        Carrega exemplos existentes ou cria novos se não houver suficientes.
        
        Args:
            min_examples: Número mínimo de exemplos de cada classe
        """
        # Carregar dados existentes
        self.load_data()
        
        # Verificar se há exemplos suficientes
        if len(self.credentials_data) < min_examples:
            logger.info(f"Gerando {min_examples - len(self.credentials_data)} exemplos de credenciais")
            self._generate_credential_examples(min_examples - len(self.credentials_data))
            
        if len(self.non_credentials_data) < min_examples:
            logger.info(f"Gerando {min_examples - len(self.non_credentials_data)} exemplos de não-credenciais")
            self._generate_non_credential_examples(min_examples - len(self.non_credentials_data))
    
    def _generate_credential_examples(self, count: int) -> None:
        """
        Gera exemplos sintéticos de credenciais.
        
        Args:
            count: Número de exemplos a gerar
        """
        # Exemplo de padrões para gerar credenciais
        patterns = [
            # Padrão de API key
            lambda: f"api_key=\"{''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=32))}\"",
            # Padrão de senha
            lambda: f"password={''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*', k=16))}",
            # Padrão de AWS key
            lambda: f"AKIA{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=16))}",
            # Padrão de token JWT
            lambda: f"eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.{''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-', k=32))}.{''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-', k=32))}",
            # Padrão de token de acesso
            lambda: f"access_token=\"{''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=40))}\"",
        ]
        
        for _ in range(count):
            pattern = random.choice(patterns)
            credential = pattern()
            self.add_credential_example(credential, source="synthetic")
    
    def _generate_non_credential_examples(self, count: int) -> None:
        """
        Gera exemplos sintéticos de não-credenciais.
        
        Args:
            count: Número de exemplos a gerar
        """
        # Exemplo de padrões para gerar não-credenciais
        patterns = [
            # Frases normais
            lambda: "Este é um exemplo de texto que não contém credenciais.",
            lambda: "Olá, como vai você hoje?",
            lambda: "O clima está agradável nesta época do ano.",
            # Números regulares
            lambda: f"O valor do produto é {random.randint(10, 1000)} reais.",
            lambda: f"Tenho {random.randint(1, 100)} anos de idade.",
            # Códigos não sensíveis
            lambda: f"ID do usuário: USER_{random.randint(100, 9999)}",
            lambda: f"Código do produto: PROD-{random.randint(1000, 9999)}",
            # URLs normais
            lambda: "Visite nosso site em https://exemplo.com.br",
            lambda: "O documento está em https://docs.exemplo.com/pagina",
        ]
        
        for _ in range(count):
            pattern = random.choice(patterns)
            non_credential = pattern()
            self.add_non_credential_example(non_credential, source="synthetic")
            
    def save_processed_data(self, output_file: str) -> None:
        """
        Salva os dados processados em um arquivo.
        
        Args:
            output_file: Caminho para o arquivo de saída
        """
        if self.X_train is None or self.y_train is None:
            logger.warning("Nenhum dado processado para salvar. Execute generate_train_test_split() primeiro.")
            return
            
        data = {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
            "y_test": self.y_test
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Dados processados salvos em {output_file}")
        
    def load_processed_data(self, input_file: str) -> None:
        """
        Carrega dados processados de um arquivo.
        
        Args:
            input_file: Caminho para o arquivo de entrada
        """
        if not os.path.exists(input_file):
            logger.error(f"Arquivo de dados processados não encontrado: {input_file}")
            return
            
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
            
        self.X_train = data.get("X_train")
        self.y_train = data.get("y_train")
        self.X_test = data.get("X_test")
        self.y_test = data.get("y_test")
        
        logger.info(f"Dados processados carregados de {input_file}")
        
    def add_examples_from_file(self, file_path: str, is_credential: bool) -> int:
        """
        Adiciona exemplos de um arquivo de texto, um por linha.
        
        Args:
            file_path: Caminho para o arquivo
            is_credential: Se os exemplos são credenciais ou não
            
        Returns:
            Número de exemplos adicionados
        """
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            return 0
            
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    if is_credential:
                        self.add_credential_example(line, source=f"file:{os.path.basename(file_path)}")
                    else:
                        self.add_non_credential_example(line, source=f"file:{os.path.basename(file_path)}")
                    count += 1
                    
        logger.info(f"Adicionados {count} exemplos de {'credenciais' if is_credential else 'não-credenciais'} do arquivo {file_path}")
        return count 