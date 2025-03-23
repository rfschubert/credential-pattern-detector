import re
import string
from typing import Dict, List, Set, Any

# Características comuns em credenciais
SPECIAL_CHARS = set('!@#$%^&*()_+-=[]{}|;:,.<>?/~`')
UPPER_CHARS = set(string.ascii_uppercase)
LOWER_CHARS = set(string.ascii_lowercase)
DIGIT_CHARS = set(string.digits)

def extract_features(text: str) -> str:
    """
    Extrai características do texto para análise de padrões de credenciais.
    
    Args:
        text: Texto a ser analisado
        
    Returns:
        String com características extraídas
    """
    # Estatísticas básicas
    total_length = len(text)
    if total_length == 0:
        return ""
        
    # Contagem de diferentes tipos de caracteres
    num_upper = sum(1 for c in text if c in UPPER_CHARS)
    num_lower = sum(1 for c in text if c in LOWER_CHARS)
    num_digit = sum(1 for c in text if c in DIGIT_CHARS)
    num_special = sum(1 for c in text if c in SPECIAL_CHARS)
    num_whitespace = sum(1 for c in text if c.isspace())
    
    # Proporções
    prop_upper = num_upper / total_length
    prop_lower = num_lower / total_length
    prop_digit = num_digit / total_length
    prop_special = num_special / total_length
    
    # Padrões comuns em credenciais
    has_eq_pattern = 1 if re.search(r'=', text) else 0
    has_colon_pattern = 1 if re.search(r':', text) else 0
    has_quotes = 1 if re.search(r'[\'"]', text) else 0
    has_password_keyword = 1 if re.search(r'(?i)password|senha|pwd', text) else 0
    has_api_keyword = 1 if re.search(r'(?i)api|key|token|secret', text) else 0
    has_hex_pattern = 1 if re.search(r'[0-9a-f]{8,}', text, re.I) else 0
    has_base64_pattern = 1 if re.search(r'[A-Za-z0-9+/]{4,}={0,2}', text) else 0
    
    # Entropia (medida aproximada)
    char_set_richness = (
        (1 if num_upper > 0 else 0) + 
        (1 if num_lower > 0 else 0) + 
        (1 if num_digit > 0 else 0) + 
        (1 if num_special > 0 else 0)
    ) / 4.0
    
    # Métricas de comprimento e espaçamento
    avg_word_length = (
        total_length - num_whitespace
    ) / (num_whitespace + 1) if num_whitespace > 0 else total_length
    
    # Padrões de alternância (uma heurística para complexidade)
    transitions = 0
    for i in range(1, total_length):
        prev_type = get_char_type(text[i-1])
        curr_type = get_char_type(text[i])
        if prev_type != curr_type:
            transitions += 1
    
    transition_rate = transitions / (total_length - 1) if total_length > 1 else 0
    
    # Formatando as características como texto para o vectorizer
    features = f"""length:{total_length} upper:{num_upper} lower:{num_lower} 
    digit:{num_digit} special:{num_special} whitespace:{num_whitespace}
    prop_upper:{prop_upper:.2f} prop_lower:{prop_lower:.2f}
    prop_digit:{prop_digit:.2f} prop_special:{prop_special:.2f}
    has_eq:{has_eq_pattern} has_colon:{has_colon_pattern}
    has_quotes:{has_quotes} has_password_kw:{has_password_keyword}
    has_api_kw:{has_api_keyword} has_hex:{has_hex_pattern}
    has_base64:{has_base64_pattern} char_set_richness:{char_set_richness:.2f}
    avg_word_length:{avg_word_length:.2f} transition_rate:{transition_rate:.2f}"""
    
    return features

def get_char_type(char: str) -> str:
    """
    Determina o tipo de um caractere (upper, lower, digit, special).
    
    Args:
        char: Caractere a ser analisado
        
    Returns:
        String indicando o tipo de caractere
    """
    if char in UPPER_CHARS:
        return "upper"
    elif char in LOWER_CHARS:
        return "lower"
    elif char in DIGIT_CHARS:
        return "digit"
    elif char in SPECIAL_CHARS:
        return "special"
    else:
        return "other"

def tokenize_text(text: str) -> List[str]:
    """
    Tokeniza um texto em palavras e símbolos.
    
    Args:
        text: Texto a ser tokenizado
        
    Returns:
        Lista de tokens
    """
    # Substitui caracteres especiais por espaços
    for char in SPECIAL_CHARS:
        text = text.replace(char, f' {char} ')
    
    # Divide por espaços e remove tokens vazios
    tokens = [token.strip() for token in text.split() if token.strip()]
    return tokens

def is_binary_text(text: str) -> bool:
    """
    Verifica se um texto contém caracteres binários/não-imprimíveis.
    
    Args:
        text: Texto a ser analisado
        
    Returns:
        True se o texto contiver caracteres binários, False caso contrário
    """
    # Verificamos caracteres não-ASCII ou controle (exceto espaços, tabs, etc.)
    for char in text:
        # Código ASCII 0-31 (excluindo alguns controles comuns) e 127 são caracteres de controle
        code = ord(char)
        if (0 <= code <= 8) or (14 <= code <= 31) or (code == 127) or (code > 255):
            return True
            
    # Outra abordagem é verificar se há muitos caracteres não-imprimíveis
    non_printable_count = sum(1 for c in text if not c.isprintable())
    if non_printable_count > len(text) * 0.1:  # Mais de 10% são não-imprimíveis
        return True
            
    return False 