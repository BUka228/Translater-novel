import os
import json
import time
import logging
import re
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from collections import deque
import tiktoken

import config
from utils.file_utils import ensure_dir_exists, save_glossary
from utils.rag_utils import initialize_rag, index_all_chapters, find_relevant_chunks

# --- Настройка логирования ---
log_file_path = config.LOG_FILE
ensure_dir_exists(config.LOG_DIR)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)

# --- Инициализация API клиента Google и токенизатора ---
client = None
tokenizer = None
try:
    if not config.GOOGLE_API_KEY:
        raise ValueError("API ключ Google AI не найден в .env или config.py")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    client = genai.GenerativeModel(config.MODEL_NAME)
    logging.info(f"Клиент API Google Gemini инициализирован для модели: {config.MODEL_NAME}")
except Exception as e:
    logging.exception("Ошибка инициализации API клиента Google Gemini.")
    client = None

try:
    # Используем tiktoken для примерной оценки, т.к. Google API не предоставляет точный подсчет заранее
    tokenizer = tiktoken.get_encoding("cl100k_base")
    logging.info("Токенизатор tiktoken (cl100k_base) инициализирован для примерной оценки.")
except Exception:
    logging.warning("Не удалось инициализировать tiktoken. Подсчет токенов будет грубым.")
    tokenizer = None

# --- Инициализация RAG ---
RAG_INITIALIZED = initialize_rag()

# --- Вспомогательные функции ---

def count_tokens(text):
    """Подсчитывает токены в тексте (примерно)."""
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.warning(f"Ошибка tiktoken.encode: {e}. Возвращаем оценку.")
            return len(text) // 3
    else:
        return len(text) // 3

def get_last_n_tokens(text, n_tokens):
    """Возвращает примерно последние N токенов текста."""
    if not tokenizer or n_tokens <= 0:
        estimated_chars = n_tokens * 4
        return text[-estimated_chars:] if estimated_chars > 0 else ""
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) <= n_tokens:
            return text
        else:
            last_tokens = tokens[-n_tokens:]
            return tokenizer.decode(last_tokens, errors='ignore')
    except Exception as e:
        logging.warning(f"Ошибка get_last_n_tokens: {e}. Возвращаем срез.")
        estimated_chars = n_tokens * 4
        return text[-estimated_chars:]

def call_gemini_api_with_retries(prompt_text):
    """Отправляет запрос к Google Gemini API с логикой повторных попыток."""
    if not client:
        logging.error("Клиент Google API не инициализирован.")
        return None

    generation_config = {"temperature": 0.7}
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            logging.debug(f"Попытка Google API №{attempt + 1}/{config.MAX_RETRIES + 1}...")
            response = client.generate_content(
                prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={'timeout': config.API_TIMEOUT}
            )

            # Добавим проверку на наличие кандидатов (может отсутствовать в ответе)
            if not hasattr(response, 'text') or not response.text:
                 # Если текста нет, проверяем причину блокировки
                 block_reason = "Неизвестно"
                 finish_reason = "Неизвестно"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                      block_reason = response.prompt_feedback.block_reason
                 if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0],'finish_reason'):
                      finish_reason = response.candidates[0].finish_reason

                 # Если заблокировано из-за безопасности или другого фатального финиша
                 if block_reason != "BLOCK_REASON_UNSPECIFIED" or finish_reason not in ["FINISH_REASON_UNSPECIFIED", "STOP"]:
                      error_message = f"Ответ заблокирован/прерван ({block_reason}/{finish_reason}). Ответ: {response}"
                      logging.error(error_message)
                      # Не повторяем попытку при блокировке
                      return f"[ОШИБКА ПЕРЕВОДА: {error_message}]"
                 else:
                      # Иная причина отсутствия текста, логгируем и пробуем снова
                      logging.warning(f"Google API вернул пустой текст (Попытка {attempt + 1}). Причина: {block_reason}/{finish_reason}. Ответ: {response}")
                      if attempt >= config.MAX_RETRIES: break
                      wait_time = 2 ** (attempt + 1); logging.info(f"Повтор через {wait_time} сек..."); time.sleep(wait_time); continue

            logging.debug(f"Google API ответ успешно получен: {response.text[:100]}...")
            return response.text

        except google_exceptions.ResourceExhausted as e:
            wait_time = 2 ** (attempt + 1); logging.warning(f"RateLimit Google API (Попытка {attempt + 1}): {e}. Повтор через {wait_time} сек."); time.sleep(wait_time)
        except (google_exceptions.RetryError, google_exceptions.DeadlineExceeded, TimeoutError) as e: # Добавили TimeoutError
             wait_time = 2 ** attempt; logging.warning(f"Сетевая/Таймаут Google API (Попытка {attempt + 1}): {e}. Повтор через {wait_time} сек."); time.sleep(wait_time)
        except google_exceptions.InvalidArgument as e:
             logging.error(f"Неправильный аргумент Google API (Попытка {attempt + 1}): {e}"); return f"[ОШИБКА ПЕРЕВОДА: Неправильный аргумент API]"
        except Exception as e:
            logging.exception(f"Неожиданная ошибка Google API (Попытка {attempt + 1}):")
            wait_time = 5; time.sleep(wait_time)

        if attempt >= config.MAX_RETRIES: logging.error("Достигнут лимит попыток API."); break

    return None


def load_glossary():
    """Загружает глоссарий из JSON файла."""
    try:
        if os.path.exists(config.GLOSSARY_FILE):
            with open(config.GLOSSARY_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip(): return {}
                glossary_data = json.loads(content)
                logging.info(f"Глоссарий загружен. Записей: {len(glossary_data)}")
                return glossary_data
        else: logging.warning(f"Файл глоссария не найден: {config.GLOSSARY_FILE}."); return {}
    except json.JSONDecodeError as e:
         logging.error(f"Ошибка декодирования JSON глоссария {config.GLOSSARY_FILE}: {e}.")
         return {}
    except Exception as e: logging.error(f"Ошибка загрузки глоссария: {e}"); return {}


def format_glossary_for_prompt(glossary_data):
    """Форматирует глоссарий для вставки в промпт."""
    if not glossary_data:
        return "Нет записей."
    # Форматируем как список для лучшего восприятия моделью
    return "Ключевые термины и имена:\n" + "\n".join([f"- {k}: {v}" for k, v in glossary_data.items()])


def parse_api_response_for_glossary(response_text):
    """Извлекает кандидатов для глоссария из ответа API."""
    candidates = {}
    # Ищем блок с более строгим маркером
    match = re.search(r"\[GLOSSARY_CANDIDATES_START\](.*?)\[GLOSSARY_CANDIDATES_END\]", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # Ищем пары "Оригинал: ПредложенныйПеревод", разрешая пробелы вокруг двоеточия
        pairs = re.findall(r"^\s*(.+?)\s*[:：]\s*(.+?)\s*$", content, re.MULTILINE)
        for original, proposed in pairs:
            original = original.strip(); proposed = proposed.strip()
            # Добавляем фильтрацию: пропускаем слишком короткие или слишком длинные термины
            if original and proposed and 1 < len(original) < 50 and 1 < len(proposed) < 50:
                 # Можно добавить проверку, что оригинал содержит хотя бы один китайский иероглиф
                 if re.search(r'[\u4e00-\u9fff]', original):
                      candidates[original] = proposed
                      logging.debug(f" -> Найден кандидат: '{original}': '{proposed}'")
                 else:
                      logging.debug(f" -> Пропущен кандидат (не содержит иероглифов): '{original}'")
            else:
                logging.debug(f" -> Пропущен кандидат (не прошел проверку длины): '{original}': '{proposed}'")

    if not candidates:
        logging.info(" -> Кандидаты для глоссария не найдены в ответе API.")
    return candidates


def update_glossary(current_glossary, candidates):
    """Обновляет глоссарий предложенными кандидатами, избегая конфликтов."""
    added_count = 0; conflict_count = 0
    glossary_changed = False
    for original, proposed in candidates.items():
        if original not in current_glossary:
            # Дополнительная проверка: не добавляем, если перевод совпадает с оригиналом (транслитерация)
            if original != proposed:
                current_glossary[original] = proposed
                added_count += 1; glossary_changed = True
                logging.info(f" -> Глоссарий+: '{original}': '{proposed}'")
            else:
                logging.debug(f" -> Пропущено добавление (перевод совпадает с оригиналом): '{original}'")
        elif current_glossary[original] != proposed:
            conflict_count += 1
            logging.warning(f" -> Конфликт глоссария '{original}': Предложено '{proposed}', оставлено '{current_glossary[original]}'")

    if added_count > 0 or conflict_count > 0:
        logging.info(f"Обновление глоссария: +{added_count}, конфликтов {conflict_count}.")
    return glossary_changed


def extract_translation_from_response(response_text):
     """Извлекает основной текст перевода, удаляя блок кандидатов и другие маркеры."""
     # Сначала ищем маркер начала перевода
     translation_match = re.search(r"\[ПЕРЕВОД_СТАРТ\](.*)", response_text, re.DOTALL | re.IGNORECASE)
     if translation_match:
         text = translation_match.group(1)
         # Затем ищем маркер конца или блок кандидатов, чтобы обрезать
         end_match = re.search(r"\[ПЕРЕВОД_КОНЕЦ\]|\[GLOSSARY_CANDIDATES_START\]", text, re.DOTALL | re.IGNORECASE)
         if end_match:
             translation = text[:end_match.start()].strip()
         else:
             translation = text.strip() # Берем все после [ПЕРЕВОД_СТАРТ], если нет других маркеров
     else:
         # Если нет маркера [ПЕРЕВОД_СТАРТ], пробуем старый метод: удалить блок кандидатов
         translation = re.sub(r"\[GLOSSARY_CANDIDATES_START\](.*?)\[GLOSSARY_CANDIDATES_END\]", "", response_text, flags=re.DOTALL | re.IGNORECASE)
         translation = translation.strip()

     # Финальная очистка от оставшихся маркеров
     translation = re.sub(r"\[/?(ПЕРЕВОД|ПЕРЕВОД_СТАРТ|ПЕРЕВОД_КОНЕЦ|GLOSSARY_CANDIDATES_START|GLOSSARY_CANDIDATES_END)\]", "", translation, flags=re.IGNORECASE)
     return translation.strip()

# --- Основная функция перевода ---

import os
import json
import time
import logging
import re
import google.generativeai as genai 
from google.api_core import exceptions as google_exceptions 
from collections import deque
import tiktoken 

import config
from utils.file_utils import ensure_dir_exists, save_glossary 
from utils.rag_utils import initialize_rag, index_all_chapters, find_relevant_chunks

# --- Настройка логирования ---
log_file_path = config.LOG_FILE
ensure_dir_exists(config.LOG_DIR)
# Закрываем предыдущие обработчики, если они есть
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)

# --- Инициализация API клиента Google и токенизатора ---
client = None
tokenizer = None
try:
    if not config.GOOGLE_API_KEY:
        raise ValueError("API ключ Google AI не найден в .env или config.py")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    # Создаем модель (клиент создается при вызове generate_content)
    client = genai.GenerativeModel(config.MODEL_NAME)
    logging.info(f"Клиент API Google Gemini инициализирован для модели: {config.MODEL_NAME}")
except Exception as e:
    logging.exception("Ошибка инициализации API клиента Google Gemini.")
    client = None

try:
    # Используем tiktoken для примерной оценки, т.к. Google API не предоставляет точный подсчет заранее
    tokenizer = tiktoken.get_encoding("cl100k_base")
    logging.info("Токенизатор tiktoken (cl100k_base) инициализирован для примерной оценки.")
except Exception:
    logging.warning("Не удалось инициализировать tiktoken. Подсчет токенов будет грубым.")
    tokenizer = None

# --- Инициализация RAG ---
RAG_INITIALIZED = initialize_rag()

# --- Вспомогательные функции ---

def count_tokens(text):
    """Подсчитывает токены в тексте (примерно)."""
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.warning(f"Ошибка tiktoken.encode: {e}. Возвращаем оценку.")
            return len(text) // 3
    else:
        return len(text) // 3 # Грубая оценка

def get_last_n_tokens(text, n_tokens):
    """Возвращает примерно последние N токенов текста."""
    if not tokenizer or n_tokens <= 0:
        estimated_chars = n_tokens * 4
        return text[-estimated_chars:] if estimated_chars > 0 else ""
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) <= n_tokens:
            return text
        else:
            last_tokens = tokens[-n_tokens:]
            return tokenizer.decode(last_tokens, errors='ignore')
    except Exception as e:
        logging.warning(f"Ошибка get_last_n_tokens (decode/encode): {e}. Возвращаем срез.")
        estimated_chars = n_tokens * 4
        return text[-estimated_chars:]

def call_gemini_api_with_retries(prompt_text):
    """Отправляет запрос к Google Gemini API с логикой повторных попыток."""
    if not client:
        logging.error("Клиент Google API не инициализирован.")
        return None

    generation_config = {"temperature": 0.7}
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}, # Ослабляем фильтры
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            logging.debug(f"Попытка Google API №{attempt + 1}/{config.MAX_RETRIES + 1}...")
            response = client.generate_content(
                prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={'timeout': config.API_TIMEOUT}
            )

            # Проверка ответа Gemini
            if hasattr(response, 'text') and response.text:
                 logging.debug(f"Google API ответ успешно получен: {response.text[:100]}...")
                 return response.text
            else:
                 block_reason = "Неизвестно"; finish_reason = "Неизвестно"
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback: block_reason = response.prompt_feedback.block_reason
                 if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0],'finish_reason'): finish_reason = response.candidates[0].finish_reason

                 logging.warning(f"Google API вернул пустой/неполный ответ (Попытка {attempt + 1}). Причина: {block_reason}/{finish_reason}. Ответ: {response}")
                 
                 # Если заблокировано или другая фатальная причина - не повторяем
                 if finish_reason not in ["FINISH_REASON_UNSPECIFIED", "STOP", "MAX_TOKENS"]: # MAX_TOKENS тоже считаем успехом (частичным)
                      error_message = f"Ответ заблокирован/прерван ({block_reason}/{finish_reason})"
                      logging.error(error_message)
                      # Возвращаем маркер ошибки, чтобы основной цикл мог это обработать
                      return f"[ОШИБКА ПЕРЕВОДА: {error_message}]" 
                 
                 # Иначе (неизвестная причина или просто пустой текст) - повторяем
                 if attempt >= config.MAX_RETRIES: logging.error("Достигнут лимит попыток для пустого/неполного ответа."); break
                 wait_time = 2 ** (attempt + 1); logging.info(f"Повтор через {wait_time} сек..."); time.sleep(wait_time); continue

        except google_exceptions.ResourceExhausted as e:
            wait_time = 2 ** (attempt + 1); logging.warning(f"RateLimit Google API (Попытка {attempt + 1}): {e}. Повтор через {wait_time} сек."); time.sleep(wait_time)
        except (google_exceptions.RetryError, google_exceptions.DeadlineExceeded, TimeoutError) as e:
             wait_time = 2 ** attempt; logging.warning(f"Сетевая/Таймаут Google API (Попытка {attempt + 1}): {e}. Повтор через {wait_time} сек."); time.sleep(wait_time)
        except google_exceptions.InvalidArgument as e:
             logging.error(f"Неправильный аргумент Google API (Попытка {attempt + 1}): {e}"); return f"[ОШИБКА ПЕРЕВОДА: Неправильный аргумент API]"
        except Exception as e:
            logging.exception(f"Неожиданная ошибка Google API (Попытка {attempt + 1}):")
            wait_time = 5; time.sleep(wait_time)

        if attempt >= config.MAX_RETRIES: logging.error("Не удалось получить ответ от Google API после всех попыток."); break
            
    return None # Возвращаем None, если все попытки не удались


def load_glossary():
    """Загружает глоссарий из JSON файла."""
    try:
        if os.path.exists(config.GLOSSARY_FILE):
            with open(config.GLOSSARY_FILE, 'r', encoding='utf-8') as f:
                content = f.read();
                if not content.strip(): return {}
                glossary_data = json.loads(content)
                logging.info(f"Глоссарий загружен. Записей: {len(glossary_data)}")
                return glossary_data
        else: logging.warning(f"Файл глоссария не найден: {config.GLOSSARY_FILE}."); return {}
    except json.JSONDecodeError as e:
         logging.error(f"Ошибка декодирования JSON глоссария {config.GLOSSARY_FILE}: {e}.")
         return {}
    except Exception as e: logging.error(f"Ошибка загрузки глоссария: {e}"); return {}


def format_glossary_for_prompt(glossary_data):
    """Форматирует глоссарий для вставки в промпт."""
    if not glossary_data:
        return "Нет записей."
    return "Ключевые термины и имена:\n" + "\n".join([f"- {k}: {v}" for k, v in glossary_data.items()])


def parse_api_response_for_glossary(response_text):
    """Извлекает кандидатов для глоссария из ответа API."""
    candidates = {}
    match = re.search(r"\[GLOSSARY_CANDIDATES_START\](.*?)\[GLOSSARY_CANDIDATES_END\]", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        pairs = re.findall(r"^\s*(.+?)\s*[:：]\s*(.+?)\s*$", content, re.MULTILINE)
        for original, proposed in pairs:
            original = original.strip(); proposed = proposed.strip()
            if original and proposed and 1 < len(original) < 50 and 1 < len(proposed) < 50:
                 if re.search(r'[\u4e00-\u9fff]', original): # Проверка на иероглифы
                      candidates[original] = proposed
                      logging.debug(f" -> Найден кандидат: '{original}': '{proposed}'")
                 else: logging.debug(f" -> Пропущен кандидат (не содержит иероглифов): '{original}'")
            else: logging.debug(f" -> Пропущен кандидат (длина): '{original}': '{proposed}'")
    if not candidates: logging.info(" -> Кандидаты для глоссария не найдены в ответе API.")
    return candidates


def update_glossary(current_glossary, candidates):
    """Обновляет глоссарий предложенными кандидатами, избегая конфликтов."""
    added_count = 0; conflict_count = 0
    glossary_changed = False
    for original, proposed in candidates.items():
        if original not in current_glossary:
            if original != proposed: # Не добавляем транслитерацию
                current_glossary[original] = proposed
                added_count += 1; glossary_changed = True
                logging.info(f" -> Глоссарий+: '{original}': '{proposed}'")
            else: logging.debug(f" -> Пропущено добавление (транслит): '{original}'")
        elif current_glossary[original] != proposed:
            conflict_count += 1
            logging.warning(f" -> Конфликт глоссария '{original}': Предложено '{proposed}', оставлено '{current_glossary[original]}'")
    if added_count > 0 or conflict_count > 0: logging.info(f"Обновление глоссария: +{added_count}, !{conflict_count}.")
    return glossary_changed


def extract_translation_from_response(response_text):
     """Извлекает основной текст перевода, удаляя блок кандидатов и другие маркеры."""
     translation_match = re.search(r"\[ПЕРЕВОД_СТАРТ\](.*?)(\[ПЕРЕВОД_КОНЕЦ\]|\[GLOSSARY_CANDIDATES_START\])", response_text, re.DOTALL | re.IGNORECASE)
     if translation_match:
         translation = translation_match.group(1).strip()
     else:
         translation = re.sub(r"\[GLOSSARY_CANDIDATES_START\](.*?)\[GLOSSARY_CANDIDATES_END\]", "", response_text, flags=re.DOTALL | re.IGNORECASE)
         translation = translation.strip()
     translation = re.sub(r"\[/?(ПЕРЕВОД|ПЕРЕВОД_СТАРТ|ПЕРЕВОД_КОНЕЦ|GLOSSARY_CANDIDATES_START|GLOSSARY_CANDIDATES_END)\]", "", translation, flags=re.IGNORECASE)
     return translation.strip()
 
 
# --- Новая функция для проверки качества перевода ---
def is_translation(text, original_text_length, min_paragraphs=3, chinese_char_threshold=0.05):
    """
    Проверяет перевод на брак:
    - Наличие значительного количества китайских иероглифов.
    - Слишком малое количество абзацев (слипшийся текст).
    Возвращает True, если обнаружен брак, иначе False.
    """
    if not text or not isinstance(text, str):
        logging.warning("[Проверка брака] Текст пустой или не строка.")
        return True # Пустой текст - это брак

    # 1. Проверка на китайские символы
    chinese_chars = 0
    total_chars = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff': # Основной диапазон CJK Unified Ideographs
            chinese_chars += 1
        # Считаем только "видимые" символы, исключая пробелы и переносы строк? (Опционально)
        if char.strip():
             total_chars += 1

    if total_chars == 0: # Если текст состоит только из пробелов/переносов
         logging.warning("[Проверка брака] Текст не содержит видимых символов.")
         return True

    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    if chinese_ratio > chinese_char_threshold:
        logging.warning(f"[Проверка брака] Обнаружено слишком много китайских символов: {chinese_chars}/{total_chars} ({chinese_ratio:.2%}) > {chinese_char_threshold:.0%}")
        return True

    # 2. Проверка на количество абзацев (разделенных \n\n)
    # Используем re.split для учета разных комбинаций переносов
    paragraphs = [p for p in re.split(r'\n\s*\n+', text) if p.strip()]
    num_paragraphs = len(paragraphs)

    # Пороговое значение может зависеть от длины оригинала?
    # Пока используем фиксированный минимум, но можно усложнить.
    # Если оригинал был очень коротким, эта проверка может быть ложной.
    # Добавим проверку на длину оригинала, чтобы не отбрасывать короткие главы.
    if original_text_length > 500 and num_paragraphs < min_paragraphs: # Проверяем только для не очень коротких глав
         logging.warning(f"[Проверка брака] Обнаружено слишком мало абзацев: {num_paragraphs} < {min_paragraphs}")
         return True

    return False # Брак не обнаружен

# --- Основная функция перевода ---
def translate_chapters():
    """
    Итеративно переводит главы (двухпроходная система с Google Gemini).
    """
    logging.info("--- Начало Фазы 2: Перевод глав (Двухпроходный с Google Gemini и RAG) ---")
    ensure_dir_exists(config.TRANSLATED_CHAPTERS_DIR)

    if config.RAG_ENABLED:
        if RAG_INITIALIZED:
            index_all_chapters(force_reindex=False)
        else:
            logging.error("RAG включен, но не удалось инициализировать. Перевод без RAG.")
    else:
        logging.info("RAG отключен.")

    glossary_data = load_glossary()

    try:
        original_files = sorted([f for f in os.listdir(config.ORIGINAL_CHAPTERS_DIR) if f.endswith(".txt")])
        logging.info(f"Найдено {len(original_files)} оригинальных глав.")
    except FileNotFoundError:
        logging.error(f"Папка с оригинальными главами не найдена: {config.ORIGINAL_CHAPTERS_DIR}")
        return False
    except Exception as e:
        logging.error(f"Ошибка чтения папки оригиналов: {e}")
        return False

    previous_chapters_context_queue = deque(maxlen=4) # Храним (filename, text)
    total_chapters = len(original_files); processed_count = 0; chapters_with_api_errors = []; chapters_with_брак = []

    chapter_index = 0
    while chapter_index < len(original_files): # Используем while для возможности повтора
        filename = original_files[chapter_index]
        chapter_number = chapter_index + 1
        original_filepath = os.path.join(config.ORIGINAL_CHAPTERS_DIR, filename)
        translated_filename = filename.replace(".txt", "_ru.txt")
        translated_filepath = os.path.join(config.TRANSLATED_CHAPTERS_DIR, translated_filename)

        logging.info(f"--- Обработка главы {chapter_number}/{total_chapters}: {filename} ---")

        # --- Проверка на существование перевода ---
        if os.path.exists(translated_filepath):
            logging.info(f" -> Пропуск: {translated_filename} уже существует.")
            try: # Загружаем оригинал для контекста следующих глав
                 with open(original_filepath, 'r', encoding=config.INPUT_FILE_ENCODING) as f: text = f.read()
                 if not any(entry[0] == filename for entry in previous_chapters_context_queue):
                      previous_chapters_context_queue.append((filename, text))
                      logging.debug(f" -> Добавлен текст пропущенной главы '{filename}' в контекст.")
            except Exception as e:
                 logging.error(f"Ошибка чтения {filename} для контекста: {e}")
            chapter_index += 1 # Переходим к следующей главе
            continue

        # --- Чтение текущей главы ---
        try:
            with open(original_filepath, 'r', encoding=config.INPUT_FILE_ENCODING) as f:
                current_chapter_text = f.read().strip()
            if not current_chapter_text:
                logging.warning(f" -> Файл главы {filename} пуст. Пропуск.")
                chapter_index += 1
                continue
            original_length = len(current_chapter_text) # Запоминаем длину оригинала
            logging.debug(f" -> Текст главы '{filename}' прочитан ({original_length} симв).")
        except Exception as e:
            logging.error(f"Ошибка чтения файла главы {filename}: {e}")
            chapter_index += 1
            continue

        # --- Переменная для результата перевода ---
        final_translated_text = None
        api_error_occurred = False

        # --- Цикл попыток перевода (включая повторы из-за брака) ---
        translation_attempts = 0
        max_translation_attempts = 2 # Сколько раз пытаться перевести главу, если получаем брак

        while translation_attempts < max_translation_attempts:
            translation_attempts += 1
            logging.info(f" -> Попытка перевода №{translation_attempts}/{max_translation_attempts} для главы {filename}...")
            api_error_occurred = False # Сбрасываем флаг ошибки API для новой попытки

            # === Проход 1: Черновой перевод и кандидаты ===
            logging.info(f"   -> Проход 1 (Попытка {translation_attempts}): Запрос...")
            formatted_glossary_p1 = format_glossary_for_prompt(glossary_data)
            glossary_tokens_p1 = count_tokens(formatted_glossary_p1)
            current_chapter_tokens = count_tokens(current_chapter_text)
            available_tokens_p1 = config.MAX_PROMPT_TOKENS - current_chapter_tokens - glossary_tokens_p1 - 2000 # Запас

            context_parts_p1 = []; context_tokens_p1 = glossary_tokens_p1; rag_context_str_p1 = ""
            if config.RAG_ENABLED and RAG_INITIALIZED and config.RAG_NUM_RESULTS > 0:
                chunks_data = find_relevant_chunks(current_chapter_text, config.RAG_NUM_RESULTS, exclude_chapter=filename)
                if chunks_data:
                    rag_context_str_p1 = "\n\n".join([f"### Контекст из {chunk['source']} (Сходство: {1-chunk['distance']:.2f}):\n{chunk['text']}\n###" for chunk in chunks_data])
                    rag_tokens = count_tokens(rag_context_str_p1)
                    if context_tokens_p1 + rag_tokens <= available_tokens_p1:
                        context_parts_p1.append(rag_context_str_p1)
                        context_tokens_p1 += rag_tokens
                        logging.info(f" -> RAG (P1): +{len(chunks_data)} чанков, {rag_tokens} т.")
                    else:
                        logging.warning(" -> RAG не поместился (P1).")
                        rag_context_str_p1 = ""

            prev_ctx_list = list(previous_chapters_context_queue); recent_ctx_parts_p1 = deque()
            # --- Блок N-3 ---
            if len(prev_ctx_list) >= 3:
                 fn, txt = prev_ctx_list[-3]
                 chunk_tk = min(config.PREVIOUS_CHUNK_TOKENS, available_tokens_p1-context_tokens_p1)
                 if chunk_tk > 50:
                      chunk = get_last_n_tokens(txt, chunk_tk)
                      ctk = count_tokens(chunk) # Определяем ctk здесь
                      if context_tokens_p1 + ctk <= available_tokens_p1: # Используем определенный ctk
                           recent_ctx_parts_p1.appendleft(f"### Недавний контекст (конец N-3: {fn}):\n{chunk}\n###")
                           context_tokens_p1 += ctk
            # --- Блок N-2 ---
            if len(prev_ctx_list) >= 2:
                 fn, txt = prev_ctx_list[-2]
                 chunk_tk = min(config.PREVIOUS_CHUNK_TOKENS, available_tokens_p1-context_tokens_p1)
                 if chunk_tk > 50:
                      chunk = get_last_n_tokens(txt, chunk_tk)
                      ctk = count_tokens(chunk) # Определяем ctk здесь
                      if context_tokens_p1 + ctk <= available_tokens_p1: # Используем определенный ctk
                           recent_ctx_parts_p1.appendleft(f"### Недавний контекст (конец N-2: {fn}):\n{chunk}\n###")
                           context_tokens_p1 += ctk
            # --- Блок N-1 ---
            if len(prev_ctx_list) >= 1:
                fn, txt = prev_ctx_list[-1]
                tk = count_tokens(txt)
                if context_tokens_p1 + tk <= available_tokens_p1:
                     recent_ctx_parts_p1.appendleft(f"### Недавний контекст (полная N-1: {fn}):\n{txt}\n###")
                     context_tokens_p1 += tk
                else: # Полная не влезла, пробуем конец
                    chunk_tk = min(config.PREVIOUS_CHUNK_TOKENS, available_tokens_p1-context_tokens_p1)
                    if chunk_tk > 50:
                         # fn, txt = prev_ctx_list[-1] # fn и txt уже определены выше
                         chunk = get_last_n_tokens(txt, chunk_tk)
                         ctk = count_tokens(chunk) # Определяем ctk здесь
                         if context_tokens_p1 + ctk <= available_tokens_p1: # Используем определенный ctk
                              recent_ctx_parts_p1.appendleft(f"### Недавний контекст (конец N-1: {fn}):\n{chunk}\n###")
                              context_tokens_p1 += ctk

            context_parts_p1.extend(list(recent_ctx_parts_p1))

            full_prompt_p1 = f"""**ИНСТРУКЦИЯ:**
Ты — эксперт-переводчик китайских веб-новелл на русский язык. Твои задачи:
1.  **Выполни точный и литературный перевод** текста из секции [ТЕКУЩАЯ_ГЛАВА]. Сохраняй стиль оригинала.
2.  **Проанализируй ОРИГИНАЛЬНЫЙ текст** в [ТЕКУЩАЯ_ГЛАВА] и **предложи КЛЮЧЕВЫЕ термины** для добавления в глоссарий. Включай только:
    *   Имена собственные (людей, организаций, мест).
    *   Названия специфических техник, артефактов, концепций, титулов, фракций.
    *   НЕ включай общие слова, прилагательные, глаголы, если они не являются частью устойчивого термина.
    *   Для каждого термина предложи наиболее подходящий русский перевод.
3.  **Выведи результат СТРОГО в формате ниже:**

[ПЕРЕВОД_СТАРТ]
(Здесь ТОЛЬКО полный русский перевод текста из [ТЕКУЩАЯ_ГЛАВА])
[ПЕРЕВОД_КОНЕЦ]

[GLOSSARY_CANDIDATES_START]
(Здесь список кандидатов в формате 'ОригиналТермин: ПредложенныйРусскийПеревод', каждая пара на новой строке. Если кандидатов нет, оставь эту секцию пустой.)
[GLOSSARY_CANDIDATES_END]

**ИСПОЛЬЗУЙ ЭТИ ДАННЫЕ ДЛЯ ПЕРЕВОДА И АНАЛИЗА:**

**ТЕКУЩИЙ ГЛОССАРИЙ (используй для перевода и избегай повторного предложения):**
{formatted_glossary_p1}

**ПРЕДЫДУЩИЙ КОНТЕКСТ (RAG и недавние главы):**
{"".join(context_parts_p1)}

**ТЕКСТ ДЛЯ ПЕРЕВОДА И АНАЛИЗА:**
[ТЕКУЩАЯ_ГЛАВА: {filename}]
{current_chapter_text}
[/ТЕКУЩАЯ_ГЛАВА]

**РЕЗУЛЬТАТ:**
"""
            final_prompt_tokens_p1 = count_tokens(full_prompt_p1)
            logging.info(f"   -> Промпт P1 (Попытка {translation_attempts}): {final_prompt_tokens_p1} т.")
            if final_prompt_tokens_p1 > config.MAX_PROMPT_TOKENS: logging.warning(" -> Промпт P1 ПРЕВЫШАЕТ лимит!")

            response_p1 = call_gemini_api_with_retries(full_prompt_p1)
            if not response_p1 or "[ОШИБКА ПЕРЕВОДА:" in response_p1:
                logging.error(f"Не получен валидный ответ P1 для {filename} (Попытка {translation_attempts}). Пропуск главы."); api_error_occurred = True; break

            draft_translation = extract_translation_from_response(response_p1)
            glossary_candidates = parse_api_response_for_glossary(response_p1)
            if not draft_translation: logging.warning(f"Не извлечен черновой перевод P1: {response_p1[:200]}...")
            glossary_updated = update_glossary(glossary_data, glossary_candidates)
            if glossary_updated:
                if not save_glossary(glossary_data, config.GLOSSARY_FILE): logging.error("Крит. ошибка: Не сохранен глоссарий!");
                formatted_glossary_p2 = format_glossary_for_prompt(glossary_data)
            else:
                formatted_glossary_p2 = formatted_glossary_p1

            # === Проход 2: Финальный перевод ===
            logging.info(f"   -> Проход 2 (Попытка {translation_attempts}): Запрос...")
            glossary_tokens_p2 = count_tokens(formatted_glossary_p2)
            available_tokens_p2 = config.MAX_PROMPT_TOKENS - current_chapter_tokens - glossary_tokens_p2 - 1000
            context_parts_p2 = []; context_tokens_p2 = glossary_tokens_p2; rag_tokens_to_add_p2 = 0
            if rag_context_str_p1:
                 rag_tokens_check=count_tokens(rag_context_str_p1);
                 if context_tokens_p2+rag_tokens_check<=available_tokens_p2: context_parts_p2.append(rag_context_str_p1); context_tokens_p2+=rag_tokens_check; rag_tokens_to_add_p2=rag_tokens_check; logging.info(f" -> RAG (P2): +{rag_tokens_to_add_p2} т.")
                 else: logging.warning(" -> RAG не поместился (P2).")
            else: logging.info(" -> RAG не добавлялся (P2).")
            temp_prev_chapters_p2=list(previous_chapters_context_queue); recent_context_parts_p2=deque(); available_for_recent=available_tokens_p2-context_tokens_p2
            # --- Блок N-3 для P2 (ИСПРАВЛЕН) ---
            if len(temp_prev_chapters_p2)>=3:
                 fn,txt=temp_prev_chapters_p2[-3];
                 chunk_tk=min(config.PREVIOUS_CHUNK_TOKENS,available_for_recent);
                 if chunk_tk>50:
                     chunk=get_last_n_tokens(txt,chunk_tk); ctk=count_tokens(chunk);
                     if ctk<=available_for_recent: # Проверка внутри
                          recent_context_parts_p2.appendleft(f"### Недавний контекст (конец N-3: {fn}):\n{chunk}\n###"); available_for_recent-=ctk
            # --- Блок N-2 для P2 (ИСПРАВЛЕН) ---
            if len(temp_prev_chapters_p2)>=2:
                 fn,txt=temp_prev_chapters_p2[-2];
                 chunk_tk=min(config.PREVIOUS_CHUNK_TOKENS,available_for_recent);
                 if chunk_tk>50:
                     chunk=get_last_n_tokens(txt,chunk_tk); ctk=count_tokens(chunk);
                     if ctk<=available_for_recent: # Проверка внутри
                          recent_context_parts_p2.appendleft(f"### Недавний контекст (конец N-2: {fn}):\n{chunk}\n###"); available_for_recent-=ctk
            # --- Блок N-1 для P2 (ИСПРАВЛЕН) ---
            if len(temp_prev_chapters_p2)>=1:
                fn,txt=temp_prev_chapters_p2[-1]; tk=count_tokens(txt);
                if tk<=available_for_recent: # Полная глава
                    recent_context_parts_p2.appendleft(f"### Недавний контекст (полная N-1: {fn}):\n{txt}\n###"); available_for_recent-=tk
                else: # Конец главы
                    chunk_tk=min(config.PREVIOUS_CHUNK_TOKENS,available_for_recent);
                    if chunk_tk>50:
                         # fn и txt уже определены из блока if len >= 1
                         chunk=get_last_n_tokens(txt,chunk_tk); ctk=count_tokens(chunk);
                         if ctk<=available_for_recent: # Проверка внутри
                              recent_context_parts_p2.appendleft(f"### Недавний контекст (конец N-1: {fn}):\n{chunk}\n###"); available_for_recent-=ctk
            context_parts_p2.extend(list(recent_context_parts_p2))

            full_prompt_p2 = f"""**ИНСТРУКЦИЯ:**
Ты — профессиональный переводчик китайских веб-новелл на русский язык. Твоя задача - максимально точно перевести текст из секции [ТЕКУЩАЯ_ГЛАВА].
*   **СТРОГО следуй переводам** имен и терминов, указанным в [ГЛОССАРИЙ]. Не придумывай другие переводы для них.
*   Используй [ПРЕДЫДУЩИЙ_КОНТЕКСТ] для понимания сюжета и стиля.
*   **Не добавляй информацию**, которой нет в [ТЕКУЩАЯ_ГЛАВА].
*   **В ответе предоставь ТОЛЬКО финальный русский перевод** текста из [ТЕКУЩАЯ_ГЛАВА], без каких-либо пояснений, заголовков или маркеров секций.

**ИСПОЛЬЗУЙ ЭТИ ДАННЫЕ:**

**ГЛОССАРИЙ (Обязателен к использованию):**
{formatted_glossary_p2}

**ПРЕДЫДУЩИЙ КОНТЕКСТ:**
{"".join(context_parts_p2)}

**ТЕКСТ ДЛЯ ПЕРЕВОДА:**
[ТЕКУЩАЯ_ГЛАВА: {filename}]
{current_chapter_text}
[/ТЕКУЩАЯ_ГЛАВА]

**ФИНАЛЬНЫЙ ПЕРЕВОД:**
"""
            final_prompt_tokens_p2 = count_tokens(full_prompt_p2)
            logging.info(f"   -> Промпт P2 (Попытка {translation_attempts}): {final_prompt_tokens_p2} т.")
            if final_prompt_tokens_p2 > config.MAX_PROMPT_TOKENS: logging.warning(" -> Промпт P2 ПРЕВЫШАЕТ лимит!")

            final_translated_text = call_gemini_api_with_retries(full_prompt_p2)

            # --- Проверка на брак ---
            if final_translated_text and "[ОШИБКА ПЕРЕВОДА:" not in final_translated_text:
                if is_translation(final_translated_text, original_length):
                    logging.warning(f" -> Обнаружен брак в переводе главы {filename} (Попытка {translation_attempts}). Повтор...")
                    final_translated_text = None # Сбрасываем результат, чтобы цикл повторился
                    time.sleep(config.DELAY_BETWEEN_REQUESTS * 2) # Увеличим паузу
                else:
                    logging.info(f" -> Проверка на брак пройдена (Попытка {translation_attempts}).")
                    break # Брак не обнаружен, выходим из цикла попыток
            elif not final_translated_text: # Если API вернул None после всех ретраев
                 logging.error(f"Не получен финальный перевод P2 для {filename} (Попытка {translation_attempts}).")
                 api_error_occurred = True
                 break # Выходим из цикла попыток
            else: # Если API вернул маркер ошибки
                 logging.error(f"Ошибка API при финальном переводе P2 для {filename}: {final_translated_text}")
                 api_error_occurred = True
                 break # Выходим из цикла попыток
            # --- Конец проверки на брак ---

        # --- Обработка результата после цикла попыток ---
        if api_error_occurred:
             chapters_with_api_errors.append(filename)
        elif final_translated_text:
            try:
                with open(translated_filepath, 'w', encoding='utf-8') as f: f.write(final_translated_text.strip())
                logging.info(f" -> Финальный перевод сохранен в: {translated_filename}")
                previous_chapters_context_queue.append((filename, current_chapter_text))
                processed_count += 1
            except IOError as e: logging.error(f"Ошибка записи перевода {translated_filename}: {e}")
        else: # Если все попытки не дали результата (из-за брака)
             logging.error(f"Не удалось получить качественный перевод для {filename} после {max_translation_attempts} попыток (брак). Глава пропущена.")
             chapters_with_брак.append(filename)

        logging.debug(f"Пауза {config.DELAY_BETWEEN_REQUESTS} сек...")
        time.sleep(config.DELAY_BETWEEN_REQUESTS)
        chapter_index += 1 # Переходим к следующей главе
        # --- КОНЕЦ ЦИКЛА WHILE ПО ГЛАВАМ ---

    # --- ЗАВЕРШЕНИЕ ФАЗЫ 2 ---
    logging.info(f"--- Завершение Фазы 2: Перевод глав (Google Gemini). Успешно: {processed_count}/{total_chapters} ---")
    if chapters_with_api_errors: logging.warning(f"Главы с ошибками API: {chapters_with_api_errors}")
    if chapters_with_брак: logging.warning(f"Главы с обнаруженным браком (пропущены): {chapters_with_брак}")
    save_glossary(glossary_data, config.GLOSSARY_FILE)
    return total_chapters > 0 and (processed_count > 0 or not (chapters_with_api_errors or chapters_with_брак)) # Успех, если нет непереведенных из-за ошибок

# if __name__ == "__main__":
#     translate_chapters()