import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла (должен лежать рядом с config.py)
# Убедитесь, что файл .env существует и содержит GOOGLE_API_KEY="ВАШ_КЛЮЧ"
load_dotenv()

# --- Основные Пути ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
ORIGINAL_CHAPTERS_DIR = os.path.join(DATA_DIR, 'chapters_original_cn')
TRANSLATED_CHAPTERS_DIR = os.path.join(DATA_DIR, 'chapters_translated_ru')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
LOG_DIR = os.path.join(DATA_DIR, 'logs')
GLOSSARY_FILE = os.path.join(DATA_DIR, 'glossary.json')
LOG_FILE = os.path.join(LOG_DIR, 'translation.log')

# Имя входного файла новеллы (должен лежать в data/input/)
INPUT_NOVEL_FILENAME = 'Найденная_ночь_Полностью_00ksw_Selenium.txt'
INPUT_NOVEL_FILE = os.path.join(INPUT_DIR, INPUT_NOVEL_FILENAME)

# --- Настройки Разделения глав ---
CHAPTER_HEADER_REGEX = r'^##\s+(.*)'
INPUT_FILE_ENCODING = 'utf-8'

# --- Настройки API (Google Gemini) ---
# Ключ API читается из переменной окружения GOOGLE_API_KEY (файл .env)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# Название модели Gemini. 'gemini-1.5-flash-latest' или 'gemini-1.5-pro-latest'
MODEL_NAME = "gemini-2.5-flash-preview-04-17"
# Таймаут ожидания ответа от API в секундах
API_TIMEOUT = 300  # 5 минут
# Максимальное количество повторных попыток при ошибках API
MAX_RETRIES = 3
# Задержка между успешными запросами к API в секундах
DELAY_BETWEEN_REQUESTS = 1.5

# --- Настройки RAG (Retrieval-Augmented Generation) ---
RAG_ENABLED = True
# Модель для создания эмбеддингов (векторов)
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
# Директория для хранения локальной векторной БД ChromaDB
CHROMA_DB_PATH = os.path.join(DATA_DIR, 'chroma_db')
# Имя коллекции в ChromaDB
CHROMA_COLLECTION_NAME = "novel_chapters"
# Количество релевантных RAG-фрагментов для контекста
RAG_NUM_RESULTS = 5
# Стратегия разбиения глав на чанки для RAG
RAG_CHUNK_STRATEGY = 'paragraph'

TRANSLATED_CHAPTERS_WITH_TITLES_DIR = os.path.join(DATA_DIR, 'chapters_translated_ru_with_titles')

# --- Настройки Контекста и Перевода ---
# Максимальное количество токенов для промпта (Примерная оценка! Google считает иначе)
MAX_PROMPT_TOKENS = 800000
# Максимальное количество токенов из "хвоста" предыдущих глав (N-2, N-3...)
PREVIOUS_CHUNK_TOKENS = 1000 # 0, чтобы отключить

# --- Настройки Сборки EPUB ---
EPUB_FILENAME = "Найденная_ночь_Перевод_Gemini.epub"
EPUB_AUTHOR = "会说话的肘子"
EPUB_LANGUAGE = "ru"

# --- Прочее ---
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL