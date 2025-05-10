import logging
import re
import chromadb
from chromadb.utils import embedding_functions
import config
import os
from tqdm import tqdm # Для индикатора прогресса

# --- Инициализация (Глобальные переменные модуля) ---
# Сбрасываем их при загрузке модуля
client = None
collection = None
embedding_function = None
rag_init_success = False # Флаг успешной инициализации

def initialize_rag():
    """
    Инициализирует клиент ChromaDB, модель эмбеддингов и коллекцию.
    Устанавливает глобальный флаг rag_init_success.
    Возвращает True в случае успеха, False при ошибке.
    """
    global client, collection, embedding_function, rag_init_success
    # Сбрасываем состояние перед попыткой
    client = None
    collection = None
    embedding_function = None
    rag_init_success = False

    if not config.RAG_ENABLED:
        logging.info("RAG отключен в конфигурации.")
        return False # Инициализация не требуется и не удалась

    try:
        # 1. Инициализация модели эмбеддингов
        logging.info(f"[RAG Init] Инициализация модели эмбеддингов: {config.EMBEDDING_MODEL_NAME}")
        # Создаем локальную переменную, чтобы не присваивать глобальную до успеха
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        logging.info("[RAG Init] Модель эмбеддингов инициализирована.")

        # 2. Инициализация клиента ChromaDB
        logging.info(f"[RAG Init] Инициализация ChromaDB в папке: {config.CHROMA_DB_PATH}")
        ensure_dir_exists(config.CHROMA_DB_PATH) # Убедимся, что папка существует
        
        _client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=chromadb.Settings(anonymized_telemetry=False) # Отключаем телеметрию на всякий случай
        )
        logging.info("[RAG Init] Клиент ChromaDB инициализирован.")

        # 3. Получение или создание коллекции
        logging.info(f"[RAG Init] Получение/создание коллекции: {config.CHROMA_COLLECTION_NAME}")
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=_embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        collection_count = _collection.count() # Получаем количество записей
        logging.info(f"[RAG Init] Коллекция '{config.CHROMA_COLLECTION_NAME}' готова. Записей: {collection_count}")

        # --- Фиксация успеха: Присваиваем глобальные переменные ---
        embedding_function = _embedding_function
        client = _client
        collection = _collection
        rag_init_success = True # Устанавливаем флаг успеха
        # --- Конец фиксации ---

        logging.info("[RAG Init] Инициализация RAG успешно завершена.")
        return True # Возвращаем успех

    except ImportError as e:
        logging.error(f"[RAG Init] Ошибка импорта: {e}. Убедитесь, что 'sentence-transformers' и 'chromadb' установлены.")
        return False
    except Exception as e:
        logging.exception("[RAG Init] Непредвиденная ошибка инициализации RAG:")
        return False

def chunk_text_by_paragraph(text):
    """Делит текст на абзацы."""
    paragraphs = re.split(r'\n\s*\n+', text) # Разделяем по одной или нескольким пустым строкам
    chunks = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 10] # Добавим минимальную длину чанка
    return chunks

def index_chapter(chapter_filename, chapter_text):
    """Индексирует одну главу в ChromaDB."""
    # Используем глобальный флаг для проверки инициализации
    if not rag_init_success or not collection or not config.RAG_ENABLED:
        logging.warning(f"RAG не инициализирован или отключен, пропуск индексации главы {chapter_filename}")
        return

    logging.debug(f"Индексация главы: {chapter_filename}")
    if config.RAG_CHUNK_STRATEGY == 'paragraph':
        chunks = chunk_text_by_paragraph(chapter_text)
    else:
        logging.warning(f"Неизвестная стратегия чанкинга: {config.RAG_CHUNK_STRATEGY}. Используем абзацы.")
        chunks = chunk_text_by_paragraph(chapter_text)

    if not chunks:
        logging.warning(f"Нет подходящих фрагментов (>10 симв.) для индексации в главе {chapter_filename}")
        return

    base_filename = os.path.splitext(chapter_filename)[0]
    ids = [f"{base_filename}-chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"source_chapter": chapter_filename} for _ in range(len(chunks))]

    try:
        # Проверяем, существуют ли уже ID (чтобы избежать ошибки ChromaDB)
        existing_data = collection.get(ids=ids, include=[]) # Просто проверяем ID
        existing_ids = set(existing_data.get('ids', []))

        ids_to_add = []
        docs_to_add = []
        meta_to_add = []

        for i, chunk_id in enumerate(ids):
            if chunk_id not in existing_ids:
                ids_to_add.append(chunk_id)
                docs_to_add.append(chunks[i])
                meta_to_add.append(metadatas[i])

        if ids_to_add:
            collection.add(documents=docs_to_add, metadatas=meta_to_add, ids=ids_to_add)
            logging.debug(f" -> Добавлено {len(ids_to_add)} новых чанков для {chapter_filename}.")
        else:
            logging.debug(f" -> Все чанки для {chapter_filename} уже проиндексированы.")

    except Exception as e:
         logging.error(f"Ошибка добавления чанков для {chapter_filename} в ChromaDB: {e}")


def index_all_chapters(force_reindex=False):
    """Индексирует все оригинальные главы."""
    # Используем флаг для проверки
    if not rag_init_success or not collection or not config.RAG_ENABLED:
        logging.info("RAG не инициализирован или отключен. Индексация пропущена.")
        return

    logging.info("Начало проверки и индексации глав для RAG...")
    if force_reindex:
        logging.warning("Принудительная переиндексация: удаляем старую коллекцию.")
        try:
            client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
            # Важно: После удаления нужно снова вызвать initialize_rag, чтобы пересоздать коллекцию
            if not initialize_rag():
                 logging.error("Не удалось пересоздать коллекцию после удаления для переиндексации.")
                 return
        except Exception as e:
            logging.error(f"Не удалось удалить/пересоздать коллекцию: {e}")
            return

    try:
        original_files = sorted([f for f in os.listdir(config.ORIGINAL_CHAPTERS_DIR) if f.endswith(".txt")])
    except FileNotFoundError:
        logging.error(f"Папка {config.ORIGINAL_CHAPTERS_DIR} не найдена."); return
    except Exception as e:
        logging.error(f"Ошибка чтения папки {config.ORIGINAL_CHAPTERS_DIR}: {e}"); return

    # Получаем список УЖЕ проиндексированных ГЛАВ (не чанков)
    indexed_chapters_set = set()
    try:
         # Получаем все метаданные, извлекаем уникальные имена глав
         all_metadata = collection.get(include=['metadatas']).get('metadatas', [])
         if all_metadata:
              for meta in all_metadata:
                   if meta and 'source_chapter' in meta:
                        indexed_chapters_set.add(meta['source_chapter'])
         logging.info(f"Обнаружено {len(indexed_chapters_set)} уникальных глав в индексе ChromaDB.")
    except Exception as e:
         logging.warning(f"Не удалось получить метаданные из ChromaDB: {e}")

    chapters_to_index = [f for f in original_files if f not in indexed_chapters_set]

    if not chapters_to_index:
        logging.info("Новых глав для индексации не найдено.")
        return

    logging.info(f"Найдено {len(chapters_to_index)} глав для индексации. Начинаем процесс...")
    for filename in tqdm(chapters_to_index, desc="Индексация глав"):
        filepath = os.path.join(config.ORIGINAL_CHAPTERS_DIR, filename)
        try:
            with open(filepath, 'r', encoding=config.INPUT_FILE_ENCODING) as f: chapter_text = f.read()
            if chapter_text: index_chapter(filename, chapter_text)
            else: logging.warning(f"Пропущен пустой файл: {filename}")
        except Exception as e: logging.error(f"Ошибка чтения/индексации {filename}: {e}")

    logging.info("Индексация глав завершена.")


def find_relevant_chunks(query_text, num_results=5, exclude_chapter=None):
    """Находит наиболее релевантные чанки в БД для заданного текста."""
    global collection, rag_init_success # Убедимся, что флаг проверяется
    if not rag_init_success or not collection or not config.RAG_ENABLED or num_results <= 0:
        logging.debug("RAG поиск пропущен (не инициализирован, отключен или num_results=0).")
        return []

    logging.debug(f"Поиск {num_results} RAG чанков для: '{query_text[:100]}...'")
    try:
        where_filter = None
        if exclude_chapter:
             where_filter = {"source_chapter": {"$ne": exclude_chapter}}
             logging.debug(f"Исключаем чанки из главы: {exclude_chapter}")

        results = collection.query(
            query_texts=[query_text], n_results=num_results, where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        relevant_context = []
        if documents:
             for doc, meta, dist in zip(documents, metadatas, distances):
                  source = meta.get('source_chapter', 'unknown') if meta else 'unknown'
                  relevant_context.append({"text": doc, "source": source, "distance": dist})
                  # Оставим детальный лог на DEBUG уровне
                  logging.debug(f" -> Найден RAG чанк [{source}] (Dist: {dist:.4f}): '{doc[:80]}...'")
        # --- ДОБАВЛЕННЫЙ/ИЗМЕНЕННЫЙ ЛОГ ---
        if relevant_context:
             logging.info(f" -> RAG Поиск: Найдено {len(relevant_context)} релевантных чанков.")
        else:
             logging.info(" -> RAG Поиск: Релевантных чанков не найдено.")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        return relevant_context

    except Exception as e:
        logging.error(f"Ошибка поиска в ChromaDB: {e}")
        return []

# --- Остальные утилиты ---
def ensure_dir_exists(dir_path): # Перенесём сюда из file_utils для локальности
    """Создает директорию, если она не существует."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logging.info(f"Создана директория: {dir_path}")
        except OSError as e:
             logging.error(f"Не удалось создать директорию {dir_path}: {e}")
             raise