import os
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import config # Импортируем основную конфигурацию для путей и настроек

# --- Настройка логирования для теста ---
logging.basicConfig(
    level=logging.DEBUG, # Установим DEBUG для подробной информации
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler() # Вывод только в консоль для теста
    ]
)

def test_chromadb_initialization_and_basic_ops():
    """
    Тестирует инициализацию ChromaDB, добавление и получение данных.
    """
    logging.info("--- Начало теста ChromaDB ---")

    # --- 1. Проверка конфигурации RAG ---
    if not config.RAG_ENABLED:
        logging.info("RAG отключен в config.py. Тест не будет выполнен.")
        return

    logging.info(f"Путь к базе данных: {config.CHROMA_DB_PATH}")
    logging.info(f"Имя коллекции: {config.CHROMA_COLLECTION_NAME}")
    logging.info(f"Модель эмбеддингов: {config.EMBEDDING_MODEL_NAME}")

    client = None
    collection = None
    _embedding_function = None # Используем локальную переменную

    # --- 2. Попытка инициализации модели эмбеддингов ---
    try:
        logging.info("[Тест] Инициализация модели эмбеддингов...")
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        logging.info("[Тест] Модель эмбеддингов инициализирована УСПЕШНО.")
    except Exception as e:
        logging.exception("[Тест] КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать модель эмбеддингов:")
        return # Прерываем тест, без эмбеддингов ничего не получится

    # --- 3. Попытка инициализации клиента ChromaDB ---
    try:
        logging.info("[Тест] Инициализация клиента ChromaDB...")
        # Убедимся, что папка существует
        if not os.path.exists(config.CHROMA_DB_PATH):
             os.makedirs(config.CHROMA_DB_PATH)
             logging.info(f"[Тест] Создана директория для БД: {config.CHROMA_DB_PATH}")

        client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        logging.info("[Тест] Клиент ChromaDB инициализирован УСПЕШНО.")
    except Exception as e:
        logging.exception("[Тест] КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать клиент ChromaDB:")
        return

    # --- 4. Попытка получить/создать коллекцию и получить count ---
    # Это место, где возникала ошибка в логах
    try:
        logging.info(f"[Тест] Попытка получить/создать коллекцию '{config.CHROMA_COLLECTION_NAME}'...")
        collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=_embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        logging.info("[Тест] Коллекция получена/создана УСПЕШНО.")

        # --- Проверка .count() ---
        logging.info("[Тест] Попытка вызова collection.count()...")
        count = collection.count()
        logging.info(f"[Тест] collection.count() выполнен УСПЕШНО. Текущее количество записей: {count}")
        # --- Конец проверки .count() ---

    except Exception as e:
        logging.exception("[Тест] КРИТИЧЕСКАЯ ОШИБКА: Ошибка при работе с коллекцией (get_or_create_collection или count):")
        # Попробуем получить больше деталей об ошибке
        if "Error loading hnsw index" in str(e):
            logging.error("[Тест] Похоже, проблема именно с загрузкой существующего индекса HNSW.")
        # Пытаемся сбросить клиент, чтобы посмотреть, поможет ли это при следующем запуске
        try:
            logging.warning("[Тест] Попытка сбросить клиент ChromaDB (client.reset())...")
            client.reset() # ВНИМАНИЕ: Это удалит все коллекции в этой БД!
            logging.info("[Тест] Сброс клиента ChromaDB выполнен.")
        except Exception as reset_e:
            logging.error(f"[Тест] Не удалось сбросить клиент ChromaDB: {reset_e}")
        return

    # --- 5. Попытка добавить тестовые данные (если коллекция пуста) ---
    if collection.count() == 0:
        try:
            logging.info("[Тест] Коллекция пуста. Попытка добавить тестовые данные...")
            test_ids = ["test_doc_1", "test_doc_2"]
            test_docs = ["Это первый тестовый документ.", "Это второй документ для теста."]
            test_meta = [{"source": "test_script"} for _ in test_ids]
            collection.add(ids=test_ids, documents=test_docs, metadatas=test_meta)
            new_count = collection.count()
            logging.info(f"[Тест] Тестовые данные добавлены УСПЕШНО. Новое количество записей: {new_count}")
            if new_count != len(test_ids):
                 logging.warning(f"[Тест] Ожидалось {len(test_ids)} записей после добавления, но получено {new_count}.")
        except Exception as e:
            logging.exception("[Тест] ОШИБКА: Не удалось добавить тестовые данные:")
            # Не прерываем, чтобы попробовать получить данные

    # --- 6. Попытка получить тестовые данные ---
    try:
        logging.info("[Тест] Попытка получить данные из коллекции...")
        results = collection.get(include=['metadatas', 'documents'])
        num_results = len(results.get('ids', []))
        logging.info(f"[Тест] Данные получены УСПЕШНО. Количество записей: {num_results}")
        if num_results > 0 and num_results <= 5: # Печатаем немного данных для примера
             logging.debug(f"[Тест] Пример данных: {results}")
        elif num_results > 5:
             logging.info(f"[Тест] Пример первых 5 ID: {results.get('ids', [])[:5]}")
    except Exception as e:
        logging.exception("[Тест] ОШИБКА: Не удалось получить данные из коллекции:")

    # --- 7. Попытка выполнить тестовый поиск ---
    try:
        logging.info("[Тест] Попытка выполнить тестовый поиск (query)...")
        query_results = collection.query(query_texts=["тестовый документ"], n_results=1, include=['documents', 'distances'])
        logging.info("[Тест] Тестовый поиск выполнен УСПЕШНО.")
        logging.debug(f"[Тест] Результаты поиска: {query_results}")
    except Exception as e:
        logging.exception("[Тест] ОШИБКА: Не удалось выполнить тестовый поиск:")

    logging.info("--- Тест ChromaDB завершен ---")

if __name__ == "__main__":
    test_chromadb_initialization_and_basic_ops()