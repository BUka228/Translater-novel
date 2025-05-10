import os
import logging
import config # Наша конфигурация
from utils.file_utils import ensure_dir_exists # Для создания папки логов, если нужно

# --- Настройка логирования (простое, для вывода в консоль и файл) ---
log_file_path = os.path.join(config.LOG_DIR, 'empty_chapters_check.log')
ensure_dir_exists(config.LOG_DIR)

# Закрываем предыдущие обработчики, если они есть
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8', mode='w'), # Перезаписываем лог каждый раз
        logging.StreamHandler()
    ]
)

def find_empty_chapters():
    """
    Находит и выводит список пустых или содержащих только пробелы файлов глав
    в директории config.ORIGINAL_CHAPTERS_DIR.
    """
    logging.info(f"Поиск пустых глав в директории: {config.ORIGINAL_CHAPTERS_DIR}")
    empty_chapters = []
    files_checked = 0

    if not os.path.exists(config.ORIGINAL_CHAPTERS_DIR):
        logging.error(f"Директория не найдена: {config.ORIGINAL_CHAPTERS_DIR}")
        return

    for filename in os.listdir(config.ORIGINAL_CHAPTERS_DIR):
        if filename.endswith(".txt"):
            files_checked += 1
            filepath = os.path.join(config.ORIGINAL_CHAPTERS_DIR, filename)
            try:
                with open(filepath, 'r', encoding=config.INPUT_FILE_ENCODING) as f:
                    content = f.read()
                if not content.strip(): # Проверяем, пуст ли файл после удаления пробелов
                    empty_chapters.append(filename)
                    logging.info(f"Найдена пустая глава: {filename}")
            except Exception as e:
                logging.error(f"Ошибка при чтении файла {filename}: {e}")

    if empty_chapters:
        logging.info(f"\n--- Список пустых глав ({len(empty_chapters)} из {files_checked} проверенных): ---")
        for chapter_name in empty_chapters:
            print(chapter_name) # Дополнительный вывод в консоль для удобства
            logging.info(chapter_name)
    else:
        logging.info(f"Пустые главы не найдены. Проверено файлов: {files_checked}")

    logging.info("Поиск пустых глав завершен.")

if __name__ == "__main__":
    find_empty_chapters()