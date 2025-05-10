import os
import logging
import config # Наша конфигурация
from utils.file_utils import ensure_dir_exists

# --- Настройка логирования ---
log_file_path = os.path.join(config.LOG_DIR, 'delete_empty_chapters.log')
ensure_dir_exists(config.LOG_DIR)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)

def remove_empty_chapters():
    """
    Находит и УДАЛЯЕТ пустые или содержащие только пробелы файлы глав
    в директории config.ORIGINAL_CHAPTERS_DIR.
    """
    logging.info(f"Поиск и удаление пустых глав в директории: {config.ORIGINAL_CHAPTERS_DIR}")
    empty_chapters_found = []
    deleted_count = 0
    files_checked = 0

    if not os.path.exists(config.ORIGINAL_CHAPTERS_DIR):
        logging.error(f"Директория не найдена: {config.ORIGINAL_CHAPTERS_DIR}")
        return

    # Получаем список файлов для итерации, чтобы избежать проблем при удалении
    filenames_to_check = [f for f in os.listdir(config.ORIGINAL_CHAPTERS_DIR) if f.endswith(".txt")]

    for filename in filenames_to_check:
        files_checked += 1
        filepath = os.path.join(config.ORIGINAL_CHAPTERS_DIR, filename)
        try:
            is_empty = False
            with open(filepath, 'r', encoding=config.INPUT_FILE_ENCODING) as f:
                content = f.read()
            if not content.strip():
                is_empty = True
                empty_chapters_found.append(filename)

            if is_empty:
                try:
                    os.remove(filepath)
                    logging.info(f"Удалена пустая глава: {filename}")
                    deleted_count += 1
                except OSError as e:
                    logging.error(f"Ошибка при удалении файла {filename}: {e}")

        except Exception as e:
            logging.error(f"Ошибка при обработке файла {filename}: {e}")

    if empty_chapters_found:
        logging.info(f"\n--- Итог удаления: ---")
        logging.info(f"Обнаружено пустых глав: {len(empty_chapters_found)}")
        logging.info(f"Удалено файлов: {deleted_count}")
        if len(empty_chapters_found) != deleted_count:
             logging.warning("Не все обнаруженные пустые главы были удалены из-за ошибок.")
    else:
        logging.info(f"Пустые главы не найдены для удаления. Проверено файлов: {files_checked}")

    logging.info("Удаление пустых глав завершено.")

if __name__ == "__main__":
    # Добавим подтверждение перед удалением
    print(f"ВНИМАНИЕ! Этот скрипт удалит все пустые .txt файлы из папки:")
    print(f"'{os.path.abspath(config.ORIGINAL_CHAPTERS_DIR)}'")
    print("Рекомендуется сначала запустить 'list_empty_chapters.py' для проверки.")
    confirm = input("Вы уверены, что хотите продолжить? (yes/no): ")

    if confirm.lower() == 'yes':
        remove_empty_chapters()
    else:
        print("Удаление отменено.")