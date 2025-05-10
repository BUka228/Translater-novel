import os
import time
import logging
import config # Наша конфигурация
from phase1_split import split_novel_into_chapters
from phase2_translate import translate_chapters
from phase3_assemble import assemble_epub
from utils.file_utils import ensure_dir_exists

# --- Настройка логирования ---
log_file_path = config.LOG_FILE
ensure_dir_exists(config.LOG_DIR)
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)

def run_pipeline():
    """Запускает последовательно все фазы обработки новеллы."""
    start_time = time.time()
    logging.info("--- Запуск полного пайплайна перевода новеллы ---")

    # --- Фаза 1: Разделение на главы ---
    logging.info("--- Начало Фазы 1: Разделение на главы ---")
    success_phase1 = False
    try:
        ensure_dir_exists(config.ORIGINAL_CHAPTERS_DIR) # Убедимся, что папка для глав существует
        success_phase1 = split_novel_into_chapters() # Вызываем всегда
        if success_phase1:
            logging.info("--- Завершение Фазы 1: Разделение на главы успешно (или пропущено для существующих) ---")
        else:
            logging.error("--- Завершение Фазы 1: Разделение на главы с ошибкой ---")
            return # Прерываем, если разделение не удалось
    except Exception as e:
        logging.exception("Непредвиденная ошибка во время Фазы 1 (Разделение)")
        return

    # --- Фаза 2: Перевод глав ---
    if success_phase1:
        logging.info("--- Начало Фазы 2: Перевод глав ---")
        success_phase2 = False
        try:
            ensure_dir_exists(config.TRANSLATED_CHAPTERS_DIR)
            success_phase2 = translate_chapters()
            if success_phase2:
                logging.info("--- Завершение Фазы 2: Перевод глав успешно ---")
            else:
                 logging.warning("--- Завершение Фазы 2: Перевод глав завершился с ошибкой или без новых переводов ---")
        except Exception as e:
            logging.exception("Непредвиденная ошибка во время Фазы 2 (Перевод)")
    else:
         logging.error("Фаза 1 не была успешной, Фаза 2 пропускается.")


    # --- Фаза 3: Сборка EPUB ---
    logging.info("--- Начало Фазы 3: Сборка EPUB ---")
    try:
        if not os.path.exists(config.TRANSLATED_CHAPTERS_DIR) or not os.listdir(config.TRANSLATED_CHAPTERS_DIR):
             logging.warning("Нет переведенных глав для сборки EPUB. Фаза 3 пропущена.")
        # Проверяем, была ли Фаза 2 хоть как-то успешна (или если просто не было ошибок)
        elif success_phase2 or (not success_phase1 and os.path.exists(config.TRANSLATED_CHAPTERS_DIR) and os.listdir(config.TRANSLATED_CHAPTERS_DIR)):
             ensure_dir_exists(config.OUTPUT_DIR)
             assemble_epub()
             logging.info("--- Завершение Фазы 3: Сборка EPUB успешно ---")
        else:
             logging.warning("Фаза 2 не была успешной или нет переведенных глав. Сборка EPUB пропущена.")
    except Exception as e:
        logging.exception("Ошибка во время Фазы 3 (Сборка EPUB)")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"--- Пайплайн завершен за {total_time:.2f} секунд ({total_time/60:.2f} минут) ---")

# --- Блок if __name__ == "__main__": ---
if __name__ == "__main__":
    ensure_dir_exists(config.DATA_DIR)
    ensure_dir_exists(config.INPUT_DIR)
    if not os.path.exists(config.GLOSSARY_FILE):
         try:
             with open(config.GLOSSARY_FILE, 'w', encoding='utf-8') as f: f.write("{\n}")
             logging.info(f"Создан пустой файл глоссария: {config.GLOSSARY_FILE}")
         except IOError as e: logging.error(f"Не удалось создать файл глоссария: {e}")
    run_pipeline()