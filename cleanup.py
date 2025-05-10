import os
import re
import logging
import unicodedata # Для проверки иероглифов
import config # Импортируем настройки для путей и кодировки
from utils.file_utils import ensure_dir_exists

# --- Настройка логирования ---
LOG_FILENAME = 'cleanup_брак.log' # Отдельный лог для очистки
LOG_DIR = config.LOG_DIR # Используем папку логов из конфига
ensure_dir_exists(LOG_DIR)
log_file_path = os.path.join(LOG_DIR, LOG_FILENAME)

# Настраиваем отдельный логгер для этого скрипта
logger = logging.getLogger('cleanup_брак')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Обработчик для файла
fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='w') # Перезаписываем лог при каждом запуске
fh.setFormatter(formatter)
logger.addHandler(fh)
# Обработчик для консоли
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


# --- Параметры проверки на брак ---
# (Можно вынести в config.py или оставить здесь)
MIN_PARAGRAPHS = 3 # Минимум абзацев для не коротких глав
CHINESE_CHAR_THRESHOLD = 0.05 # Макс. допустимая доля китайских символов (5%)
ORIGINAL_LENGTH_THRESHOLD = 500 # Порог длины оригинала для проверки абзацев

# --- Функции проверки (адаптировано из phase2_translate) ---
def check_брак_in_file(filepath):
    """
    Проверяет один файл перевода на брак.
    Возвращает True, если брак обнаружен, иначе False.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text:
            logger.warning(f"Файл пуст: {os.path.basename(filepath)}")
            return True # Считаем пустой файл браком

        # 1. Проверка на китайские символы
        chinese_chars = 0
        total_chars = 0
        for char in text:
            # Диапазон CJK Unified Ideographs + расширения A, B и т.д. (более полный)
            if '\u4e00' <= char <= '\u9fff' or \
               '\u3400' <= char <= '\u4dbf' or \
               '\U00020000' <= char <= '\U0002a6df' or \
               '\U0002a700' <= char <= '\U0002ebef' or \
               '\U0002f800' <= char <= '\U0002fa1f':
                chinese_chars += 1
            if char.strip(): # Считаем только видимые символы
                 total_chars += 1

        if total_chars == 0:
             logger.warning(f"Файл содержит только пробелы/переносы: {os.path.basename(filepath)}")
             return True

        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        if chinese_ratio > CHINESE_CHAR_THRESHOLD:
            logger.info(f"БРАК (Китайский): {os.path.basename(filepath)} ({chinese_ratio:.1%})")
            return True

        # 2. Проверка на количество абзацев
        # Ищем оригинальный файл, чтобы проверить его длину
        original_filename = os.path.basename(filepath).replace('_ru.txt', '.txt')
        original_filepath = os.path.join(config.ORIGINAL_CHAPTERS_DIR, original_filename)
        original_length = 0
        if os.path.exists(original_filepath):
             try:
                 original_length = os.path.getsize(original_filepath)
             except Exception:
                 logger.warning(f"Не удалось получить размер оригинала для {original_filename}")

        paragraphs = [p for p in re.split(r'\n\s*\n+', text) if p.strip()]
        num_paragraphs = len(paragraphs)

        if original_length > ORIGINAL_LENGTH_THRESHOLD and num_paragraphs < MIN_PARAGRAPHS:
            logger.info(f"БРАК (Абзацы): {os.path.basename(filepath)} ({num_paragraphs} < {MIN_PARAGRAPHS}, длина ориг: {original_length})")
            return True

        return False # Брак не обнаружен

    except FileNotFoundError:
        logger.error(f"Файл не найден при проверке: {filepath}")
        return False # Не можем проверить - не удаляем
    except Exception as e:
        logger.error(f"Ошибка при проверке файла {filepath}: {e}")
        return False # Не можем проверить - не удаляем


def cleanup_бракованные_chapters(dry_run=True):
    """
    Проверяет все файлы в папке переводов и удаляет бракованные.
    dry_run=True: Только показывает, что будет удалено.
    dry_run=False: Реально удаляет файлы.
    """
    target_dir = config.TRANSLATED_CHAPTERS_DIR
    logger.info(f"--- Начало проверки на брак в папке: {target_dir} ---")
    if dry_run:
        logger.info("--- РЕЖИМ ПРОБНОГО ЗАПУСКА (dry_run=True): Файлы не будут удалены ---")

    if not os.path.isdir(target_dir):
        logger.error(f"Папка с переводами не найдена: {target_dir}")
        return

    files_to_delete = []
    total_files_checked = 0

    # Получаем список файлов _ru.txt
    try:
         translated_files = [f for f in os.listdir(target_dir) if f.endswith("_ru.txt")]
         total_files_checked = len(translated_files)
         logger.info(f"Найдено {total_files_checked} файлов для проверки.")
    except Exception as e:
         logger.error(f"Ошибка чтения директории {target_dir}: {e}")
         return

    # Проверяем каждый файл
    for filename in translated_files:
        filepath = os.path.join(target_dir, filename)
        if check_брак_in_file(filepath):
            files_to_delete.append(filepath)

    # Выводим/Удаляем файлы
    if files_to_delete:
        logger.info(f"\n--- Обнаружено {len(files_to_delete)} бракованных файлов для удаления ---")
        for filepath in files_to_delete:
            logger.info(f"  - {os.path.basename(filepath)}")

        if not dry_run:
            logger.warning("--- НАЧАЛО УДАЛЕНИЯ ФАЙЛОВ ---")
            deleted_count = 0
            for filepath in files_to_delete:
                try:
                    os.remove(filepath)
                    logger.info(f"УДАЛЕН: {os.path.basename(filepath)}")
                    deleted_count += 1
                except OSError as e:
                    logger.error(f"НЕ УДАЛОСЬ удалить {os.path.basename(filepath)}: {e}")
            logger.info(f"--- Удалено {deleted_count} из {len(files_to_delete)} файлов ---")
        else:
            logger.info("--- Пробный запуск завершен. Файлы не удалялись. ---")
    else:
        logger.info("--- Бракованных файлов не обнаружено ---")

    logger.info(f"--- Проверка завершена. Проверено файлов: {total_files_checked} ---")


# --- Запуск скрипта ---
if __name__ == "__main__":
    # ВНИМАНИЕ: Установите dry_run=False для реального удаления!
    # Перед реальным удалением рекомендуется сделать резервную копию папки chapters_translated_ru
    cleanup_бракованные_chapters(dry_run=True)
    # cleanup_бракованные_chapters(dry_run=False) # Для реального удаления