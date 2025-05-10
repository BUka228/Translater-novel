import os
import re
import logging
from utils.file_utils import sanitize_filename, ensure_dir_exists
import config

# Настройка логирования
# ... (остается без изменений) ...
log_file_path = config.LOG_FILE
ensure_dir_exists(config.LOG_DIR)
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)

def split_novel_into_chapters():
    """
    Разделяет большой текстовый файл новеллы на отдельные файлы глав,
    не перезаписывая существующие.
    """
    logging.info(f"Начинаем разделение файла '{config.INPUT_NOVEL_FILE}' на главы (пропуск существующих).")
    # ... (проверка INPUT_NOVEL_FILE и создание ORIGINAL_CHAPTERS_DIR как раньше) ...
    if not os.path.exists(config.INPUT_NOVEL_FILE):
        logging.error(f"Входной файл не найден: {config.INPUT_NOVEL_FILE}")
        return False
    ensure_dir_exists(config.ORIGINAL_CHAPTERS_DIR)


    current_chapter_content = []
    # Имя файла для введения
    intro_title_base = "0000_Введение"
    intro_filepath = os.path.join(config.ORIGINAL_CHAPTERS_DIR, f"{intro_title_base}.txt")
    current_output_filepath = intro_filepath # Начинаем с файла введения
    current_title_for_log = intro_title_base
    chapter_counter = 0
    output_file = None
    header_found = False

    # --- Открываем файл для введения, только если он не существует или пуст (для первой записи) ---
    should_write_intro = not os.path.exists(intro_filepath) or os.path.getsize(intro_filepath) == 0
    if should_write_intro:
        try:
            output_file = open(intro_filepath, 'w', encoding=config.INPUT_FILE_ENCODING)
            logging.debug(f"Открыт файл для введения: {intro_filepath}")
        except IOError as e:
            logging.error(f"Не удалось открыть файл для введения '{intro_filepath}': {e}")
            # Если не можем писать даже файл введения, это проблема
            return False
    else:
        logging.info(f"Файл введения '{intro_filepath}' уже существует и не пуст. Запись в него не будет производиться до первого заголовка.")
        # output_file остается None, пока не найдем новый заголовок для несуществующего файла


    try:
        with open(config.INPUT_NOVEL_FILE, 'r', encoding=config.INPUT_FILE_ENCODING) as infile:
            for line_num, line in enumerate(infile, 1):
                match = re.match(config.CHAPTER_HEADER_REGEX, line)

                if match: # Найдено начало новой главы
                    header_found = True
                    # 1. Сохраняем предыдущую главу/введение (если output_file был открыт)
                    if output_file and current_chapter_content:
                        try:
                            output_file.write("".join(current_chapter_content).strip() + "\n")
                            logging.info(f" -> Содержимое для '{current_title_for_log}' записано в '{os.path.basename(current_output_filepath)}'.")
                        except IOError as e:
                            logging.error(f"Ошибка записи в файл '{current_output_filepath}': {e}")
                        finally:
                            output_file.close()
                            logging.debug(f" -> Файл '{current_output_filepath}' закрыт.")
                        output_file = None # Сбрасываем для следующей главы

                    # 2. Определяем имя файла для новой главы
                    chapter_counter += 1
                    current_chapter_title_raw = match.group(1).strip()
                    safe_title = sanitize_filename(current_chapter_title_raw, allow_spaces=False)
                    current_title_for_log = f"{chapter_counter:04d}_{safe_title}"
                    current_output_filepath = os.path.join(config.ORIGINAL_CHAPTERS_DIR, f"{current_title_for_log}.txt")

                    # 3. Открываем новый файл для записи, ТОЛЬКО ЕСЛИ ОН НЕ СУЩЕСТВУЕТ
                    current_chapter_content = [] # Очищаем буфер
                    if not os.path.exists(current_output_filepath):
                        try:
                             output_file = open(current_output_filepath, 'w', encoding=config.INPUT_FILE_ENCODING)
                             logging.info(f"Начало новой главы {chapter_counter}: '{current_chapter_title_raw}' -> {os.path.basename(current_output_filepath)}")
                             # output_file.write(line) # Можно записать заголовок
                        except IOError as e:
                             logging.error(f"Не удалось открыть файл для главы '{current_title_for_log}': {e}")
                             output_file = None
                    else:
                        logging.info(f"Файл главы '{os.path.basename(current_output_filepath)}' уже существует. Пропуск записи.")
                        output_file = None # Не будем в него писать

                # Если output_file открыт (т.е. это новая глава или введение, которое мы пишем), добавляем строку
                elif output_file:
                    current_chapter_content.append(line)
                # Если output_file is None, значит текущая глава уже существует или это текст введения, который уже есть.
                # Мы просто читаем строки дальше, пока не найдем заголовок для новой, несуществующей главы.

            # Сохраняем самую последнюю главу (если output_file был открыт)
            if output_file and current_chapter_content:
                 try:
                    output_file.write("".join(current_chapter_content).strip() + "\n")
                    logging.info(f" -> Содержимое для '{current_title_for_log}' записано в '{os.path.basename(current_output_filepath)}'.")
                 except IOError as e:
                     logging.error(f"Ошибка записи в файл '{current_output_filepath}': {e}")
                 finally:
                    output_file.close()
                    logging.debug(f" -> Файл '{current_output_filepath}' закрыт.")

        # Удаляем файл введения, если он был создан пустым, и не было найдено заголовков
        if not header_found and os.path.exists(intro_filepath):
             if os.path.getsize(intro_filepath) == 0:
                  try: os.remove(intro_filepath); logging.info("Удален пустой файл '0000_Введение.txt', т.к. заголовков не найдено.")
                  except OSError as e: logging.warning(f"Не удалось удалить пустой файл введения: {e}")
        # Или если файл введения был создан, но в него ничего не записалось (т.к. первый заголовок был сразу)
        elif os.path.exists(intro_filepath) and os.path.getsize(intro_filepath) == 0 and chapter_counter > 0:
             try: os.remove(intro_filepath); logging.info("Удален пустой файл '0000_Введение.txt'.")
             except OSError as e: logging.warning(f"Не удалось удалить пустой файл введения: {e}")


        logging.info(f"Разделение на главы завершено. Новых глав создано (или обновлено, если были пустыми): {chapter_counter}.")
        return True

    except FileNotFoundError:
        logging.error(f"Критическая ошибка: Файл '{config.INPUT_NOVEL_FILE}' не найден.")
        return False
    except Exception as e:
        logging.exception(f"Произошла непредвиденная ошибка во время разделения файла:")
        if output_file and not output_file.closed:
            output_file.close()
        return False