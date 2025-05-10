import os
import re
import json
import time
import logging
import ebooklib
from ebooklib import epub
# BeautifulSoup здесь может не понадобиться, так как мы работаем с текстом
import config
from utils.file_utils import ensure_dir_exists, sanitize_filename
# --- Импортируем функцию вызова API из phase2_translate ---
# Это немного нарушает модульность, но для простоты пока так.
# В идеале, API-вызовы должны быть в отдельном utils модуле.
from phase2_translate import call_gemini_api_with_retries, count_tokens

# Настройка логирования
# ... (остается как было) ...
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

def extract_original_title_from_filename(filename_cn):
    """Извлекает "чистое" оригинальное название главы из имени файла."""
    # Пример имени файла: 0001_1、想等的人（修）.txt
    # Хотим получить: 1、想等的人（修）
    match = re.match(r"^\d+_(.*)\.txt$", filename_cn)
    if match:
        return match.group(1).replace('_', ' ') # Заменяем подчеркивания на пробелы, если они были от sanitize_filename
    return os.path.splitext(filename_cn)[0] # Возвращаем имя файла без номера и расширения как fallback


def translate_chapter_titles_batch(titles_cn_list):
    """
    Переводит список названий глав через API ОДНИМ ПАКЕТОМ (если возможно).
    Возвращает словарь {оригинал: перевод}.
    """
    if not titles_cn_list:
        return {}

    logging.info(f"Пакетный перевод {len(titles_cn_list)} названий глав...")

    # Собираем все названия в один пронумерованный список для промпта
    prompt_titles_section = []
    for i, title_cn in enumerate(titles_cn_list):
        clean_title = title_cn.strip()
        if clean_title: # Пропускаем пустые названия
            prompt_titles_section.append(f"{i+1}. {clean_title}")

    if not prompt_titles_section:
        logging.warning("Нет непустых названий глав для пакетного перевода.")
        return {title_cn: title_cn for title_cn in titles_cn_list} # Возвращаем оригиналы

    titles_to_translate_str = "\n".join(prompt_titles_section)

    # Оптимизированный промпт для пакетного перевода названий
    # Просим модель вернуть результат в том же пронумерованном формате.
    system_prompt = f"""Ты — профессиональный переводчик. Твоя задача - точно и кратко перевести на русский язык следующий пронумерованный список названий глав китайской новеллы. Сохраняй смысл и стиль каждого названия.
В своем ответе верни ТОЛЬКО пронумерованный список переведенных названий, сохраняя исходную нумерацию. Каждое переведенное название должно быть на новой строке.

Например, если на вход подано:
1. 龙王传说
2. 斗罗大陆

Твой ответ должен быть:
1. Легенда о Короле Драконов
2. Боевой Континент

Не добавляй никаких других слов, пояснений или маркеров.
"""

    full_prompt = f"{system_prompt}\n\nОРИГИНАЛЬНЫЕ НАЗВАНИЯ ДЛЯ ПЕРЕВОДА:\n{titles_to_translate_str}\n\nПЕРЕВЕДЕННЫЕ НАЗВАНИЯ:\n"

    estimated_tokens = count_tokens(full_prompt)
    logging.info(f"Промпт для пакетного перевода названий: {estimated_tokens} токенов.")
    # Установим более низкий лимит для пакетного перевода названий, т.к. ответ тоже будет длинным
    # и модели могут хуже справляться с длинными списками однотипных задач.
    # Этот лимит можно настроить в config.py отдельно, если нужно.
    PACKET_TITLE_TRANSLATION_MAX_TOKENS = 20000 # Примерный лимит для этого типа запроса
    if estimated_tokens > PACKET_TITLE_TRANSLATION_MAX_TOKENS:
        logging.error(f"Промпт для пакетного перевода названий ({estimated_tokens} т.) превышает установленный лимит ({PACKET_TITLE_TRANSLATION_MAX_TOKENS} т.). Пакетный перевод отменен. Попробуйте уменьшить количество глав или переводить индивидуально.")
        # Возвращаем оригиналы, чтобы основной процесс не падал
        return {title_cn: title_cn for title_cn in titles_cn_list}

    # Вызываем API
    response_text = call_gemini_api_with_retries(full_prompt)
    translated_titles_map = {}

    if response_text and "[ОШИБКА ПЕРЕВОДА:" not in response_text:
        logging.debug(f"Ответ API на пакетный перевод названий (сырой):\n{response_text}")
        translated_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        parsed_count = 0

        # Создаем словарь из оригинальных названий для быстрого доступа по индексу
        # Исключаем пустые оригинальные названия из этого сопоставления
        indexed_original_titles = [title.strip() for title in titles_cn_list if title.strip()]

        temp_translations = {} # Временный словарь для распарсенных переводов по номеру

        for line_num, translated_line in enumerate(translated_lines):
            # Пытаемся извлечь номер и текст, даже если форматирование немного нарушено
            # Например, "1. Перевод", "1 Перевод", "1) Перевод"
            match = re.match(r"^\s*(\d+)\s*[.)]?\s*(.*)", translated_line)
            if match:
                try:
                    num = int(match.group(1))
                    text = match.group(2).strip()
                    if text: # Сохраняем только если есть текст перевода
                        temp_translations[num] = text
                        logging.debug(f"  Распарсено из ответа: Номер {num}, Текст '{text}'")
                except ValueError:
                    logging.warning(f"  Не удалось распознать номер в строке ответа: '{translated_line}'")
            else:
                # Если строка не начинается с номера, но мы еще не распарсили все,
                # и это единственная строка, или она не похожа на мусор,
                # попробуем присвоить ее текущему ожидаемому номеру, если не было явного номера.
                # Это рискованно, но может помочь с ответами без нумерации.
                # Пока оставим более строгий парсинг.
                logging.debug(f"  Строка не соответствует ожидаемому формату (номер. текст): '{translated_line}'")


        # Теперь сопоставляем с оригинальными названиями по их порядковому номеру
        for i, original_title_cn_raw in enumerate(titles_cn_list):
            original_title_cn = original_title_cn_raw.strip() # Берем очищенный для ключа
            if not original_title_cn: # Если оригинальное название было пустым
                translated_titles_map[original_title_cn_raw] = original_title_cn_raw # Оставляем как есть
                continue

            expected_index = i + 1 # Номера в промпте начинались с 1
            
            if expected_index in temp_translations:
                translated_titles_map[original_title_cn_raw] = temp_translations[expected_index]
                logging.debug(f"    Сопоставлено: '{original_title_cn_raw}' -> '{temp_translations[expected_index]}' (пакетно)")
                parsed_count += 1
            else:
                logging.warning(f"  Не удалось найти/сопоставить перевод для №{expected_index} '{original_title_cn_raw}' в пакетном ответе. Используется оригинал.")
                translated_titles_map[original_title_cn_raw] = original_title_cn_raw
        
        # indexed_original_titles был нужен для проверки, все ли непустые оригиналы обработаны
        if parsed_count < len(indexed_original_titles):
             logging.warning(f"Распарсено {parsed_count} из {len(indexed_original_titles)} непустых названий. Некоторые переводы могут отсутствовать.")

    else:
        logging.error(f"Не удалось получить или обработать пакетный перевод названий. Ответ API: {response_text}")
        # В случае ошибки возвращаем оригинальные названия
        for title_cn in titles_cn_list:
            translated_titles_map[title_cn] = title_cn

    return translated_titles_map


def prepare_chapters_with_titles(original_titles_map, translated_titles_map):
    """
    Создает новые файлы глав с добавленным переведенным названием в начало.
    Возвращает путь к папке с подготовленными главами.
    """
    logging.info("Подготовка глав с переведенными названиями...")
    ensure_dir_exists(config.TRANSLATED_CHAPTERS_WITH_TITLES_DIR)
    processed_count = 0

    for original_filename, original_title_text in original_titles_map.items():
        translated_chapter_filepath = os.path.join(config.TRANSLATED_CHAPTERS_DIR, original_filename.replace(".txt", "_ru.txt"))
        output_filepath_with_title = os.path.join(config.TRANSLATED_CHAPTERS_WITH_TITLES_DIR, original_filename.replace(".txt", "_ru.txt")) # Имя файла сохраняем

        if not os.path.exists(translated_chapter_filepath):
            logging.warning(f"Пропущен файл (нет переведенной версии): {original_filename}")
            continue

        try:
            with open(translated_chapter_filepath, 'r', encoding='utf-8') as f_in:
                chapter_content_ru = f_in.read()

            # Получаем переведенное название (или используем оригинальное, если перевод не удался)
            translated_title = translated_titles_map.get(original_title_text, original_title_text)
            # Убираем возможные номера и маркеры из переведенного названия для чистоты
            clean_display_title = re.sub(r"^\d+[、.\s]+", "", translated_title).strip()


            # Проверяем, не содержится ли уже заголовок (очень простая проверка)
            first_few_lines = "\n".join(chapter_content_ru.splitlines()[:3])
            if clean_display_title.lower() in first_few_lines.lower():
                 content_with_title = chapter_content_ru # Заголовок уже есть
                 logging.debug(f"Заголовок '{clean_display_title}' уже присутствует в {original_filename}")
            else:
                 content_with_title = f"## {clean_display_title}\n\n{chapter_content_ru}"

            with open(output_filepath_with_title, 'w', encoding='utf-8') as f_out:
                f_out.write(content_with_title)
            processed_count +=1
            logging.debug(f" -> Подготовлен файл с заголовком: {os.path.basename(output_filepath_with_title)}")

        except Exception as e:
            logging.error(f"Ошибка при подготовке файла {original_filename} с заголовком: {e}")

    logging.info(f"Подготовлено {processed_count} файлов глав с добавленными названиями.")
    return config.TRANSLATED_CHAPTERS_WITH_TITLES_DIR


def create_epub_chapter_from_prepared_file(chapter_filepath, chapter_title_fallback="Без названия"):
    """Читает подготовленный файл главы (с заголовком в первой строке) и создает объект epub.EpubHtml."""
    try:
        with open(chapter_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        title_for_toc = lines[0].strip() if lines else chapter_title_fallback
        title_for_toc = re.sub(r'^#+\s*', '', title_for_toc).strip()
        if not title_for_toc: title_for_toc = chapter_title_fallback

        # Весь остальной текст (включая заголовок, который будет внутри h1)
        # уже должен быть в файле, поэтому просто собираем HTML
        content_text = "".join(lines).strip() # Берем все строки, включая заголовок
        paragraphs = content_text.split('\n\n')
        html_parts = []
        first_paragraph_is_title = True # Предполагаем, что первая строка - заголовок

        for i, p_text in enumerate(paragraphs):
            p_text_stripped = p_text.strip()
            if not p_text_stripped: continue

            escaped_p_text = p_text_stripped.replace('&', '&').replace('<', '<').replace('>', '>')
            if i == 0 and first_paragraph_is_title: # Первую "строку" (которая наш заголовок) делаем H1
                 # Удаляем маркеры ## если они еще там
                 escaped_p_text = re.sub(r'^#+\s*', '', escaped_p_text).strip()
                 html_parts.append(f'<h1>{escaped_p_text}</h1>')
            else:
                 html_parts.append(f'<p>{escaped_p_text}</p>')

        content_html = "\n".join(html_parts)
        return title_for_toc, content_html

    except Exception as e:
        logging.error(f"Ошибка обработки файла {chapter_filepath} для EPUB: {e}")
        return chapter_title_fallback, f"<p>Ошибка обработки главы: {os.path.basename(chapter_filepath)}</p>"


def assemble_epub():
    """
    Переводит названия глав, подготавливает файлы и собирает EPUB.
    """
    logging.info("--- Начало Фазы 3: Подготовка названий и сборка EPUB ---")
    ensure_dir_exists(config.OUTPUT_DIR)
    ensure_dir_exists(config.TRANSLATED_CHAPTERS_WITH_TITLES_DIR) # Папка для глав с заголовками

    # 1. Получаем список оригинальных файлов и извлекаем из них названия
    original_chapter_files = []
    try:
        original_chapter_files = sorted([
            f for f in os.listdir(config.ORIGINAL_CHAPTERS_DIR)
            if f.endswith(".txt") and f != "0000_Введение.txt" # Исключаем введение, если оно есть
        ])
        if not original_chapter_files:
             logging.warning("Не найдены оригинальные файлы глав для извлечения названий.")
             # Попробуем собрать EPUB из того, что есть в TRANSLATED_CHAPTERS_DIR напрямую
             path_to_chapters_for_epub = config.TRANSLATED_CHAPTERS_DIR
        else:
             logging.info(f"Найдено {len(original_chapter_files)} оригинальных файлов для извлечения названий.")
    except Exception as e:
        logging.error(f"Ошибка чтения папки оригинальных глав: {e}")
        path_to_chapters_for_epub = config.TRANSLATED_CHAPTERS_DIR


    if original_chapter_files: # Если есть оригинальные главы для перевода названий
        original_titles_map = {
            filename: extract_original_title_from_filename(filename)
            for filename in original_chapter_files
        }
        titles_cn_list = list(original_titles_map.values())

        # 2. Переводим названия глав
        translated_titles_map_by_original_text = translate_chapter_titles_batch(titles_cn_list)

        # Создаем карту filename -> translated_title
        filename_to_translated_title = {
            fname: translated_titles_map_by_original_text.get(orig_title, orig_title)
            for fname, orig_title in original_titles_map.items()
        }

        # 3. Подготавливаем файлы с переведенными заголовками
        path_to_chapters_for_epub = prepare_chapters_with_titles(original_titles_map, translated_titles_map_by_original_text)
    else: # Если оригинальных глав не было (например, только переведенные), используем их как есть
         logging.info("Оригинальные файлы глав не найдены, используем существующие переведенные файлы для сборки EPUB.")
         path_to_chapters_for_epub = config.TRANSLATED_CHAPTERS_DIR


    # 4. Сборка EPUB из папки path_to_chapters_for_epub
    logging.info(f"Сборка EPUB из папки: {path_to_chapters_for_epub}")
    files_for_epub = []
    try:
        files_for_epub = sorted([f for f in os.listdir(path_to_chapters_for_epub) if f.endswith("_ru.txt")])
    except FileNotFoundError:
         logging.error(f"Папка с подготовленными/переведенными главами не найдена: {path_to_chapters_for_epub}")
         return False
    if not files_for_epub:
        logging.warning("Нет файлов для сборки EPUB.")
        return True

    epub_filepath = os.path.join(config.OUTPUT_DIR, config.EPUB_FILENAME)
    try:
        first_file_match = re.match(r"^\d+_(.*)_ru\.txt$", files_for_epub[0])
        novel_title = first_file_match.group(1).replace('_', ' ') if first_file_match else "Переведенная Новелла"

        book = epub.EpubBook()
        book.set_identifier(f'urn:uuid:{config.EPUB_FILENAME}-{time.time()}')
        book.set_title(novel_title)
        book.set_language(config.EPUB_LANGUAGE); book.add_author(config.EPUB_AUTHOR)

        epub_chapters_list = []; toc_links = []
        logging.info(f"Добавление глав в EPUB '{config.EPUB_FILENAME}'...")
        for i, filename in enumerate(files_for_epub):
            filepath = os.path.join(path_to_chapters_for_epub, filename)
            chapter_epub_filename = f'chapter_{i+1:04d}.xhtml'
            fallback_title = os.path.splitext(filename)[0].replace('_ru', '').replace('_', ' ')

            title_for_toc, content_html = create_epub_chapter_from_prepared_file(filepath, fallback_title)

            if content_html:
                epub_chap = epub.EpubHtml(title=title_for_toc, file_name=chapter_epub_filename, lang=config.EPUB_LANGUAGE)
                epub_chap.content = content_html
                book.add_item(epub_chap); epub_chapters_list.append(epub_chap)
                toc_links.append(epub.Link(chapter_epub_filename, title_for_toc, f'uid_chap_{i+1:04d}'))
                logging.debug(f" -> Добавлена глава: {title_for_toc}")
            else: logging.warning(f" -> Не удалось обработать контент для: {filename}")

        if not epub_chapters_list: logging.error("Не добавлено ни одной главы в EPUB."); return False

        book.spine = ['nav'] + epub_chapters_list
        book.toc = tuple(toc_links)
        book.add_item(epub.EpubNcx()); book.add_item(epub.EpubNav())

        epub.write_epub(epub_filepath, book, {})
        logging.info(f"EPUB успешно создан: {epub_filepath}. Глав: {len(epub_chapters_list)}")
        return True
    except Exception as e: logging.exception("Ошибка создания EPUB:"); return False

# if __name__ == "__main__":
#     assemble_epub()