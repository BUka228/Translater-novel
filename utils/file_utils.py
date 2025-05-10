import re
import os
import json
import logging

# ... (функция sanitize_filename без изменений) ...
def sanitize_filename(name, allow_spaces=False):
    name = re.sub(r'^#+\s+', '', name).strip()
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    if allow_spaces: name = re.sub(r'\s+', ' ', name).strip()
    else: name = re.sub(r'\s+', '_', name).strip()
    max_len = 100
    base, ext = os.path.splitext(name)
    if len(base) > max_len: base = base[:max_len]
    name = base + ext
    name = name.strip('.')
    if not name or name == ext: name = "_invalid_name_" + ext
    return name

# ... (функция ensure_dir_exists без изменений) ...
def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logging.info(f"Создана директория: {dir_path}") # Используем logging
        except OSError as e:
             logging.error(f"Не удалось создать директорию {dir_path}: {e}")
             raise # Передаем исключение дальше, чтобы основной код знал об ошибке



def save_glossary(glossary_data, filepath):
    """Сохраняет данные глоссария в JSON файл."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Сохраняем с отступами для читаемости и с поддержкой кириллицы
            json.dump(glossary_data, f, ensure_ascii=False, indent=4)
        logging.debug(f"Глоссарий сохранен в: {filepath}")
        return True
    except IOError as e:
        logging.error(f"Ошибка записи файла глоссария {filepath}: {e}")
        return False
    except Exception as e:
        logging.error(f"Неожиданная ошибка при сохранении глоссария: {e}")
        return False
