"""Шаг 2. Перевод вопросов на другие языки.

Три режима:
	model  — переводит сама модель (по умолчанию). Качество перевода нужно проверять вручную.
	export — сохранить вопросы в TSV, чтобы перевести внешним переводчиком.
	import — загрузить готовый перевод из TSV (столбцы: id, текст).

В этом проекте переводы Gemma оказались непригодными, поэтому тестовые и обучающие вопросы
переведены внешней нейросетью (Claude) и загружены режимом import.

Примеры:
	python 02_translate.py --lang ru
	python 02_translate.py --lang kk --mode export
	python 02_translate.py --lang kk --mode import --file kk_translated.tsv
	python 02_translate.py --lang ru --split train      (для обратного направления)
	python 02_translate.py --lang kk --split train --mode import --file train_translated.tsv --column 3 --drop-flagged

Результат: data/questions_{split}_{lang}.jsonl и data/review_{split}_{lang}.tsv —
оригинал и перевод рядом, для проверки. Автопроверка помечает подозрительные переводы
(модель ответила вместо перевода, не тот язык, чужие символы, обрезано) в столбце problems.
Исправленный review-файл можно загрузить обратно: --mode import --file data/review_test_ru.tsv
"""

import argparse

import config
from neolib import common, metrics

# Вопросы из LIMA — это просьбы к ассистенту («Напиши шутку…»). Маленькие модели норовят
# выполнить просьбу вместо перевода, поэтому прямо запрещаем это и даём пример.
TRANSLATE_PROMPT = (
	"You are a professional translator. Translate the text inside <text></text> from English into {language}.\n"
	"The text is a request or a question addressed to an AI assistant. Do NOT answer it and do NOT carry out "
	"the request: only translate it. Keep its meaning, form and length. Keep code, formulas and formatting unchanged.\n"
	"Output only the translation, without the tags and without any comments.\n\n"
	"Example:\n<text>Write a joke about cats.</text>\n{example}\n\n"
	"<text>{text}</text>"
)
EXAMPLES = {
	"ru": "Напишите шутку про кошек.",
	"kk": "Мысықтар туралы әзіл жазыңыз.",
	"tt": "Мәчеләр турында шаян сүз языгыз.",
	"uk": "Напишіть жарт про котів.",
}


def clean_translation(text):
	"""Убирает теги, если модель их повторила."""
	text = text.strip()
	for tag in ("<text>", "</text>", "<translation>", "</translation>"):
		text = text.replace(tag, "")
	return text.strip()


def check_translations(source_rows, translated_rows, lang):
	"""Ищет подозрительные переводы. Возвращает словарь id -> список замечаний."""
	originals = {}
	for row in source_rows:
		originals[row["id"]] = row["question"]
	problems = {}
	for row in translated_rows:
		found = metrics.translation_problems(
			originals[row["id"]], row["question"], lang, row.get("truncated", False),
		)
		if found:
			problems[row["id"]] = found
	return problems


def report_problems(problems, total, review_path):
	if not problems:
		print(f"Автопроверка: подозрительных переводов нет (из {total}). Всё равно просмотрите 20 строк.")
		return
	print(f"\nАвтопроверка: подозрительных переводов {len(problems)} из {total}:")
	for row_id in sorted(problems):
		print(f"  {row_id}: " + "; ".join(problems[row_id]))
	print(f"\nИсправьте их в {review_path} (столбец translation, последний) в текстовом редакторе,")
	print("например в VS Code или Блокноте (Excel может испортить кодировку), и загрузите обратно:")
	print(f"  python 02_translate.py --lang ... --mode import --file {review_path}")


def write_review(path, source_rows, translated_rows, problems):
	"""Оригинал и перевод рядом. Перевод — последний столбец, поэтому этот файл можно
	исправить руками и загрузить обратно через --mode import."""
	with open(path, "w", encoding="utf-8") as f:
		f.write("id\tproblems\toriginal\ttranslation\n")
		for source, translated in zip(source_rows, translated_rows):
			found = problems.get(source["id"], [])
			f.write(
				source["id"] + "\t"
				+ metrics.escape_tsv("; ".join(found)) + "\t"
				+ metrics.escape_tsv(source["question"]) + "\t"
				+ metrics.escape_tsv(translated["question"]) + "\n"
			)


def translate_with_model(source_rows, lang):
	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	language = config.LANG_NAMES.get(lang, lang)
	if lang not in EXAMPLES:
		raise SystemExit(f"Добавьте пример перевода для '{lang}' в EXAMPLES в 02_translate.py")
	prompts = []
	for row in source_rows:
		prompts.append(TRANSLATE_PROMPT.format(language=language, example=EXAMPLES[lang], text=row["question"]))
	results = common.generate_many(
		model, tokenizer, prompts, neo_id, None,
		config.TRANSLATE_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc=f"перевод на {lang}",
	)
	translated = []
	for row, result in zip(source_rows, results):
		translated.append({
			"id": row["id"],
			"question": clean_translation(result["text"]),
			"truncated": result["truncated"],
		})
	return translated


def export_tsv(source_rows, path):
	with open(path, "w", encoding="utf-8") as f:
		f.write("id\ttext\n")
		for row in source_rows:
			f.write(row["id"] + "\t" + metrics.escape_tsv(row["question"]) + "\n")
	print(f"Сохранено для перевода: {path}")


def import_tsv(source_rows, path, column=None):
	"""column — номер столбца с переводом (0 = id); по умолчанию последний."""
	by_id = {}
	with open(path, encoding="utf-8") as f:
		lines = f.read().splitlines()
	for line in lines[1:]:
		if not line.strip():
			continue
		parts = line.split("\t")
		if len(parts) < 2:
			continue
		# По умолчанию берём последний столбец: при переводе в таблице перевод обычно оказывается справа.
		if column is None:
			cell = parts[-1]
		elif column < len(parts):
			cell = parts[column]
		else:
			continue
		by_id[parts[0].strip()] = metrics.unescape_tsv(cell.strip())
	translated = []
	missing = 0
	for row in source_rows:
		if row["id"] in by_id:
			translated.append({"id": row["id"], "question": by_id[row["id"]]})
		else:
			missing += 1
	if missing:
		print(f"ВНИМАНИЕ: нет перевода для {missing} вопросов, они пропущены")
	return translated


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", required=True, help="код языка: ru, kk, ...")
	parser.add_argument("--split", default="test", choices=["test", "train"])
	parser.add_argument("--mode", default="model", choices=["model", "export", "import"])
	parser.add_argument("--file", help="TSV для export/import")
	parser.add_argument("--column", type=int, help="import: номер столбца с переводом (0 = id, 1 = оригинал); по умолчанию последний")
	parser.add_argument("--drop-flagged", action="store_true",
		help="не сохранять подозрительные переводы (для обучающих вопросов, где их не исправить руками)")
	args = parser.parse_args()
	common.set_seed(config.SEED)

	source_rows = common.read_jsonl(common.data_path(f"questions_{args.split}_en.jsonl"))
	out_path = common.data_path(f"questions_{args.split}_{args.lang}.jsonl")
	review_path = common.data_path(f"review_{args.split}_{args.lang}.tsv")

	if args.mode == "export":
		path = args.file or common.data_path(f"to_translate_{args.split}.tsv")
		export_tsv(source_rows, path)
		return
	if args.mode == "import":
		if not args.file:
			raise SystemExit("Укажите --file с переводом")
		translated = import_tsv(source_rows, args.file, args.column)
	else:
		translated = translate_with_model(source_rows, args.lang)

	problems = check_translations(source_rows, translated, args.lang)
	rows_to_save = []
	dropped = 0
	for row in translated:
		if args.drop_flagged and row["id"] in problems:
			dropped += 1
			continue
		rows_to_save.append({"id": row["id"], "question": row["question"]})
	common.write_jsonl(out_path, rows_to_save)
	if args.drop_flagged:
		print(f"Подозрительных переводов отброшено: {dropped}, сохранено вопросов: {len(rows_to_save)}")
	sources_by_id = {}
	for row in source_rows:
		sources_by_id[row["id"]] = row
	sources_kept = []
	for row in translated:
		sources_kept.append(sources_by_id[row["id"]])
	write_review(review_path, sources_kept, translated, problems)
	print(f"Готово: {out_path}")
	print(f"Оригинал и перевод рядом: {review_path}")
	report_problems(problems, len(translated), review_path)


if __name__ == "__main__":
	main()
