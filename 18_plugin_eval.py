"""Проверка: работает ли собственное описание слова вместо самого слова (plug-in, как в исходной статье).

Берётся ответ модели из 17_self_describe.py (на каждом языке свой): по умолчанию на вопрос
«Человек пишет: „Дай ~card ответ.“ Какой ответ он хочет?», с --question meaning — на «Что значит слово ~card?»,
и подставляется в промпт вместо неологизма: «Ответь в соответствии с этим описанием: <описание>».
Если описание даёт тот же эффект, что и вектор, модель «понимает», что выучила, и может это выразить словами.
Если эффект намного слабее — выученное нельзя пересказать словами (или модель описывает его неверно).
Для сравнения рядом печатается эффект самого вектора на тех же вопросах.

Запуск (после 17_self_describe.py): python 18_plugin_eval.py
	--runs card_en card_concise card_concise_lr003   (по умолчанию)
	--limit 50
Около 15–20 минут. Прерванный запуск продолжится с места остановки.
Результат: analysis/plugin_eval.jsonl (ответы), analysis/plugin_eval.csv
"""

import argparse
import os

import config
from neolib import common, files, probe

DEFAULT_RUNS = ["card_en", "card_concise", "card_concise_lr003"]
TEMPLATES = {
	"en": "Answer according to this description: {desc}",
	"ru": "Ответь в соответствии с этим описанием: {desc}",
	"kk": "Осы сипаттамаға сәйкес жауап бер: {desc}",
}
# Чем заменить само слово, если описание его упоминает: вектора в этой проверке нет.
PLACEHOLDER = {"en": "this", "ru": "это", "kk": "бұл"}


def load_descriptions(question):
	"""Описания обученных векторов из 17_self_describe.py: (пробег, язык) → текст."""
	path = os.path.join("analysis", "self_describe.jsonl")
	if not os.path.exists(path):
		raise SystemExit("Нет analysis/self_describe.jsonl — сначала запустите 17_self_describe.py")
	descriptions = {}
	for row in files.read_jsonl(path):
		if row["kind"] == "trained" and row["question"] == question:
			descriptions[(row["run"], row["lang"])] = row["answer"].replace("\n", " ").strip()
	return descriptions


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
	parser.add_argument("--limit", type=int, default=50)
	parser.add_argument("--question", default="request", choices=["request", "meaning"],
		help="какой ответ из 17_self_describe.py брать как описание")
	parser.add_argument("--max-new-tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
	args = parser.parse_args()
	common.set_seed(config.SEED)
	probe.use_card_token()
	descriptions = load_descriptions(args.question)

	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	os.makedirs("analysis", exist_ok=True)
	cache_path = os.path.join("analysis", "plugin_eval.jsonl")
	cache = probe.load_cache(cache_path)
	rows = []
	for run in args.runs:
		for lang in probe.LANGS:
			if (run, lang) not in descriptions:
				print(f"Нет описания для {run}/{lang} — пропускаю")
				continue
			description = descriptions[(run, lang)].replace(config.NEOLOGISM, PLACEHOLDER[lang])
			suffix = TEMPLATES[lang].format(desc=description)
			questions = probe.test_questions(lang, args.limit)
			question_ids = []
			for row in questions:
				question_ids.append(row["id"])
			levels = probe.reference_levels(run, lang, question_ids)
			print(f"\n{run}/{lang}: {suffix[:200]!r}")
			variant = f"{run}: описание ({args.question})"
			answers = probe.generate_variant(model, tokenizer, neo_id, None, variant, lang,
				questions, suffix, cache, cache_path, args.max_new_tokens)
			row = probe.summarize(variant, lang, answers, levels)
			row["description"] = description
			rows.append(row)
			saved = probe.saved_answers(run, "neologism")
			vector_answers = []
			for question_id in question_ids:
				if (lang, question_id) in saved:
					vector_answers.append(saved[(lang, question_id)])
			if len(vector_answers) == len(question_ids):
				rows.append(probe.summarize(f"{run}: сам вектор", lang, vector_answers, levels))

	probe.print_table(rows, "Описание вместо слова против самого вектора")
	csv_path = os.path.join("analysis", "plugin_eval.csv")
	probe.write_rows(rows, csv_path)
	print(f"\nСохранено: {cache_path}, {csv_path}")


if __name__ == "__main__":
	main()
