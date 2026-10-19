"""Шаг 5 для ~short: ответы на тестовые вопросы во всех условиях.

Условия:
	base      — просто вопрос;
	neologism — вопрос + «Give me a ~short answer» / «Дай ~short ответ» (обученный вектор);
	target    — вопрос + явная просьба ответить не более чем в двух предложениях;
	control   — как neologism, но со стартовым, необученным вектором.

Дополнительные условия (запускаются явно через --conditions, вектор не нужен):
	target_en    — вопрос на любом языке + английская просьба «Answer in at most two sentences.»;
	short_en     — вопрос + «Give me a short answer.» (шаблон неологизма с обычным английским словом);
	short_native — вопрос + «Дай короткий ответ.» / «Маған қысқа жауап бер.» (то же на языке вопроса).
Они отвечают на вопрос, нужен ли неологизм, если можно просто дописать английскую или обычную просьбу.

Запуск:  python 05_evaluate.py --run b002 --langs en ru kk
Быстрая проверка: python 05_evaluate.py --run b002 --langs en --limit 20 --out quick.jsonl

Результат: runs/{run}/generations.jsonl (по строке на ответ). Скрипт дописывает файл
и пропускает уже готовые ответы, поэтому его можно прервать и запустить снова.
"""

import argparse
import os

import torch

import config
from neolib import common, metrics

CONDITIONS = ["base", "neologism", "target", "control"]
EXTRA_CONDITIONS = ["target_en", "short_en", "short_native"]
# Условия, ответы на которые не зависят от обученного вектора: их можно взять из другого запуска.
VECTOR_FREE = ["base", "target", "target_en", "short_en", "short_native"]
# Для английского эти условия совпадают с target и short_en — их не считаем.
SAME_AS_OTHER_FOR_EN = ["target_en", "short_native"]
CHUNK = 64


def suffix_for(condition, lang):
	"""Приписка к вопросу для условия проверки."""
	if condition == "base":
		return ""
	if condition == "target":
		return common.target_suffix(lang)
	if condition == "target_en":
		return common.target_suffix("en")
	if condition == "short_en":
		return config.TEMPLATES["en"]["short"]
	if condition == "short_native":
		return config.TEMPLATES[lang]["short"]
	return common.neologism_suffix(lang)


def reuse_answers(args, saved, out_path, done):
	"""Копирует из другого запуска ответы, которые не зависят от обученного вектора.
	Генерация жадная, модель и вопросы те же, поэтому ответы были бы точно такими же."""
	other_saved = common.load_neologism(common.run_path(args.reuse_from, "neologism.pt"))
	reusable = list(VECTOR_FREE)
	if torch.equal(other_saved["init_vector"], saved["init_vector"]):
		reusable.append("control")
	else:
		print("Стартовые векторы разные — control посчитаю заново.")
	rows = []
	for row in common.read_jsonl(common.run_path(args.reuse_from, args.out)):
		key = (row["id"], row["lang"], row["condition"])
		if row["lang"] not in args.langs or row["condition"] not in args.conditions:
			continue
		if row["condition"] not in reusable or key in done:
			continue
		if args.limit and int(row["id"].split("-")[-1]) >= args.limit:
			continue
		rows.append(row)
		done.add(key)
	common.append_jsonl(out_path, rows)
	print(f"Взято готовых ответов из {args.reuse_from}: {len(rows)} (условия: {', '.join(reusable)})")


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--run", required=True)
	parser.add_argument("--langs", nargs="+", default=config.LANGS)
	parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS + EXTRA_CONDITIONS)
	parser.add_argument("--limit", type=int, default=0, help="взять только первые N вопросов")
	parser.add_argument("--out", default="generations.jsonl")
	parser.add_argument("--vector", default="neologism.pt", help="файл вектора внутри папки запуска")
	parser.add_argument("--reuse-from", help="запуск, из которого взять готовые ответы условий без обученного вектора "
		"(base, target, target_en, short_en, short_native; control — если стартовый вектор тот же)")
	args = parser.parse_args()
	common.set_seed(config.SEED)

	saved = common.load_neologism(common.run_path(args.run, args.vector))
	vectors = {
		"neologism": saved["vector"],
		"control": saved["init_vector"],
	}
	out_path = common.run_path(args.run, args.out)

	done = set()
	if os.path.exists(out_path):
		for row in common.read_jsonl(out_path):
			done.add((row["id"], row["lang"], row["condition"]))
		print(f"Уже готово ответов: {len(done)}")

	if args.reuse_from:
		reuse_answers(args, saved, out_path, done)

	model, tokenizer, neo_id = common.load_model_and_tokenizer()

	for lang in args.langs:
		questions_path = common.data_path(f"questions_test_{lang}.jsonl")
		if not os.path.exists(questions_path):
			print(f"Нет {questions_path} — сначала переведите вопросы (02_translate.py). Пропускаю {lang}.")
			continue
		questions = common.read_jsonl(questions_path)
		if args.limit:
			questions = questions[:args.limit]
		for condition in args.conditions:
			if lang == "en" and condition in SAME_AS_OTHER_FOR_EN:
				continue
			todo = []
			for row in questions:
				if (row["id"], lang, condition) not in done:
					todo.append(row)
			if not todo:
				continue
			print(f"\n{lang} / {condition}: осталось {len(todo)} вопросов")
			suffix = suffix_for(condition, lang)
			vector = vectors.get(condition)
			for start in range(0, len(todo), CHUNK):
				chunk = todo[start:start + CHUNK]
				texts = []
				for row in chunk:
					texts.append(common.build_user_text(row["question"], suffix))
				results = common.generate_many(
					model, tokenizer, texts, neo_id, vector,
					config.EVAL_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc=f"{lang}/{condition}",
				)
				rows = []
				for row, result in zip(chunk, results):
					rows.append({
						"id": row["id"],
						"lang": lang,
						"condition": condition,
						"question": row["question"],
						"suffix": suffix,
						"response": result["text"],
						"words": metrics.count_words(result["text"]),
						"truncated": result["truncated"],
						"detected_lang": metrics.detect_language(result["text"]),
					})
				common.append_jsonl(out_path, rows)
	print(f"\nГотово: {out_path}")
	print(f"Дальше: python 06_analyze.py --run {args.run}" + ("" if args.out == "generations.jsonl" else f" --file {args.out}"))


if __name__ == "__main__":
	main()
