"""Шаг 5 для ~card: ответы на тестовые вопросы во всех условиях.

Условия:
	base         — просто вопрос;
	neologism    — вопрос + «Give me a ~card answer.» / «Дай ~card ответ.» (обученный вектор);
	target       — вопрос + полная явная инструкция «ответ-карточка» на языке вопроса (это 100%);
	target_en    — та же инструкция по-английски внутри русского или казахского промпта;
	four         — частичная просьба «Answer in four sentences.» / «Ответь в четырёх предложениях.»;
	short_en     — «Give me a short answer.» (проверка: не сводится ли ~card к краткости);
	short_native — «Дай короткий ответ.» / «Маған қысқа жауап бер.»;
	control      — как neologism, но со стартовым, необученным вектором.

base, short_en, short_native и control те же, что в основном опыте с ~short (тот же текст
промпта, тот же стартовый вектор), поэтому их можно взять готовыми: --reuse-from b002.
Из другого запуска ~card берутся ещё target, target_en и four, а control — если стартовое слово то же.
Можно указать несколько запусков: --reuse-from card_en b002.

Запуск: python 09_card_evaluate.py --run card_en --reuse-from b002
Повторное обучение ~card: python 09_card_evaluate.py --run card_concise --reuse-from card_en
Результат: runs/{run}/generations.jsonl; дальше — python 10_card_analyze.py --run card_en
"""

import argparse
import os

import torch

import config
from neolib import card, common, metrics

CONDITIONS = ["base", "neologism", "target", "target_en", "four", "short_en", "short_native", "control"]
# Для английского эти условия совпадают с target и short_en — их не считаем.
SAME_AS_OTHER_FOR_EN = ["target_en", "short_native"]
# Условия без обученного вектора: их можно взять из другого запуска, если текст приписки совпадает
# до буквы (из запуска ~short подходят base, short_en, short_native; из запуска ~card — ещё target,
# target_en и four). Совпадение текста проверяется, так что чужая «явная просьба» не подмешается.
REUSABLE = ["base", "target", "target_en", "four", "short_en", "short_native"]
NEOLOGISM_TOKENS = ["~short", config.CARD_NEOLOGISM]
CHUNK = 64


def suffix_for(condition, lang):
	"""Приписка к вопросу для условия проверки."""
	if condition == "base":
		return ""
	if condition == "target":
		return config.CARD_TEMPLATES[lang]["target"]
	if condition == "target_en":
		return config.CARD_TEMPLATES["en"]["target"]
	if condition == "four":
		return config.CARD_TEMPLATES[lang]["four"]
	if condition == "short_en":
		return config.TEMPLATES["en"]["short"]
	if condition == "short_native":
		return config.TEMPLATES[lang]["short"]
	return common.neologism_suffix(lang)


def control_suffixes(lang):
	"""Приписки, при которых control даёт те же числа: на месте ~short или ~card стоит стартовый вектор."""
	suffixes = []
	for token in NEOLOGISM_TOKENS:
		suffixes.append(config.TEMPLATES[lang]["neologism"].format(neo=token))
	return suffixes


def reuse_answers(args, other_run, saved, out_path, done):
	"""Копирует из другого запуска ответы, промпт которых совпадает с нашим до буквы.
	control берётся, если стартовые векторы равны: на месте ~short и ~card стоит один и тот же вектор,
	а остальной текст приписки одинаковый, так что модель получает те же числа."""
	other_saved = common.load_neologism(common.run_path(other_run, "neologism.pt"))
	same_init = torch.equal(other_saved["init_vector"], saved["init_vector"])
	if not same_init:
		print(f"У {other_run} другой стартовый вектор — control оттуда не беру.")
	rows = []
	counts = {}
	for row in common.read_jsonl(common.run_path(other_run, "generations.jsonl")):
		key = (row["id"], row["lang"], row["condition"])
		if row["lang"] not in args.langs or row["condition"] not in args.conditions or key in done:
			continue
		if args.limit and int(row["id"].split("-")[-1]) >= args.limit:
			continue
		if row["condition"] == "control":
			if not same_init:
				continue
			expected = control_suffixes(row["lang"])
		elif row["condition"] in REUSABLE:
			expected = [suffix_for(row["condition"], row["lang"])]
		else:
			continue
		if row.get("suffix", "") not in expected:
			continue
		rows.append(row)
		done.add(key)
		counts[row["condition"]] = counts.get(row["condition"], 0) + 1
	common.append_jsonl(out_path, rows)
	parts = []
	for name, count in counts.items():
		parts.append(f"{name} {count}")
	print(f"Взято готовых ответов из {other_run}: {len(rows)} ({', '.join(parts)})")


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--run", required=True)
	parser.add_argument("--langs", nargs="+", default=config.LANGS)
	parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
	parser.add_argument("--limit", type=int, default=0, help="взять только первые N вопросов")
	parser.add_argument("--reuse-from", nargs="+", default=[],
		help="запуски, из которых взять готовые ответы условий без обученного вектора (можно несколько)")
	args = parser.parse_args()
	common.set_seed(config.SEED)
	config.NEOLOGISM = config.CARD_NEOLOGISM

	saved = common.load_neologism(common.run_path(args.run, "neologism.pt"))
	if saved["meta"].get("neologism") != config.CARD_NEOLOGISM:
		raise SystemExit(f"В запуске {args.run} обучен {saved['meta'].get('neologism')}, а не {config.CARD_NEOLOGISM}. "
			"Обучите: python 04_train.py --concept card --lang en --run " + args.run)
	vectors = {
		"neologism": saved["vector"],
		"control": saved["init_vector"],
	}
	out_path = common.run_path(args.run, "generations.jsonl")

	done = set()
	if os.path.exists(out_path):
		for row in common.read_jsonl(out_path):
			done.add((row["id"], row["lang"], row["condition"]))
		print(f"Уже готово ответов: {len(done)}")

	for other_run in args.reuse_from:
		reuse_answers(args, other_run, saved, out_path, done)

	model, tokenizer, neo_id = common.load_model_and_tokenizer()

	for lang in args.langs:
		questions_path = common.data_path(f"questions_test_{lang}.jsonl")
		if not os.path.exists(questions_path):
			print(f"Нет {questions_path}. Пропускаю {lang}.")
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
					check = card.check(result["text"])
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
						"n_sentences": check["n_sentences"],
						"card_score": check["score"],
					})
				common.append_jsonl(out_path, rows)
	print(f"\nГотово: {out_path}")
	print(f"Дальше: python 10_card_analyze.py --run {args.run}")


if __name__ == "__main__":
	main()
