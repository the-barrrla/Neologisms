"""Общие функции для проверок с готовыми векторами ~card (скрипты 14–18).

Векторы берутся из runs/{пробег}/neologism.pt (обученный вектор и стартовый), ответы без приписки
и с явной инструкцией — из runs/{пробег}/generations.jsonl: они нужны для доли закрытого разрыва.
"""

import math
import os

import torch

import config
from neolib import card, common, files, metrics

LANGS = ["en", "ru", "kk"]
MEASURES = ["count", "first", "rest", "plain", "full", "score"]
MEASURE_NAMES = {
	"count": "ровно 4 предложения",
	"first": "1-е ≤ 20 слов",
	"rest": "остальные ≤ 15 слов",
	"plain": "без разметки",
	"full": "все 4 условия",
	"score": "балл",
}


def use_card_token():
	"""Все проверки подставляют вектор на место токена ~card в шаблоне «Give me a ~card answer.»."""
	config.NEOLOGISM = config.CARD_NEOLOGISM


def load_vectors(run):
	"""(обученный вектор, стартовый вектор, настройки) пробега."""
	saved = common.load_neologism(common.run_path(run, "neologism.pt"))
	return saved["vector"].float(), saved["init_vector"].float(), saved["meta"]


def measures(text):
	"""Меры формата одного ответа: 1.0 или 0.0 по каждому условию, «все 4» и балл."""
	result = card.check(text)
	values = {"score": result["score"]}
	if result["full"]:
		values["full"] = 1.0
	else:
		values["full"] = 0.0
	for name in card.CRITERIA:
		if result[name]:
			values[name] = 1.0
		else:
			values[name] = 0.0
	return values


def mean_measures(texts):
	"""Средние меры формата по списку ответов."""
	sums = {}
	for name in MEASURES:
		sums[name] = 0.0
	for text in texts:
		values = measures(text)
		for name in MEASURES:
			sums[name] += values[name]
	result = {}
	for name in MEASURES:
		if texts:
			result[name] = sums[name] / len(texts)
		else:
			result[name] = float("nan")
	return result


def test_questions(lang, limit):
	"""Тестовые вопросы языка; при limit — только первые limit."""
	rows = files.read_jsonl(common.data_path(f"questions_test_{lang}.jsonl"))
	if limit:
		rows = rows[:limit]
	return rows


def saved_answers(run, condition):
	"""{(язык, id): ответ} из generations.jsonl пробега для одного условия."""
	answers = {}
	path = common.run_path(run, "generations.jsonl")
	if not os.path.exists(path):
		return answers
	for row in files.read_jsonl(path):
		if row["condition"] == condition:
			answers[(row["lang"], row["id"])] = row["response"]
	return answers


def reference_levels(run, lang, question_ids):
	"""Средние меры без приписки и с явной инструкцией на тех же вопросах (для доли разрыва)."""
	levels = {}
	for condition in ["base", "target"]:
		answers = saved_answers(run, condition)
		texts = []
		for question_id in question_ids:
			if (lang, question_id) in answers:
				texts.append(answers[(lang, question_id)])
		levels[condition] = mean_measures(texts)
	return levels


def closure(base, value, target):
	"""(значение − без приписки) / (явная инструкция − без приписки); nan, если разрыва почти нет."""
	return card.rise_closure(base, value, target)


def load_cache(path):
	"""{(вариант, язык, id): ответ} из уже сгенерированного файла — чтобы прерванную проверку можно было продолжить."""
	done = {}
	if os.path.exists(path):
		for row in files.read_jsonl(path):
			done[(row["variant"], row["lang"], row["id"])] = row["response"]
	return done


def generate_variant(model, tokenizer, neo_id, vector, variant, lang, questions, suffix, cache, cache_path, max_new_tokens):
	"""Ответы на вопросы с данным вектором; готовые берутся из кэша, новые дописываются в cache_path.
	Возвращает список ответов в порядке вопросов."""
	todo = []
	for row in questions:
		if (variant, lang, row["id"]) not in cache:
			todo.append(row)
	if todo:
		texts = []
		for row in todo:
			texts.append(common.build_user_text(row["question"], suffix))
		results = common.generate_many(
			model, tokenizer, texts, neo_id, vector,
			max_new_tokens, config.GEN_BATCH_SIZE, desc=f"{variant}/{lang}",
		)
		new_rows = []
		for row, result in zip(todo, results):
			cache[(variant, lang, row["id"])] = result["text"]
			new_rows.append({
				"variant": variant,
				"lang": lang,
				"id": row["id"],
				"suffix": suffix,
				"response": result["text"],
				"words": metrics.count_words(result["text"]),
				"truncated": result["truncated"],
				"detected_lang": metrics.detect_language(result["text"]),
			})
		files.append_jsonl(cache_path, new_rows)
	answers = []
	for row in questions:
		answers.append(cache[(variant, lang, row["id"])])
	return answers


def summarize(variant, lang, answers, levels):
	"""Строка итоговой таблицы: средние меры и доля закрытого разрыва по каждой мере."""
	values = mean_measures(answers)
	row = {"variant": variant, "lang": lang, "n": len(answers)}
	for name in MEASURES:
		row[name] = values[name]
		row[name + "_closure"] = closure(levels["base"][name], values[name], levels["target"][name])
	return row


def fmt(value):
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return "—"
	return f"{value * 100:.0f}".replace("-", "−")


def print_table(rows, title):
	print(f"\n{title}")
	header = f"  {'вариант':34}{'язык':6}"
	for name in MEASURES:
		header += f"{name:>8}"
	header += f"{'доля по баллу':>15}"
	print(header)
	for row in rows:
		line = f"  {row['variant']:34}{row['lang']:6}"
		for name in MEASURES:
			line += f"{fmt(row[name]):>8}"
		line += f"{fmt(row['score_closure']) + '%':>15}"
		print(line)


def write_rows(rows, path):
	import csv
	columns = ["variant", "lang", "n"]
	for name in MEASURES:
		columns.append(name)
	for name in MEASURES:
		columns.append(name + "_closure")
	extra = []
	for row in rows:
		for key in row:
			if key not in columns and key not in extra:
				extra.append(key)
	columns = columns + extra
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(columns)
		for row in rows:
			line = []
			for column in columns:
				value = row.get(column, "")
				if isinstance(value, float):
					value = round(value, 4)
				line.append(value)
			writer.writerow(line)


def word_vector(model, tokenizer, word):
	"""Эмбеддинг обычного слова (одного токена), как стартовый вектор в 04_train.py."""
	if not word.startswith(" "):
		word = " " + word
	ids = common.encode(tokenizer, word)
	if ids.shape[0] != 1:
		raise SystemExit(f"Слово {word!r} режется на {ids.shape[0]} токена — возьмите другое.")
	return common.initial_vector(model, tokenizer, word).float().cpu()


def as_tensor(vector):
	return torch.as_tensor(vector).float()
