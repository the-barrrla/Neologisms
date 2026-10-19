"""Проверка: что модель сама говорит о выученном слове (self-verbalization, как в исходной статье).

С каждым вектором модели задаются вопросы на трёх языках:
	«Человек пишет: „Дай ~card ответ.“ Какой ответ он хочет?» (тот же контекст, что при обучении),
	«Что значит слово ~card?» и «Назови пять синонимов слова ~card».
Слово в вопросе — тот же токен, на место которого подставляется вектор, поэтому модель «видит» именно его.
Обученный вектор очень близок к стартовому слову (косинус 0,86–0,95), поэтому модель может описывать
просто стартовое слово. Чтобы это отличить, те же вопросы задаются со стартовыми (необученными) векторами:
если ответы с обученным и стартовым вектором одинаковые, выученное словами не выражается.
Генерация короткая, около 5 минут.

Запуск: python 17_self_describe.py
	--runs card_en card_concise card_concise_lr003 b002   (по умолчанию)
Результат: analysis/self_describe.md (для чтения) и analysis/self_describe.jsonl (для 18_plugin_eval.py)
"""

import argparse
import os

import torch

import config
from neolib import common, files, probe

DEFAULT_RUNS = ["card_en", "card_concise", "card_concise_lr003", "b002"]
# request — вопрос в том же контексте, в каком слово учили («Give me a ~card answer.»): там выученный сдвиг
# работает наверняка. meaning и synonyms — слово отдельно, в кавычках, как в исходной статье.
QUESTIONS = {
	"request": {
		"en": "Someone writes: \"{request}\" What kind of answer do they want? Describe it in one sentence.",
		"ru": "Человек пишет: «{request}» Какой ответ он хочет? Опиши одним предложением.",
		"kk": "Біреу: «{request}» деп жазды. Ол қандай жауап қалайды? Бір сөйлеммен сипатта.",
	},
	"meaning": {
		"en": "What does the word \"{neo}\" mean? Answer in one sentence.",
		"ru": "Что значит слово «{neo}»? Ответь одним предложением.",
		"kk": "«{neo}» сөзі нені білдіреді? Бір сөйлеммен жауап бер.",
	},
	"synonyms": {
		"en": "Give five synonyms for the word \"{neo}\". Write only the words, separated by commas.",
		"ru": "Назови пять синонимов слова «{neo}». Напиши только слова через запятую.",
		"kk": "«{neo}» сөзіне бес синоним жаз. Тек сөздерді үтір арқылы жаз.",
	},
}
MAX_NEW_TOKENS = 120


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
	args = parser.parse_args()
	common.set_seed(config.SEED)
	probe.use_card_token()

	vectors = []
	seen_starts = []
	for run in args.runs:
		vector, init, meta = probe.load_vectors(run)
		vectors.append((f"{run} (обученный)", run, "trained", vector))
		start_known = False
		for known in seen_starts:
			if torch.equal(known, init):
				start_known = True
		if not start_known:
			seen_starts.append(init)
			vectors.append((f"старт {meta.get('init_word', '').strip()!r} (без обучения)", run, "start", init))

	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	records = []
	for name, run, kind, vector in vectors:
		for question_kind, by_lang in QUESTIONS.items():
			texts = []
			langs = []
			for lang in probe.LANGS:
				request = config.TEMPLATES[lang]["neologism"].format(neo=config.NEOLOGISM)
				text = by_lang[lang].format(neo=config.NEOLOGISM, request=request)
				ids = common.encode(tokenizer, common.chat_prompt(tokenizer, text))
				if common.count_token(ids, neo_id) != 1:
					raise SystemExit(f"В вопросе {text!r} слово {config.NEOLOGISM} не стало одним токеном")
				texts.append(text)
				langs.append(lang)
			results = common.generate_many(model, tokenizer, texts, neo_id, vector, MAX_NEW_TOKENS, len(texts))
			for lang, text, result in zip(langs, texts, results):
				records.append({
					"name": name, "run": run, "kind": kind, "question": question_kind,
					"lang": lang, "prompt": text, "answer": result["text"],
				})
				print(f"[{name} | {question_kind} | {lang}] {result['text'][:160]!r}")

	os.makedirs("analysis", exist_ok=True)
	files.write_jsonl(os.path.join("analysis", "self_describe.jsonl"), records)
	lines = ["# Что модель говорит о выученном слове", ""]
	lines.append("Вопросы задаются с токеном ~card, на место которого подставлен вектор. «Без обучения» — стартовый вектор.")
	current = None
	for record in records:
		if record["name"] != current:
			current = record["name"]
			lines.append("")
			lines.append(f"## {current}")
		answer = record["answer"].replace("\n", " ").strip()
		lines.append(f"- **{record['question']}, {record['lang']}:** {answer}")
	path = os.path.join("analysis", "self_describe.md")
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nСохранено: {path}, analysis/self_describe.jsonl")


if __name__ == "__main__":
	main()
