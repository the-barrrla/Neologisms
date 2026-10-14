"""Шаг 3. Пары «хороший ответ — плохой ответ» для обучения неологизма.

На каждый обучающий вопрос модель отвечает дважды:
	плохой ответ (rejected) — на сам вопрос;
	хороший ответ (chosen)  — на вопрос + «ответь не более чем в двух предложениях».
Затем программа оставляет только пары, где хороший ответ правда короткий
и оба ответа написаны на нужном языке.

Сырые ответы сохраняются в data/raw_pairs_{lang}.jsonl, поэтому фильтр можно
перезапустить без новой генерации:  python 03_make_pairs.py --refilter

Запуск: python 03_make_pairs.py --lang en
Быстрая проба на 32 вопросах: python 03_make_pairs.py --lang en --limit 32
Результат: data/pairs_{lang}.jsonl
"""

import argparse
import os

import config
from neolib import common, metrics


def generate_raw(questions, lang):
	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	plain_texts = []
	short_texts = []
	for row in questions:
		plain_texts.append(common.build_user_text(row["question"], ""))
		short_texts.append(common.build_user_text(row["question"], common.target_suffix(lang)))
	print("Генерирую обычные ответы (будущие плохие примеры) ...")
	plain = common.generate_many(
		model, tokenizer, plain_texts, neo_id, None,
		config.PAIR_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc="обычные",
	)
	print("Генерирую короткие ответы (будущие хорошие примеры) ...")
	short = common.generate_many(
		model, tokenizer, short_texts, neo_id, None,
		config.PAIR_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc="короткие",
	)
	raw = []
	for row, rejected, chosen in zip(questions, plain, short):
		raw.append({
			"id": row["id"],
			"question": row["question"],
			"chosen": chosen["text"],
			"chosen_truncated": chosen["truncated"],
			"rejected": rejected["text"],
			"rejected_truncated": rejected["truncated"],
		})
	print(common.gpu_memory_report())
	return raw


def is_short_enough(chosen_words, rejected_words):
	if chosen_words == 0:
		return False
	if chosen_words > config.CHOSEN_MAX_WORDS:
		return False
	return chosen_words <= config.CHOSEN_MAX_RATIO * rejected_words


def filter_pairs(raw, lang, fallback_truncate):
	pairs = []
	reasons = {"обрезан хороший": 0, "не тот язык": 0, "недостаточно короткий": 0}
	n_fallback = 0
	for row in raw:
		chosen = row["chosen"]
		rejected = row["rejected"]
		chosen_words = metrics.count_words(chosen)
		rejected_words = metrics.count_words(rejected)
		source = "model"
		ok = True
		if row["chosen_truncated"]:
			reasons["обрезан хороший"] += 1
			ok = False
		elif metrics.detect_language(chosen) != lang or metrics.detect_language(rejected) != lang:
			reasons["не тот язык"] += 1
			ok = False
		elif not is_short_enough(chosen_words, rejected_words):
			reasons["недостаточно короткий"] += 1
			ok = False
		if not ok and fallback_truncate and metrics.detect_language(rejected) == lang:
			# Запасной способ: хороший ответ = первые два предложения обычного ответа.
			candidate = metrics.first_sentences(rejected, 2)
			candidate_words = metrics.count_words(candidate)
			if is_short_enough(candidate_words, rejected_words):
				chosen = candidate
				chosen_words = candidate_words
				source = "truncated"
				ok = True
				n_fallback += 1
		if ok:
			pairs.append({
				"id": row["id"],
				"question": row["question"],
				"chosen": chosen,
				"rejected": rejected,
				"chosen_words": chosen_words,
				"rejected_words": rejected_words,
				"chosen_source": source,
			})
	return pairs, reasons, n_fallback


def print_stats(raw, pairs, reasons, n_fallback):
	print(f"\nВсего вопросов: {len(raw)}, оставлено пар: {len(pairs)}")
	for reason, count in reasons.items():
		print(f"  отброшено — {reason}: {count}")
	if n_fallback:
		print(f"  из них хороший ответ получен обрезкой до двух предложений: {n_fallback}")
	truncated_rejected = 0
	for row in raw:
		if row["rejected_truncated"]:
			truncated_rejected += 1
	print(f"  обычных ответов, обрезанных по длине: {truncated_rejected}")
	chosen_lengths = []
	rejected_lengths = []
	for pair in pairs:
		chosen_lengths.append(pair["chosen_words"])
		rejected_lengths.append(pair["rejected_words"])
	print(f"  средняя длина: хороший {metrics.mean(chosen_lengths):.0f} слов, плохой {metrics.mean(rejected_lengths):.0f} слов")
	print("\nТри примера пар:")
	for pair in pairs[:3]:
		print("-" * 60)
		print("ВОПРОС:", pair["question"][:200])
		print("ХОРОШИЙ:", pair["chosen"][:300])
		print("ПЛОХОЙ:", pair["rejected"][:300], "...")


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", default=config.LANGS[0])
	parser.add_argument("--limit", type=int, default=0, help="взять только первые N вопросов")
	parser.add_argument("--refilter", action="store_true", help="не генерировать, только заново отфильтровать")
	parser.add_argument("--fallback-truncate", action="store_true", help="запасной способ для неудачных пар")
	args = parser.parse_args()
	common.set_seed(config.SEED)

	raw_path = common.data_path(f"raw_pairs_{args.lang}.jsonl")
	out_path = common.data_path(f"pairs_{args.lang}.jsonl")

	if args.refilter:
		if not os.path.exists(raw_path):
			raise SystemExit(f"Нет файла {raw_path}: сначала запустите без --refilter")
		raw = common.read_jsonl(raw_path)
	else:
		questions = common.read_jsonl(common.data_path(f"questions_train_{args.lang}.jsonl"))
		if args.limit:
			questions = questions[:args.limit]
		raw = generate_raw(questions, args.lang)
		common.write_jsonl(raw_path, raw)

	pairs, reasons, n_fallback = filter_pairs(raw, args.lang, args.fallback_truncate)
	common.write_jsonl(out_path, pairs)
	print_stats(raw, pairs, reasons, n_fallback)
	print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
	main()
