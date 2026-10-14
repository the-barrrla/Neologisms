"""Шаг 3 для второго неологизма ~card: пары «ответ-карточка — не карточка».

Ответ-карточка: ровно 4 предложения, первое ≤ 20 слов, остальные ≤ 15, без разметки
(проверка — neolib/card.py, инструкция — CARD_TEMPLATES в config.py).

На каждый обучающий вопрос:
	хороший ответ (chosen) — модель отвечает с длинной явной инструкцией; ответ берётся,
		только если прошёл все 4 проверки. Если не прошёл, модель один раз переписывает
		свой ответ по той же инструкции, и проверка повторяется;
	плохой ответ (rejected) — наполовину «почти правильный» (короткий ответ на «ответь кратко,
		можно списками» с нарушением формата), наполовину обычный длинный ответ.
		Без «почти правильных» вектор выучил бы просто «покороче», а не формат.

Обычные ответы берутся из data/raw_pairs_{lang}.jsonl (шаг 3 для ~short), если он есть.
Сырые ответы сохраняются в data/raw_card_{lang}.jsonl; перефильтровать без генерации:
	python 08_card_pairs.py --lang en --refilter

Запуск: python 08_card_pairs.py --lang en   (около получаса)
Быстрая проба: python 08_card_pairs.py --lang en --limit 32
Результат: data/pairs_card_{lang}.jsonl
"""

import argparse
import os

import config
from neolib import card, common, metrics

DRAFT_MAX_CHARS = 2500
REWRITE = {
	"en": ("Here is a draft answer to this question:", "Rewrite the draft. Output only the new answer."),
	"ru": ("Вот черновик ответа на этот вопрос:", "Перепиши черновик. Выведи только новый ответ."),
	"kk": ("Бұл сұраққа жауаптың жобасы:", "Жобаны қайта жаз. Тек жаңа жауапты шығар."),
}


def card_target(lang):
	return config.CARD_TEMPLATES[lang]["target"]


def rewrite_text(question, draft, lang):
	label, line = REWRITE[lang]
	draft = draft.strip()[:DRAFT_MAX_CHARS]
	return question.strip() + "\n\n" + label + "\n\n" + draft + "\n\n" + line + " " + card_target(lang)


def load_plain_answers(lang):
	"""Обычные ответы из шага 3 для ~short: {id: (текст, обрезан ли)}."""
	path = common.data_path(f"raw_pairs_{lang}.jsonl")
	answers = {}
	if not os.path.exists(path):
		return answers
	for row in common.read_jsonl(path):
		answers[row["id"]] = (row["rejected"], row["rejected_truncated"])
	return answers


def is_good_card(text, truncated, lang):
	if truncated or not text:
		return False
	if metrics.detect_language(text) != lang:
		return False
	return card.check(text)["full"]


def generate_raw(questions, lang):
	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	plain = load_plain_answers(lang)
	missing = []
	for row in questions:
		if row["id"] not in plain:
			missing.append(row)
	print(f"Обычных ответов из raw_pairs_{lang}.jsonl: {len(questions) - len(missing)}, сгенерировать: {len(missing)}")
	if missing:
		texts = []
		for row in missing:
			texts.append(common.build_user_text(row["question"], ""))
		results = common.generate_many(
			model, tokenizer, texts, neo_id, None,
			config.PAIR_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc="обычные",
		)
		for row, result in zip(missing, results):
			plain[row["id"]] = (result["text"], result["truncated"])

	print("Генерирую ответы-карточки по явной инструкции (будущие хорошие примеры) ...")
	texts = []
	for row in questions:
		texts.append(common.build_user_text(row["question"], card_target(lang)))
	cards = common.generate_many(
		model, tokenizer, texts, neo_id, None,
		config.CARD_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc="карточки",
	)

	print("Переписываю ответы, не прошедшие проверку ...")
	redo = []
	redo_texts = []
	for i, row in enumerate(questions):
		if is_good_card(cards[i]["text"], cards[i]["truncated"], lang):
			continue
		draft = cards[i]["text"]
		if not draft:
			draft = plain[row["id"]][0]
		redo.append(i)
		redo_texts.append(rewrite_text(row["question"], draft, lang))
	rewrites = {}
	if redo:
		results = common.generate_many(
			model, tokenizer, redo_texts, neo_id, None,
			config.CARD_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc="переписанные",
		)
		for i, result in zip(redo, results):
			rewrites[i] = result

	print("Генерирую «почти правильные» ответы (будущие плохие примеры) ...")
	texts = []
	for row in questions:
		texts.append(common.build_user_text(row["question"], config.CARD_TEMPLATES[lang]["near_miss"]))
	near = common.generate_many(
		model, tokenizer, texts, neo_id, None,
		config.NEAR_MISS_MAX_NEW_TOKENS, config.GEN_BATCH_SIZE, desc="почти правильные",
	)

	raw = []
	for i, row in enumerate(questions):
		rewrite = rewrites.get(i, {"text": "", "truncated": False})
		raw.append({
			"id": row["id"],
			"question": row["question"],
			"plain": plain[row["id"]][0],
			"plain_truncated": plain[row["id"]][1],
			"card": cards[i]["text"],
			"card_truncated": cards[i]["truncated"],
			"rewrite": rewrite["text"],
			"rewrite_truncated": rewrite["truncated"],
			"near_miss": near[i]["text"],
			"near_miss_truncated": near[i]["truncated"],
		})
	print(common.gpu_memory_report())
	return raw


def usable_near_miss(row, lang):
	"""«Почти правильный» ответ годится, если он на нужном языке, не обрезан и НЕ карточка."""
	text = row["near_miss"]
	if row["near_miss_truncated"] or not text:
		return False
	if metrics.detect_language(text) != lang:
		return False
	return not card.check(text)["full"]


def usable_plain(row, lang):
	text = row["plain"]
	if not text:
		return False
	if metrics.detect_language(text) != lang:
		return False
	return not card.check(text)["full"]


def filter_pairs(raw, lang):
	pairs = []
	reasons = {"нет годной карточки": 0, "нет годного плохого ответа": 0}
	n_near = 0
	for row in raw:
		chosen = None
		source = None
		if is_good_card(row["card"], row["card_truncated"], lang):
			chosen = row["card"]
			source = "model"
		elif is_good_card(row["rewrite"], row["rewrite_truncated"], lang):
			chosen = row["rewrite"]
			source = "rewrite"
		if chosen is None:
			reasons["нет годной карточки"] += 1
			continue
		near_ok = usable_near_miss(row, lang)
		plain_ok = usable_plain(row, lang)
		# Держим долю «почти правильных» около CARD_NEAR_MISS_SHARE.
		want_near = n_near < config.CARD_NEAR_MISS_SHARE * (len(pairs) + 1)
		if near_ok and (want_near or not plain_ok):
			rejected = row["near_miss"]
			kind = "near_miss"
			n_near += 1
		elif plain_ok:
			rejected = row["plain"]
			kind = "plain"
		else:
			reasons["нет годного плохого ответа"] += 1
			continue
		pairs.append({
			"id": row["id"],
			"question": row["question"],
			"chosen": chosen,
			"rejected": rejected,
			"chosen_source": source,
			"rejected_kind": kind,
			"chosen_words": metrics.count_words(chosen),
			"rejected_words": metrics.count_words(rejected),
		})
	return pairs, reasons


def print_stats(raw, pairs, reasons, lang):
	print(f"\nВсего вопросов: {len(raw)}, оставлено пар: {len(pairs)}")
	for reason, count in reasons.items():
		print(f"  отброшено — {reason}: {count}")
	first_pass = 0
	failed = {}
	for name in card.CRITERIA:
		failed[name] = 0
	for row in raw:
		if is_good_card(row["card"], row["card_truncated"], lang):
			first_pass += 1
		result = card.check(row["card"])
		for name in card.failed_criteria(result):
			failed[name] += 1
	print(f"  явная инструкция с первого раза дала карточку: {first_pass} из {len(raw)} ({100 * first_pass / max(1, len(raw)):.0f}%)")
	print("  что чаще всего нарушалось с первого раза:")
	for name in card.CRITERIA:
		print(f"    {card.CRITERION_NAMES[name]}: {failed[name]}")
	sources = {"model": 0, "rewrite": 0}
	kinds = {"near_miss": 0, "plain": 0}
	chosen_lengths = []
	near_lengths = []
	plain_lengths = []
	for pair in pairs:
		sources[pair["chosen_source"]] += 1
		kinds[pair["rejected_kind"]] += 1
		chosen_lengths.append(pair["chosen_words"])
		if pair["rejected_kind"] == "near_miss":
			near_lengths.append(pair["rejected_words"])
		else:
			plain_lengths.append(pair["rejected_words"])
	print(f"  хорошие ответы: с первого раза {sources['model']}, после переписывания {sources['rewrite']}")
	print(f"  плохие ответы: «почти правильные» {kinds['near_miss']}, обычные {kinds['plain']}")
	print(f"  средняя длина: карточка {metrics.mean(chosen_lengths):.0f} слов, "
		f"«почти правильный» {metrics.mean(near_lengths):.0f}, обычный {metrics.mean(plain_lengths):.0f}")
	if len(pairs) < 300:
		print("  ВНИМАНИЕ: пар меньше 300 — слово может выучиться плохо.")
	print("\nПримеры пар:")
	shown = {"near_miss": 0, "plain": 0}
	for pair in pairs:
		if shown[pair["rejected_kind"]] >= 2:
			continue
		shown[pair["rejected_kind"]] += 1
		print("-" * 60)
		print("ВОПРОС:", pair["question"][:200])
		print(f"ХОРОШИЙ ({pair['chosen_source']}):", pair["chosen"][:400])
		print(f"ПЛОХОЙ ({pair['rejected_kind']}):", pair["rejected"][:300], "...")


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", default=config.LANGS[0])
	parser.add_argument("--limit", type=int, default=0, help="взять только первые N вопросов")
	parser.add_argument("--refilter", action="store_true", help="не генерировать, только заново отфильтровать")
	args = parser.parse_args()
	common.set_seed(config.SEED)

	raw_path = common.data_path(f"raw_card_{args.lang}.jsonl")
	out_path = common.data_path(f"pairs_card_{args.lang}.jsonl")

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

	pairs, reasons = filter_pairs(raw, args.lang)
	common.write_jsonl(out_path, pairs)
	print_stats(raw, pairs, reasons, args.lang)
	print(f"\nСохранено: {out_path}")
	print(f"Дальше: python 04_train.py --concept card --lang {args.lang} --run card_{args.lang}")


if __name__ == "__main__":
	main()
