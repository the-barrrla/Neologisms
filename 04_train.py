"""Шаг 4. Обучение вектора неологизма. Модель заморожена, учится только один вектор.

Запуск: python 04_train.py --lang en --run short_en
Второй неологизм ~card (пары из 08_card_pairs.py): python 04_train.py --concept card --lang en --run card_en
Быстрая проба: python 04_train.py --lang en --run test_run --limit 40 --epochs 1

Результат в runs/{run}/:
	neologism.pt   — обученный вектор (и стартовый, для контроля);
	log.csv        — потеря и другие показатели по ходу обучения;
	settings.json  — с какими настройками обучали.

Что смотреть в выводе:
	* Проверка на старте: потеря должна быть 1.386 (2·ln 2). Если нет — ошибка в подсчёте вероятностей.
	* «потеря» падает, «хороший−старт» растёт (хороший ответ становится вероятнее).
	* «норма» — длина вектора относительно обычных слов. Если улетает выше ~1.5–2,
	включите NORM_PENALTY в config.py.
	* После каждой эпохи печатаются 3 ответа со словом — они должны становиться короче.
"""

import argparse
import csv
import json
import math
import os
import random
import time

import torch

import config
from neolib import card, common, metrics


def build_examples(pairs, tokenizer, neo_id, lang):
	"""Токены промпта и обоих ответов для каждой пары; ответы обрезаются до MAX_RESPONSE_TOKENS."""
	suffix = common.neologism_suffix(lang)
	examples = []
	for pair in pairs:
		user_text = common.build_user_text(pair["question"], suffix)
		prompt_ids = common.encode(tokenizer, common.chat_prompt(tokenizer, user_text))
		if common.count_token(prompt_ids, neo_id) != 1:
			raise ValueError("Неологизм должен встречаться в промпте ровно один раз — ошибка в шаблоне TEMPLATES")
		chosen_ids = common.encode(tokenizer, common.chat_response_part(tokenizer, user_text, pair["chosen"]))
		rejected_ids = common.encode(tokenizer, common.chat_response_part(tokenizer, user_text, pair["rejected"]))
		examples.append({
			"id": pair["id"],
			"prompt_ids": prompt_ids,
			"chosen_ids": chosen_ids[:config.MAX_RESPONSE_TOKENS],
			"rejected_ids": rejected_ids[:config.MAX_RESPONSE_TOKENS],
		})
	return examples


def compute_reference(model, examples, neo_id, init_vector, cache_path):
	"""log p хорошего и плохого ответа со стартовым вектором — точка отсчёта для потери."""
	if os.path.exists(cache_path):
		cached = torch.load(cache_path)
		if len(cached) == len(examples):
			print(f"Беру готовые опорные вероятности из {cache_path}")
			return cached
	print("Считаю опорные вероятности со стартовым вектором ...")
	reference = []
	start = time.time()
	with torch.no_grad():
		for i, example in enumerate(examples):
			lc0 = common.response_logprob(model, example["prompt_ids"], example["chosen_ids"], neo_id, init_vector).item()
			lr0 = common.response_logprob(model, example["prompt_ids"], example["rejected_ids"], neo_id, init_vector).item()
			reference.append([lc0, lr0])
			if (i + 1) % 100 == 0:
				print(f"  {i + 1}/{len(examples)}, {time.time() - start:.0f} с")
	torch.save(reference, cache_path)
	return reference


def peek(model, tokenizer, neo_id, vector, lang, questions):
	"""Несколько ответов со словом — посмотреть глазами, укорачиваются ли они."""
	was_training = model.training
	model.eval()
	texts = []
	for question in questions:
		texts.append(common.build_user_text(question, common.neologism_suffix(lang)))
	results = common.generate_many(model, tokenizer, texts, neo_id, vector.detach(), 300, len(texts))
	for question, result in zip(questions, results):
		words = metrics.count_words(result["text"])
		label = f"{words} слов"
		if config.NEOLOGISM == config.CARD_NEOLOGISM:
			result_check = card.check(result["text"])
			failed = card.failed_criteria(result_check)
			if failed:
				label += f", {result_check['n_sentences']} предл., не выполнено: {', '.join(failed)}"
			else:
				label += ", карточка OK"
		print(f"    [{label}] {question[:60]!r} -> {result['text'][:150]!r}")
	if was_training:
		model.train()


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--lang", default=config.LANGS[0], help="на каком языке пары для обучения")
	parser.add_argument("--run", required=True, help="имя запуска, например short_en")
	parser.add_argument("--epochs", type=int, default=config.EPOCHS)
	parser.add_argument("--lr", type=float, default=config.LR)
	parser.add_argument("--limit", type=int, default=0, help="взять только первые N пар")
	parser.add_argument("--offset", type=int, default=0, help="сначала пропустить N пар (вместе с --limit даёт другой поднабор)")
	parser.add_argument("--seed", type=int, default=config.SEED, help="зерно случайности (порядок пар и т. п.)")
	parser.add_argument("--beta", type=float, default=config.BETA, help="крутизна потери APO-up (меньше — потеря позже насыщается)")
	parser.add_argument("--init-word", default=config.INIT_WORD, help="обычное слово, с вектора которого начинается обучение, например concise (пробел впереди добавится сам); random — случайный вектор типичной длины")
	parser.add_argument("--concept", default="short", choices=["short", "card"], help="какой неологизм учить: ~short или ~card")
	parser.add_argument("--grad-accum", type=int, default=config.GRAD_ACCUM, help="сколько пар на один шаг оптимизатора")
	args = parser.parse_args()
	if not args.init_word.startswith(" "):
		args.init_word = " " + args.init_word
	common.set_seed(args.seed)

	run_dir = os.path.join(config.RUNS_DIR, args.run)
	os.makedirs(run_dir, exist_ok=True)

	if args.concept == "card":
		config.NEOLOGISM = config.CARD_NEOLOGISM
		pairs_file = f"pairs_card_{args.lang}.jsonl"
	else:
		pairs_file = f"pairs_{args.lang}.jsonl"
	print(f"Неологизм: {config.NEOLOGISM}, пары: data/{pairs_file}")
	pairs = common.read_jsonl(common.data_path(pairs_file))
	if args.offset:
		pairs = pairs[args.offset:]
	if args.limit:
		pairs = pairs[:args.limit]
	print(f"Пар для обучения: {len(pairs)}")

	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	examples = build_examples(pairs, tokenizer, neo_id, args.lang)

	rms, median_norm = common.embedding_stats(model)
	if args.init_word.strip() == "random":
		# Старт без смысла: случайное направление с типичной длиной эмбеддинга (зависит от --seed).
		generator = torch.Generator().manual_seed(args.seed)
		dimension = model.get_input_embeddings().weight.shape[1]
		init_vector = torch.randn(dimension, generator=generator)
		init_vector = init_vector / init_vector.norm() * median_norm
	else:
		init_ids = common.encode(tokenizer, args.init_word)
		if init_ids.shape[0] != 1:
			raise SystemExit(f"Стартовое слово {args.init_word!r} режется на {init_ids.shape[0]} токена — возьмите другое слово.")
		init_vector = common.initial_vector(model, tokenizer, args.init_word)
	print(f"Типичный эмбеддинг: RMS координаты {rms:.4f}, медианная норма {median_norm:.2f}")
	print(f"Стартовое слово {args.init_word!r}: норма {init_vector.norm().item():.2f}")

	reference = compute_reference(model, examples, neo_id, init_vector, os.path.join(run_dir, "ref_logprobs.pt"))

	neologism = common.Neologism(init_vector.to(common.model_device(model)), rms)
	optimizer = torch.optim.AdamW([neologism.u], lr=args.lr, weight_decay=0.0)
	common.enable_training_mode(model)
	scale = common.loss_scale(model)
	if scale != 1.0:
		print(f"float16: потеря перед backward умножается на {scale:.0f}")

	# Проверка на старте: со стартовым вектором потеря обязана быть 2·ln 2.
	first = examples[0]
	with torch.no_grad():
		lc = common.response_logprob(model, first["prompt_ids"], first["chosen_ids"], neo_id, neologism.vector())
		lr = common.response_logprob(model, first["prompt_ids"], first["rejected_ids"], neo_id, neologism.vector())
		loss0 = common.apo_up_loss(lc, lr, reference[0][0], reference[0][1], args.beta).item()
	expected = 2 * math.log(2)
	print(f"\nПроверка на старте: потеря {loss0:.4f}, ожидается {expected:.4f}", end=" ")
	if abs(loss0 - expected) < 0.02:
		print("[OK]")
	else:
		print("[ОШИБКА: потеря на старте не равна 2·ln 2 — неверный подсчёт вероятностей]")

	peek_questions = []
	test_path = common.data_path(f"questions_test_{args.lang}.jsonl")
	if os.path.exists(test_path):
		for row in common.read_jsonl(test_path)[:config.PEEK_QUESTIONS]:
			peek_questions.append(row["question"])

	settings = {
		"model": config.MODEL_NAME,
		"neologism": config.NEOLOGISM,
		"concept": args.concept,
		"init_word": args.init_word,
		"train_lang": args.lang,
		"n_pairs": len(examples),
		"offset": args.offset,
		"seed": args.seed,
		"epochs": args.epochs,
		"lr": args.lr,
		"beta": args.beta,
		"grad_accum": args.grad_accum,
		"norm_penalty": config.NORM_PENALTY,
		"embedding_rms": rms,
		"embedding_median_norm": median_norm,
	}
	with open(os.path.join(run_dir, "settings.json"), "w", encoding="utf-8") as f:
		json.dump(settings, f, ensure_ascii=False, indent="\t")

	log_path = os.path.join(run_dir, "log.csv")
	log_file = open(log_path, "w", newline="", encoding="utf-8")
	log_writer = csv.writer(log_file)
	log_writer.writerow(["epoch", "pairs_done", "loss", "chosen_gain", "rejected_gain", "accuracy", "norm_ratio", "seconds"])

	print(f"\nОбучение: {args.epochs} эпохи по {len(examples)} пар, lr={args.lr}, beta={args.beta}, пар на шаг={args.grad_accum}")
	start = time.time()
	pairs_done = 0
	rng = random.Random(args.seed)
	window = {"loss": [], "chosen_gain": [], "rejected_gain": [], "correct": []}

	for epoch in range(1, args.epochs + 1):
		order = list(range(len(examples)))
		rng.shuffle(order)
		optimizer.zero_grad()
		for position, index in enumerate(order):
			example = examples[index]
			lc0, lr0 = reference[index]
			vector = neologism.vector()
			lc = common.response_logprob(model, example["prompt_ids"], example["chosen_ids"], neo_id, vector)
			lr = common.response_logprob(model, example["prompt_ids"], example["rejected_ids"], neo_id, vector)
			loss = common.apo_up_loss(lc, lr, lc0, lr0, args.beta)
			norm_ratio = vector.norm() / median_norm
			if config.NORM_PENALTY > 0:
				loss = loss + config.NORM_PENALTY * torch.relu(norm_ratio - config.NORM_MAX_RATIO)
			if not torch.isfinite(loss):
				print("  ВНИМАНИЕ: потеря не число, пропускаю пример", example["id"])
				continue
			(loss * scale / args.grad_accum).backward()

			window["loss"].append(loss.item())
			window["chosen_gain"].append(lc.item() - lc0)
			window["rejected_gain"].append(lr.item() - lr0)
			if (lc.item() - lr.item()) > (lc0 - lr0):
				window["correct"].append(1.0)
			else:
				window["correct"].append(0.0)
			pairs_done += 1

			is_last = position == len(order) - 1
			if pairs_done % args.grad_accum == 0 or is_last:
				grad = neologism.u.grad
				if grad is not None and bool(torch.isfinite(grad).all()):
					optimizer.step()
				else:
					print("  ВНИМАНИЕ: градиент не число (переполнение во float16?), пропускаю шаг")
				optimizer.zero_grad()

			if pairs_done % config.LOG_EVERY == 0 or is_last:
				elapsed = time.time() - start
				row = [
					epoch,
					pairs_done,
					metrics.mean(window["loss"]),
					metrics.mean(window["chosen_gain"]),
					metrics.mean(window["rejected_gain"]),
					metrics.mean(window["correct"]),
					norm_ratio.item(),
					round(elapsed),
				]
				log_writer.writerow(row)
				log_file.flush()
				print(
					f"  эпоха {epoch}, пар {pairs_done}: потеря {row[2]:.3f}, "
					f"хороший−старт {row[3]:+.1f}, плохой−старт {row[4]:+.1f}, "
					f"доля верных {row[5]:.2f}, норма {row[6]:.2f}, {elapsed / 60:.1f} мин"
				)
				for key in window:
					window[key] = []

		epoch_path = os.path.join(run_dir, f"neologism_epoch{epoch}.pt")
		common.save_neologism(epoch_path, neologism.vector(), init_vector, settings)
		common.save_neologism(os.path.join(run_dir, "neologism.pt"), neologism.vector(), init_vector, settings)
		print(f"Эпоха {epoch} готова, сохранено: {epoch_path}. {common.gpu_memory_report()}")
		if peek_questions:
			print("  Ответы со словом:")
			peek(model, tokenizer, neo_id, neologism.vector(), args.lang, peek_questions)

	log_file.close()
	print(f"\nГотово за {(time.time() - start) / 60:.1f} мин. Вектор: {os.path.join(run_dir, 'neologism.pt')}")


if __name__ == "__main__":
	main()
