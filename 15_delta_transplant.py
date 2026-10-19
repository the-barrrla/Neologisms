"""Проверка: пересадка выученного сдвига на другое стартовое слово.

У двух пробегов ~card с разными стартами: v_A = e_A + Δ_A (старт «accurate»), v_B = e_B + Δ_B (старт «concise»).
Собираем новые векторы и смотрим, что они делают:
	e_A + Δ_B      — старт «accurate» со сдвигом, выученным от «concise»;
	e_B + Δ_A      — старт «concise» со сдвигом, выученным от «accurate»;
	e_N            — нейтральное слово само по себе (контроль для двух следующих);
	e_N + Δ_B      — нейтральное слово со сдвигом от «concise»;
	e_N + Δ_A      — нейтральное слово со сдвигом от «accurate»;
	Δ_B, Δ_A       — только сдвиг, растянутый до длины обычного эмбеддинга (без стартового слова).
Если поведение «переезжает» вместе со сдвигом, понятие сидит в сдвиге. Если e_A + Δ_B не похоже на v_B,
сдвиг работает только в паре со своим стартовым словом.
Для сравнения в таблицу добавляются e_A, v_A, e_B, v_B из generations.jsonl пробегов (их не генерируем).

Запуск: python 15_delta_transplant.py
	--a card_en --b card_concise   (по умолчанию)
	--neutral the                  (нейтральное слово, один токен)
	--limit 50                     (вопросов на язык)
Около 30–40 минут. Прерванный запуск продолжится с места остановки.
Результат: analysis/delta_transplant_{a}_vs_{b}.jsonl (ответы) и .csv
"""

import argparse
import os

import config
from neolib import common, files, probe


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--a", default="card_en", help="пробег ~card с первым стартовым словом")
	parser.add_argument("--b", default="card_concise", help="пробег ~card со вторым стартовым словом")
	parser.add_argument("--neutral", default="the", help="нейтральное слово для старта")
	parser.add_argument("--limit", type=int, default=50)
	parser.add_argument("--max-new-tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
	args = parser.parse_args()
	common.set_seed(config.SEED)
	probe.use_card_token()

	v_a, e_a, meta_a = probe.load_vectors(args.a)
	v_b, e_b, meta_b = probe.load_vectors(args.b)
	delta_a = v_a - e_a
	delta_b = v_b - e_b
	name_a = meta_a.get("init_word", "A").strip()
	name_b = meta_b.get("init_word", "B").strip()
	print(f"A = {args.a} (старт {name_a!r}), B = {args.b} (старт {name_b!r})")
	print(f"cos(Δ_A, Δ_B) = {float((delta_a @ delta_b) / (delta_a.norm() * delta_b.norm())):.3f}")

	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	e_n = probe.word_vector(model, tokenizer, args.neutral)
	typical = float(meta_b.get("embedding_median_norm", e_b.norm()))
	name_n = args.neutral.strip()

	generated = [
		(f"{name_a} + Δ от {name_b}", e_a + delta_b),
		(f"{name_b} + Δ от {name_a}", e_b + delta_a),
		(f"{name_n} (без сдвига)", e_n),
		(f"{name_n} + Δ от {name_b}", e_n + delta_b),
		(f"{name_n} + Δ от {name_a}", e_n + delta_a),
		(f"только Δ от {name_b}", delta_b / delta_b.norm() * typical),
		(f"только Δ от {name_a}", delta_a / delta_a.norm() * typical),
	]
	reused = [
		(f"{name_a} (без сдвига)", args.a, "control"),
		(f"{name_a} + свой Δ (обученный)", args.a, "neologism"),
		(f"{name_b} (без сдвига)", args.b, "control"),
		(f"{name_b} + свой Δ (обученный)", args.b, "neologism"),
	]

	os.makedirs("analysis", exist_ok=True)
	# Имя файла содержит оба пробега: у разных пробегов с одним стартовым словом подписи вариантов совпадают.
	name = f"delta_transplant_{args.a}_vs_{args.b}"
	cache_path = os.path.join("analysis", name + ".jsonl")
	cache = probe.load_cache(cache_path)
	rows = []
	for lang in probe.LANGS:
		questions = probe.test_questions(lang, args.limit)
		question_ids = []
		for row in questions:
			question_ids.append(row["id"])
		levels = probe.reference_levels(args.b, lang, question_ids)
		suffix = common.neologism_suffix(lang)
		for variant, run, condition in reused:
			answers_by_id = probe.saved_answers(run, condition)
			answers = []
			for question_id in question_ids:
				if (lang, question_id) in answers_by_id:
					answers.append(answers_by_id[(lang, question_id)])
			if len(answers) == len(question_ids):
				rows.append(probe.summarize(variant, lang, answers, levels))
		for variant, vector in generated:
			answers = probe.generate_variant(model, tokenizer, neo_id, vector, variant, lang,
				questions, suffix, cache, cache_path, args.max_new_tokens)
			rows.append(probe.summarize(variant, lang, answers, levels))

	probe.print_table(rows, "Пересадка сдвига (доля разрыва — относительно инструкции и ответов без приписки)")
	csv_path = os.path.join("analysis", name + ".csv")
	probe.write_rows(rows, csv_path)
	print(f"\nСохранено: {cache_path}, {csv_path}")


if __name__ == "__main__":
	main()
