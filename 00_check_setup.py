"""Шаг 0. Проверка перед экспериментом (10–15 минут).

1. Видеокарта видна, модель загружается.
2. Неологизм в каждом шаблоне — ровно один токен.
3. Без неологизма генерация через векторы совпадает с обычной генерацией.
4. Знает ли модель нужные языки: по 5 простых вопросов на каждом языке.

Запуск: python 00_check_setup.py
"""

import argparse

import torch

import config
from neolib import common, metrics

# Казахские вопросы стоит перепроверить переводчиком.
CHECK_QUESTIONS = {
	"en": [
		"Why is the sky blue?",
		"How do I brew good black tea?",
		"What is photosynthesis?",
		"What should I do to sleep better?",
		"What is the capital of Kazakhstan?",
	],
	"ru": [
		"Почему небо голубое?",
		"Как правильно заварить чёрный чай?",
		"Что такое фотосинтез?",
		"Что делать, чтобы лучше спать?",
		"Какой город — столица Казахстана?",
	],
	"kk": [
		"Неге аспан көк?",
		"Қара шайды қалай дұрыс демдеу керек?",
		"Фотосинтез дегеніміз не?",
		"Жақсы ұйықтау үшін не істеу керек?",
		"Қазақстанның астанасы қай қала?",
	],
}


def check_templates(tokenizer, neo_id):
	print("\n[1] Неологизм в шаблонах")
	all_ok = True
	for lang in config.LANGS:
		suffix = common.neologism_suffix(lang)
		prompt = common.chat_prompt(tokenizer, common.build_user_text("Test question?", suffix))
		count = common.count_token(common.encode(tokenizer, prompt), neo_id)
		pieces = tokenizer.convert_ids_to_tokens(common.encode(tokenizer, suffix).tolist())
		if count == 1:
			status = "OK"
		else:
			status = "ОШИБКА"
			all_ok = False
		print(f"  {lang}: неологизм встречается {count} раз [{status}]")
		print(f"      токены приписки: {pieces}")
	return all_ok


def check_identity(model, tokenizer, neo_id):
	print("\n[2] Без неологизма путь через векторы не меняет ответы")
	all_same = True
	for question in CHECK_QUESTIONS["en"][:2]:
		plain = common.generate_plain(model, tokenizer, question, 48, repetition_penalty=1.0)
		via_embeds = common.generate_batch(model, tokenizer, [question], neo_id, None, 48, repetition_penalty=1.0)[0]["text"]
		same = plain == via_embeds
		if not same:
			all_same = False
		print(f"  {question!r}: {'совпадает' if same else 'НЕ совпадает'}")
		if not same:
			print(f"      обычная генерация: {plain[:200]!r}")
			print(f"      через векторы:     {via_embeds[:200]!r}")
	if not all_same:
		print("  Если расходится только самый конец ответа, это мелкие различия округления, не страшно.")
		print("  Если ответы совсем разные, путь через векторы (inputs_embeds) работает неверно.")
	return all_same


def check_languages(model, tokenizer, neo_id):
	print("\n[3] Знает ли модель языки (смотрите на смысл ответов глазами!)")
	for lang in config.LANGS:
		questions = CHECK_QUESTIONS.get(lang)
		if not questions:
			print(f"  {lang}: нет проверочных вопросов, добавьте их в CHECK_QUESTIONS")
			continue
		results = common.generate_many(
			model, tokenizer, questions, neo_id, None,
			config.CHECK_MAX_NEW_TOKENS, batch_size=len(questions),
		)
		right_language = 0
		print(f"\n  ===== {lang} =====")
		for question, result in zip(questions, results):
			detected = metrics.detect_language(result["text"])
			if detected == lang:
				right_language += 1
			print(f"  В: {question}")
			print(f"  О [{detected}]: {result['text'][:300]}")
			print()
		print(f"  {lang}: ответов на нужном языке {right_language} из {len(questions)}")


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--skip-languages", action="store_true", help="не проверять языки")
	args = parser.parse_args()
	common.set_seed(config.SEED)

	if torch.cuda.is_available():
		props = torch.cuda.get_device_properties(0)
		print(f"GPU: {props.name}, {props.total_memory / 1024 ** 3:.1f} ГБ")
	print(f"Загружаю {config.MODEL_NAME} ...")
	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	print(f"Модель загружена: {type(model).__name__}, {common.gpu_memory_report()}")

	templates_ok = check_templates(tokenizer, neo_id)
	identity_ok = check_identity(model, tokenizer, neo_id)
	if not args.skip_languages:
		check_languages(model, tokenizer, neo_id)

	print("\nИтог:")
	print(f"  шаблоны: {'OK' if templates_ok else 'ОШИБКА'}")
	print(f"  без неологизма модель не меняется: {'OK' if identity_ok else 'РАСХОЖДЕНИЕ, см. выше'}")
	print("  языки: ответы выше нужно оценить вручную; если ответы на каком-то языке бессмысленные, этот язык лучше исключить из config.LANGS.")


if __name__ == "__main__":
	main()
