"""Шаг 1. Вопросы: 700 для обучения и 100 для проверки.

Источники:
  lima  — датасет LIMA (как в статье). Он закрытый: сначала нажмите кнопку запроса доступа
          на https://huggingface.co/datasets/GAIR/lima и дождитесь одобрения.
  dolly — открытый датасет databricks-dolly-15k, доступ не нужен. Берутся только вопросы
          без прилагаемого текста (категории open_qa, general_qa, brainstorming, creative_writing).
  auto  — (по умолчанию) пробует LIMA, а если доступа нет, берёт Dolly.

Результат: data/questions_train_en.jsonl, data/questions_test_en.jsonl
Запуск: python 01_prepare_questions.py [--source auto|lima|dolly]
"""

import argparse
import random

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

import config
from neolib import files

DOLLY_REPO = "databricks/databricks-dolly-15k"
DOLLY_FILE = "databricks-dolly-15k.jsonl"
DOLLY_CATEGORIES = {"open_qa", "general_qa", "brainstorming", "creative_writing"}


def is_usable(question, seen):
	if not question or question in seen:
		return False
	if len(question) > config.MAX_QUESTION_CHARS:
		return False
	if config.NEOLOGISM in question:
		return False
	return True


def load_lima_questions(filename):
	path = hf_hub_download(repo_id="GAIR/lima", filename=filename, repo_type="dataset")
	questions = []
	seen = set()
	for row in files.read_jsonl(path):
		conversation = row.get("conversations") or []
		if not conversation:
			continue
		question = conversation[0].strip()
		if not is_usable(question, seen):
			continue
		seen.add(question)
		questions.append(question)
	return questions


def load_dolly_questions():
	path = hf_hub_download(repo_id=DOLLY_REPO, filename=DOLLY_FILE, repo_type="dataset")
	questions = []
	seen = set()
	for row in files.read_jsonl(path):
		if row.get("category") not in DOLLY_CATEGORIES:
			continue
		if (row.get("context") or "").strip():
			continue
		question = (row.get("instruction") or "").strip()
		if not is_usable(question, seen):
			continue
		seen.add(question)
		questions.append(question)
	return questions


def from_lima(rng):
	train = load_lima_questions("train.jsonl")
	test = load_lima_questions("test.jsonl")
	print(f"LIMA, подходящих вопросов: train {len(train)}, test {len(test)}")
	rng.shuffle(train)
	rng.shuffle(test)
	return train[:config.N_TRAIN_QUESTIONS], test[:config.N_TEST_QUESTIONS]


def from_dolly(rng):
	questions = load_dolly_questions()
	print(f"Dolly, подходящих вопросов: {len(questions)}")
	rng.shuffle(questions)
	# У Dolly один файл: первые вопросы идут в проверку, следующие — в обучение, без пересечений.
	test = questions[:config.N_TEST_QUESTIONS]
	end = config.N_TEST_QUESTIONS + config.N_TRAIN_QUESTIONS
	train = questions[config.N_TEST_QUESTIONS:end]
	return train, test


def to_rows(questions, prefix):
	rows = []
	for i, question in enumerate(questions):
		rows.append({"id": f"{prefix}-{i:04d}", "question": question})
	return rows


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", choices=["auto", "lima", "dolly"], default="auto")
	args = parser.parse_args()
	rng = random.Random(config.SEED)

	source = args.source
	if source in ("auto", "lima"):
		try:
			train, test = from_lima(rng)
			source = "lima"
		except (GatedRepoError, HfHubHTTPError) as error:
			if args.source == "lima":
				raise
			print("Нет доступа к LIMA:", str(error).splitlines()[-1])
			print("Беру открытый датасет Dolly. Запросить доступ к LIMA: https://huggingface.co/datasets/GAIR/lima")
			source = "dolly"
	if source == "dolly":
		train, test = from_dolly(rng)

	files.write_jsonl(files.data_path("questions_train_en.jsonl"), to_rows(train, "train"))
	files.write_jsonl(files.data_path("questions_test_en.jsonl"), to_rows(test, "test"))
	with open(files.data_path("questions_source.txt"), "w", encoding="utf-8") as f:
		f.write(source + "\n")
	print(f"Источник: {source}. Сохранено: {len(train)} обучающих и {len(test)} тестовых вопросов в папку {config.DATA_DIR}/")
	print("Пример тестового вопроса:", test[0][:200])


if __name__ == "__main__":
	main()
