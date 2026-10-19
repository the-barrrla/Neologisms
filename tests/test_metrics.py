"""Проверки подсчётов без видеокарты и без PyTorch.

Запуск из папки проекта: python tests/test_metrics.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neolib import metrics  # noqa: E402


def test_count_words():
	assert metrics.count_words("Hello, world!") == 2
	assert metrics.count_words("**Ответ:** два слова") == 3
	assert metrics.count_words("") == 0


def test_detect_language():
	assert metrics.detect_language("The sky is blue because of Rayleigh scattering.") == "en"
	assert metrics.detect_language("Небо голубое из-за рассеяния света.") == "ru"
	assert metrics.detect_language("Аспан көк, себебі күн сәулесі шашырайды.") == "kk"
	assert metrics.detect_language("12345 !!!") == "unknown"
	# Одна случайная «і» не делает русский текст казахским.
	assert metrics.detect_language("Это русский текст с одной буквой і внутри, довольно длинный.") == "ru"
	# Настоящие ответы Qwen2.5-1.5B из проверки в Colab:
	assert metrics.detect_language("Бул сөзүн неге аспан көк? Бирок, бул сөздөрдүн толук маалыматы менен жооп береймем.") == "ky"
	assert metrics.detect_language("Фотосинтез - это процесс, при котором растения и некоторые виды微生物合成有机物质，如葡萄糖和淀粉。这个过程主要在叶绿体中的叶绿素中进行。") == "zh"
	assert metrics.detect_language("Қазақстанның астанасы — Астана қаласы. Ол елдің солтүстігінде орналасқан.") == "kk"
	# Настоящий «казахский перевод» Gemma, который оказался украинским:
	assert metrics.detect_language("Ось деякі речі, які їм подобається: природа, китайська їжа, жива музика.") == "uk"


def test_translation_problems():
	# Хороший перевод — без замечаний.
	assert metrics.translation_problems("Why is the sky blue?", "Почему небо голубое?", "ru") == []
	assert metrics.translation_problems("Why is the sky blue?", "Неге аспан көк?", "kk") == []
	# Настоящие ошибки Gemma-3-4B при переводе тестовых вопросов:
	joke = metrics.translation_problems("Write a joke about fruit.", "Почему яблоки не ходят в кино? Потому что у них есть яблоки.", "ru")
	assert len(joke) == 1 and "длиннее" in joke[0]
	bash = metrics.translation_problems(
		"I have a Python script called preprocess.py. Write a bash command that runs it on all the files that end with "
		"\".checkpoint\" in all the subdirectories, subsubdirectories, etc. of the checkpoints/ directory.",
		"find checkpoints/ -name \"*.checkpoint\" -exec python preprocess.py {} \\;",
		"ru",
	)
	assert len(bash) == 2
	russian_instead = metrics.translation_problems(
		"What's a good itenary for a trip to Vegas? Planning a trip for 3.5 days, please exclude any museums",
		"Лайн-программа для поездки в Лас-Вегас? Планирую поездку на 3,5 дня, пожалуйста, исключите любые музеи.",
		"kk",
	)
	assert russian_instead == ["язык перевода похож на 'ru'"]
	chinese = metrics.translation_problems(
		"How much of sexual orientation is innate? Can being gay or straight be societally influenced?",
		"性取向дың қанша бөлігі тутық екен? ЛГБТ болу немесе гетеросексуал болу орталықтан ыдыраумен шектелмей ме?",
		"kk",
	)
	assert len(chinese) == 1 and chinese[0].startswith("чужие символы")
	# Много латиницы из-за команды LaTeX — это не ошибка.
	assert metrics.translation_problems("what does \\multirow{5}{*} do in latex", "Что делает \\multirow{5}{*} в LaTeX?", "ru") == []
	assert "обрезан по длине" in metrics.translation_problems("Hello there my friend", "Привет, мой друг", "ru", truncated=True)


def test_first_sentences():
	text = "First one. Second one! Third one? Fourth."
	assert metrics.first_sentences(text, 2) == "First one. Second one!"
	assert metrics.first_sentences("Only one", 2) == "Only one"


def test_mean_median_percentile():
	assert metrics.mean([1, 2, 3]) == 2
	assert math.isnan(metrics.mean([]))
	assert metrics.median([3, 1, 2]) == 2
	assert metrics.median([1, 2, 3, 4]) == 2.5
	assert metrics.percentile([0, 10], 50) == 5


def test_gap_closure():
	assert metrics.gap_closure(100, 50, 0) == 0.5
	assert metrics.gap_closure(100, 0, 0) == 1.0
	assert metrics.gap_closure(100, 100, 20) == 0.0
	assert math.isnan(metrics.gap_closure(50, 40, 60))
	triples = [(100, 50, 0), (200, 100, 0)]
	assert abs(metrics.closure_from_triples(triples) - 0.5) < 1e-9


def test_bootstrap():
	triples = []
	for i in range(50):
		triples.append((300 + i, 120 + i % 7, 40 + i % 5))
	point = metrics.closure_from_triples(triples)
	low, high = metrics.bootstrap_closure(triples, 500, seed=0)
	assert low <= point <= high
	again = metrics.bootstrap_closure(triples, 500, seed=0)
	assert again == (low, high)


def test_tsv_roundtrip():
	text = "строка 1\nстрока\t2 \\ обратная черта"
	escaped = metrics.escape_tsv(text)
	assert "\n" not in escaped and "\t" not in escaped
	assert metrics.unescape_tsv(escaped) == text


def main():
	tests = [
		test_count_words,
		test_detect_language,
		test_translation_problems,
		test_first_sentences,
		test_mean_median_percentile,
		test_gap_closure,
		test_bootstrap,
		test_tsv_roundtrip,
	]
	for test in tests:
		test()
		print(f"OK  {test.__name__}")
	print("Все проверки прошли.")


if __name__ == "__main__":
	main()
