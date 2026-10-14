"""Проверки формата «ответ-карточка» (neolib/card.py) без видеокарты и без PyTorch.

Запуск из папки проекта: python tests/test_card.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neolib import card  # noqa: E402

GOOD_EN = "Paris is the capital of France and its largest city. It sits on the Seine river. Millions of tourists visit every year. The city is famous for its museums."
GOOD_RU = "Париж — столица Франции и её крупнейший город. Он стоит на Сене. Каждый год туда приезжают миллионы туристов. Город славится своими музеями."
GOOD_KK = "Париж — Францияның астанасы және ең үлкен қаласы. Ол Сена өзенінің бойында орналасқан. Жыл сайын оған миллиондаған турист келеді. Қала мұражайларымен танымал."


def test_split_sentences():
	assert len(card.split_sentences(GOOD_EN)) == 4
	assert len(card.split_sentences(GOOD_RU)) == 4
	assert len(card.split_sentences(GOOD_KK)) == 4
	# Инициалы и сокращения не делят предложение.
	assert len(card.split_sentences("Martin R.R. George writes books. He lives in Westeros.")) == 2
	assert len(card.split_sentences("Это написал А. С. Пушкин. Он жил давно.")) == 2
	assert len(card.split_sentences("Dr. Smith agrees. So do I.")) == 2
	# Точка внутри числа и строчная буква после точки — не граница.
	assert len(card.split_sentences("It costs 3.5 dollars, e.g. at the shop. Fine.")) == 2
	# Перенос строки — граница; строка без слов не считается.
	assert len(card.split_sentences("Title\n\n---\n\nText here.")) == 2


def test_markdown():
	assert not card.has_markdown(GOOD_EN)
	assert card.has_markdown("## Heading\nText.")
	assert card.has_markdown("Intro:\n- one\n- two")
	assert card.has_markdown("Intro:\n1. one\n2. two")
	assert card.has_markdown("This is **bold** text.")
	assert card.has_markdown("Use `ls` here.")
	assert card.has_markdown("Great idea! 😂")
	# Тире в середине предложения — не список.
	assert not card.has_markdown("Париж — столица Франции.")


def test_check():
	for text in [GOOD_EN, GOOD_RU, GOOD_KK]:
		result = card.check(text)
		assert result["full"], card.failed_criteria(result)
		assert result["score"] == 1.0
	three = card.check("One. Two. Three.")
	assert not three["count"] and three["first"] and three["rest"] and three["plain"]
	assert three["score"] == 0.75
	long_first = card.check("This first sentence is deliberately made far too long, with many more words than twenty, to break the rule on purpose today. Two. Three. Four.")
	assert long_first["count"] and not long_first["first"]
	# Пустой ответ выполняет только «без разметки».
	assert card.check("")["score"] == 0.25


def test_closure():
	assert abs(card.rise_closure(0.2, 0.6, 1.0) - 0.5) < 1e-9
	assert math.isnan(card.rise_closure(0.5, 0.6, 0.51))
	triples = [(0.0, 1.0, 1.0), (0.25, 0.25, 1.0)]
	value = card.closure_from_triples(triples)
	assert abs(value - (0.625 - 0.125) / (1.0 - 0.125)) < 1e-9
	low, high = card.bootstrap_closure(triples, 200, 0)
	assert low <= value <= high


def main():
	tests = [test_split_sentences, test_markdown, test_check, test_closure]
	for test in tests:
		test()
		print(f"OK  {test.__name__}")
	print("Все проверки прошли.")


if __name__ == "__main__":
	main()
