"""Подсчёты, которым не нужен PyTorch: длина ответа, язык ответа, доля
закрытого разрыва, бутстрэп. Их можно проверить без видеокарты:
python tests/test_metrics.py
"""

import math
import random
import re

WORD_RE = re.compile(r"\w+", re.UNICODE)
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")

# Буквы, которые есть в казахской кириллице, но не в русской.
KAZAKH_LETTERS = set("әғқңөұүһі")
# Из них только казахские: ө, ү, ң есть и в киргизском, а эти — нет.
KAZAKH_ONLY_LETTERS = set("әғқұһі")
# Типичные киргизские слова, которых нет в казахском (модели путают эти языки).
KYRGYZ_MARKERS = {"бул", "жооп", "маалымат", "маалыматы", "болуп", "бирок", "үчүн", "абдан"}
# Буквы, которые есть только в украинском (модели иногда сбиваются на него).
UKRAINIAN_LETTERS = set("їєґ")


def count_words(text):
	"""Число слов: последовательности букв и цифр. Разметка Markdown не считается."""
	return len(WORD_RE.findall(text))


def is_cjk(ch):
	"""Китайские иероглифы (Qwen иногда переходит на китайский посреди ответа)."""
	code = ord(ch)
	return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def detect_language(text):
	"""Грубо определяет язык ответа по буквам: 'en', 'ru', 'kk', 'ky', 'uk', 'zh' или 'unknown'.

	Иероглифов больше 10% знаков — 'zh' (смесь с китайским). Латиница больше половины — английский.
	Иначе кириллица: есть хотя бы 2 чисто казахские буквы (ә ғ қ ұ һ і) и их не меньше 1% —
	казахский, но если киргизских слов больше 2% — 'ky'. Есть ї, є или ґ — украинский. Остальное — русский.
	"""
	latin = 0
	cyrillic = 0
	kazakh_only = 0
	ukrainian = 0
	cjk = 0
	for ch in text.lower():
		if is_cjk(ch):
			cjk += 1
		elif ch in UKRAINIAN_LETTERS:
			cyrillic += 1
			ukrainian += 1
		elif ch in KAZAKH_LETTERS:
			cyrillic += 1
			if ch in KAZAKH_ONLY_LETTERS:
				kazakh_only += 1
		elif "а" <= ch <= "я" or ch == "ё":
			cyrillic += 1
		elif "a" <= ch <= "z":
			latin += 1
	letters = latin + cyrillic + cjk
	if letters == 0:
		return "unknown"
	if cjk / letters > 0.1:
		return "zh"
	if latin / letters > 0.5:
		return "en"
	words = WORD_RE.findall(text.lower())
	kyrgyz = 0
	for word in words:
		if word in KYRGYZ_MARKERS:
			kyrgyz += 1
	if words and kyrgyz / len(words) > 0.02:
		return "ky"
	if ukrainian >= 2:
		return "uk"
	if kazakh_only >= 2 and kazakh_only / cyrillic >= 0.01:
		return "kk"
	return "ru"


def foreign_letters(text):
	"""Буквы не из латиницы и не из кириллицы (иероглифы, арабские, индийские и т. п.)."""
	found = []
	for ch in text:
		if not ch.isalpha():
			continue
		code = ord(ch)
		# 0x250–0x2AF — МФА; оттуда в казахских текстах бывает «ə» вместо «ә», это мелочь.
		if code < 0x2B0 or 0x400 <= code <= 0x52F:
			continue
		if ch not in found:
			found.append(ch)
	return found


def cyrillic_share(text):
	"""Доля кириллических букв среди всех букв."""
	letters = 0
	cyrillic = 0
	for ch in text:
		if not ch.isalpha():
			continue
		letters += 1
		if 0x400 <= ord(ch) <= 0x52F:
			cyrillic += 1
	if letters == 0:
		return 0.0
	return cyrillic / letters


def translation_problems(original, translation, lang, truncated=False):
	"""Признаки плохого перевода. Возвращает список коротких описаний; пустой список — всё в порядке.

	Главная беда маленьких моделей: вместо перевода они выполняют просьбу (пишут шутку, эссе,
	код) или отвечают на вопрос. Это видно по длине: перевод обычно не короче 0.4 и не длиннее
	2 оригиналов (в словах).
	"""
	problems = []
	if truncated:
		problems.append("обрезан по длине")
	if not translation.strip():
		problems.append("пустой перевод")
		return problems
	original_words = count_words(original)
	translated_words = count_words(translation)
	if original_words >= 4:
		ratio = translated_words / original_words
		if ratio > 2.0:
			problems.append(f"в {ratio:.1f} раза длиннее оригинала — похоже на ответ, а не перевод")
		elif ratio < 0.4:
			problems.append(f"в {1 / max(ratio, 0.01):.1f} раза короче оригинала — похоже на ответ или потерю текста")
	detected = detect_language(translation)
	if detected == "en" and lang != "en" and cyrillic_share(translation) >= 0.3:
		# Много латиницы из-за кода или названий (например, «Что делает \multirow в LaTeX?»).
		detected = lang
	if detected != lang:
		short_russian_like = False
		if lang == "kk" and detected == "ru":
			# Короткая казахская фраза может обойтись без особых букв, поэтому
			# ругаемся, только если особых букв нет совсем, а слов уже немало.
			has_kazakh = False
			for ch in translation.lower():
				if ch in KAZAKH_LETTERS:
					has_kazakh = True
					break
			short_russian_like = has_kazakh or translated_words < 5
		if not short_russian_like:
			problems.append(f"язык перевода похож на '{detected}'")
	foreign = foreign_letters(translation)
	if foreign:
		problems.append("чужие символы: " + "".join(foreign[:10]))
	return problems


def first_sentences(text, n):
	"""Первые n предложений текста (для запасного способа сделать короткий ответ)."""
	parts = SENTENCE_END_RE.split(text.strip())
	kept = []
	for part in parts:
		part = part.strip()
		if part:
			kept.append(part)
		if len(kept) >= n:
			break
	return " ".join(kept)


def mean(values):
	if not values:
		return float("nan")
	total = 0.0
	for v in values:
		total += v
	return total / len(values)


def median(values):
	if not values:
		return float("nan")
	ordered = sorted(values)
	middle = len(ordered) // 2
	if len(ordered) % 2 == 1:
		return float(ordered[middle])
	return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile(values, q):
	"""Перцентиль q (от 0 до 100) с линейной интерполяцией."""
	if not values:
		return float("nan")
	ordered = sorted(values)
	position = (len(ordered) - 1) * q / 100.0
	low = math.floor(position)
	high = math.ceil(position)
	if low == high:
		return float(ordered[low])
	weight = position - low
	return ordered[low] * (1.0 - weight) + ordered[high] * weight


def gap_closure(base, neo, target):
	"""Доля закрытого разрыва: (база − неологизм) / (база − цель).

	1.0 — слово сокращает ответы так же, как явная просьба; 0 — не сокращает;
	больше 1 — сокращает сильнее. Если явная просьба не укоротила ответы, разрыва нет: nan.
	"""
	denominator = base - target
	if denominator <= 0:
		return float("nan")
	return (base - neo) / denominator


def closure_from_triples(triples):
	"""triples — список (длина_база, длина_неологизм, длина_цель) по вопросам."""
	base_values = []
	neo_values = []
	target_values = []
	for base, neo, target in triples:
		base_values.append(base)
		neo_values.append(neo)
		target_values.append(target)
	return gap_closure(mean(base_values), mean(neo_values), mean(target_values))


def bootstrap_closure(triples, n_boot, seed):
	"""95% доверительный интервал доли закрытого разрыва: вопросы выбираются с возвращением."""
	if not triples:
		return float("nan"), float("nan")
	rng = random.Random(seed)
	estimates = []
	for _ in range(n_boot):
		sample = []
		for _ in range(len(triples)):
			sample.append(triples[rng.randrange(len(triples))])
		value = closure_from_triples(sample)
		if not math.isnan(value):
			estimates.append(value)
	return percentile(estimates, 2.5), percentile(estimates, 97.5)


def escape_tsv(text):
	"""Переносы строк и табуляции внутри ячейки TSV заменяются на \\n и \\t."""
	return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def unescape_tsv(text):
	"""Обратное к escape_tsv."""
	result = []
	i = 0
	while i < len(text):
		ch = text[i]
		if ch == "\\" and i + 1 < len(text):
			nxt = text[i + 1]
			if nxt == "n":
				result.append("\n")
				i += 2
				continue
			if nxt == "t":
				result.append("\t")
				i += 2
				continue
			if nxt == "\\":
				result.append("\\")
				i += 2
				continue
		result.append(ch)
		i += 1
	return "".join(result)
