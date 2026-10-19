"""Проверка формата «ответ-карточка» для неологизма ~card. Без PyTorch.

Ответ-карточка — это ответ, в котором выполнены все четыре условия:
	count — ровно 4 предложения;
	first — первое предложение не длиннее 20 слов;
	rest  — каждое из остальных предложений не длиннее 15 слов (нужно хотя бы 2 предложения);
	plain — нет разметки: заголовков, списков, **жирного**, `кода`, таблиц, цитат, эмодзи.
Балл ответа — доля выполненных условий (0, 0.25, 0.5, 0.75 или 1).
"""

import math
import random
import re

from neolib import metrics

CRITERIA = ["count", "first", "rest", "plain"]
CRITERION_NAMES = {
	"count": "ровно 4 предложения",
	"first": "1-е ≤ 20 слов",
	"rest": "остальные ≤ 15 слов",
	"plain": "без разметки",
}
N_SENTENCES = 4
FIRST_MAX_WORDS = 20
REST_MAX_WORDS = 15

# Конец предложения: знаки .!?… (возможно, несколько), за ними закрывающие кавычки или скобки и пробел.
END_RE = re.compile(r"[.!?…]+[\"'»”’)\]]*\s+")
# С чего может начинаться следующее предложение: заглавная буква, цифра, открывающая кавычка или тире.
NEXT_START_RE = re.compile(r"[«\"“'(\[—–-]?[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ0-9]")
# Сокращения, после которых идёт заглавная буква, но предложение не кончается.
# Инициалы: «R», «R.R», «Ш» — одиночные буквы через точку.
INITIALS_RE = re.compile(r"^(\w\.)*\w$")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "no", "jr", "sr", "им", "ул", "проф", "г", "гг"}

MARKDOWN_LINE_RES = [
	re.compile(r"^\s*#{1,6}\s"),             # заголовок
	re.compile(r"^\s*[-*•+▪◦]\s"),          # пункт списка
	re.compile(r"^\s*\d{1,2}[.)]\s"),        # нумерованный список
	re.compile(r"^\s*>"),                    # цитата
	re.compile(r"^\s*\|.*\|"),               # таблица
	re.compile(r"^\s*([-*_]\s*){3,}$"),      # горизонтальная линия
]
MARKDOWN_ANYWHERE_RES = [
	re.compile(r"\*\*[^*\n]+\*\*"),          # **жирный**
	re.compile(r"__[^_\n]+__"),              # __жирный__
	re.compile(r"`"),                        # код
]


def is_emoji(ch):
	"""Эмодзи и пиктограммы по диапазонам Юникода."""
	code = ord(ch)
	if 0x1F300 <= code <= 0x1FAFF:
		return True
	if 0x2600 <= code <= 0x27BF:
		return True
	if 0x1F000 <= code <= 0x1F2FF:
		return True
	return False


def has_markdown(text):
	"""Есть ли заголовки, списки, цитаты, таблицы, жирный, код или эмодзи."""
	for line in text.splitlines():
		for pattern in MARKDOWN_LINE_RES:
			if pattern.search(line):
				return True
	for pattern in MARKDOWN_ANYWHERE_RES:
		if pattern.search(text):
			return True
	for ch in text:
		if is_emoji(ch):
			return True
	return False


def last_word_before(text, position):
	"""Последнее «слово» перед позицией (без точки), в нижнем регистре."""
	start = position
	while start > 0 and not text[start - 1].isspace():
		start -= 1
	word = text[start:position]
	return word.strip("«\"“'([").lower()


def is_boundary(line, match):
	"""Настоящий ли это конец предложения, а не сокращение или инициал."""
	rest = line[match.end():]
	if not NEXT_START_RE.match(rest):
		return False
	if line[match.start()] != ".":
		return True
	word = last_word_before(line, match.start())
	if word in ABBREVIATIONS:
		return False
	# Инициалы перед точкой: «А. С. Пушкин», «Ш. Й. Агнон», «George R.R. Martin».
	if INITIALS_RE.match(word) and not word[-1].isdigit():
		return False
	return True


def split_sentences(text):
	"""Делит ответ на предложения. Перенос строки тоже считается границей.
	Куски без единого слова (например, «---») отбрасываются."""
	sentences = []
	for line in text.strip().splitlines():
		line = line.strip()
		if not line:
			continue
		start = 0
		for match in END_RE.finditer(line):
			if is_boundary(line, match):
				piece = line[start:match.end()].strip()
				if metrics.count_words(piece) > 0:
					sentences.append(piece)
				start = match.end()
		piece = line[start:].strip()
		if metrics.count_words(piece) > 0:
			sentences.append(piece)
	return sentences


def check(text):
	"""Словарь {условие: True/False} плюс число предложений, балл и «всё выполнено»."""
	sentences = split_sentences(text)
	lengths = []
	for sentence in sentences:
		lengths.append(metrics.count_words(sentence))
	result = {}
	result["count"] = len(sentences) == N_SENTENCES
	result["first"] = len(lengths) >= 1 and lengths[0] <= FIRST_MAX_WORDS
	rest_ok = len(lengths) >= 2
	for length in lengths[1:]:
		if length > REST_MAX_WORDS:
			rest_ok = False
	result["rest"] = rest_ok
	result["plain"] = not has_markdown(text)
	satisfied = 0
	for name in CRITERIA:
		if result[name]:
			satisfied += 1
	result["n_sentences"] = len(sentences)
	result["score"] = satisfied / len(CRITERIA)
	result["full"] = satisfied == len(CRITERIA)
	return result


def failed_criteria(result):
	failed = []
	for name in CRITERIA:
		if not result[name]:
			failed.append(name)
	return failed


# ---------- доля закрытого разрыва для «чем больше, тем лучше» ----------

def rise_closure(base, value, target):
	"""(условие − база) / (цель − база). Если явная просьба почти не подняла показатель
	(меньше чем на 0.02), разрыва нет и доля не определена: nan."""
	denominator = target - base
	if denominator < 0.02:
		return float("nan")
	return (value - base) / denominator


def closure_from_triples(triples):
	"""triples — список (база, условие, цель) по вопросам; значения — балл или 0/1."""
	base_values = []
	values = []
	target_values = []
	for base, value, target in triples:
		base_values.append(base)
		values.append(value)
		target_values.append(target)
	return rise_closure(metrics.mean(base_values), metrics.mean(values), metrics.mean(target_values))


def bootstrap_closure(triples, n_boot, seed):
	"""95% интервал: вопросы выбираются с возвращением."""
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
	return metrics.percentile(estimates, 2.5), metrics.percentile(estimates, 97.5)
