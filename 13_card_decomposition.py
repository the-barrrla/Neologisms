"""Разложение эффекта ~card на вклад стартового слова и вклад обучения (видеокарта не нужна).

Для каждого пробега ~card, языка и меры формата сравниваются четыре условия на одних и тех же вопросах:
	base      — без приписки;
	control   — шаблон с необученным (стартовым) вектором;
	neologism — шаблон с обученным вектором;
	target    — явная инструкция (100%).
Разрыв «явная инструкция − без приписки» делится на две части:
	вклад стартового слова = (control − base)   / (target − base);
	вклад обучения         = (neologism − control) / (target − base);
их сумма — обычная доля закрытого разрыва. Плюс разности в процентных пунктах (neologism − control).
Интервалы — 95%, парный бутстрэп по вопросам (одни и те же вопросы во всех условиях).

Меры: count, first, rest, plain (доля ответов с выполненным условием), full (все 4), score (балл),
words (средняя длина; для неё «лучше» — меньше, знак разрыва учитывается автоматически).

Запуск: python 13_card_decomposition.py
Результат в analysis/: card_decomposition.csv, card_decomposition.png
"""

import csv
import math
import os
import random

import config
from neolib import card, files, metrics

# Пробеги, которых ещё нет на диске, пропускаются.
RUNS = [
	("card_en", "accurate, шаг 0,01"),
	("card_accurate_s1", "accurate, сид 1"),
	("card_accurate_s2", "accurate, сид 2"),
	("card_concise", "concise, шаг 0,01"),
	("card_concise_s1", "concise, сид 1"),
	("card_concise_s2", "concise, сид 2"),
	("card_concise_lr003", "concise, шаг 0,003"),
	("card_brief", "brief"),
	("card_detailed", "detailed"),
	("card_table", "table"),
	("card_the", "the"),
	("card_random", "случайный вектор"),
]
LANGS = ["en", "ru", "kk"]
MEASURES = ["count", "first", "rest", "plain", "full", "score", "words"]
MEASURE_NAMES = {
	"count": "ровно 4 предложения",
	"first": "1-е ≤ 20 слов",
	"rest": "остальные ≤ 15 слов",
	"plain": "без разметки",
	"full": "все 4 условия",
	"score": "балл",
	"words": "длина, слов",
}
CONDITIONS = ["base", "control", "neologism", "target"]
OUT_DIR = "analysis"
N_BOOT = 2000
# Если явная инструкция почти не меняет меру (разрыв меньше 0,02), доли не считаем: делить не на что.
MIN_GAP = 0.02


def measures_of(row):
	"""Число слов, балл формата и выполнение каждого условия ~card для одного ответа."""
	result = card.check(row["response"])
	values = {"words": float(row["words"]), "score": result["score"]}
	if result["full"]:
		values["full"] = 1.0
	else:
		values["full"] = 0.0
	for name in card.CRITERIA:
		if result[name]:
			values[name] = 1.0
		else:
			values[name] = 0.0
	return values


def load_run(run):
	"""{язык: {id вопроса: {условие: меры}}} только для вопросов, где есть все четыре условия."""
	by_lang = {}
	for row in files.read_jsonl(files.run_path(run, "generations.jsonl")):
		if row["condition"] not in CONDITIONS or row["lang"] not in LANGS:
			continue
		by_question = by_lang.setdefault(row["lang"], {})
		by_question.setdefault(row["id"], {})[row["condition"]] = measures_of(row)
	complete = {}
	for lang, by_question in by_lang.items():
		complete[lang] = {}
		for question_id, by_condition in by_question.items():
			has_all = True
			for condition in CONDITIONS:
				if condition not in by_condition:
					has_all = False
			if has_all:
				complete[lang][question_id] = by_condition
	return complete


def means(rows, measure):
	"""Среднее значение меры по каждому условию."""
	result = {}
	for condition in CONDITIONS:
		values = []
		for by_condition in rows:
			values.append(by_condition[condition][measure])
		result[condition] = metrics.mean(values)
	return result


def decompose(mean_values):
	"""Вклад старта, вклад обучения, их сумма (доли разрыва) и разность neologism − control."""
	gap = mean_values["target"] - mean_values["base"]
	difference = mean_values["neologism"] - mean_values["control"]
	if abs(gap) < MIN_GAP:
		return float("nan"), float("nan"), float("nan"), difference
	init_part = (mean_values["control"] - mean_values["base"]) / gap
	trained_part = difference / gap
	return init_part, trained_part, init_part + trained_part, difference


def bootstrap(rows, measure, seed):
	"""95% интервалы для четырёх величин из decompose: вопросы выбираются с возвращением."""
	rng = random.Random(seed)
	samples = [[], [], [], []]
	for _ in range(N_BOOT):
		resampled = []
		for _ in range(len(rows)):
			resampled.append(rows[rng.randrange(len(rows))])
		values = decompose(means(resampled, measure))
		for k in range(4):
			if not math.isnan(values[k]):
				samples[k].append(values[k])
	intervals = []
	for k in range(4):
		intervals.append((metrics.percentile(samples[k], 2.5), metrics.percentile(samples[k], 97.5)))
	return intervals


def analyse():
	"""Разложение доли разрыва на вклад слова и вклад обучения для всех пробегов, языков и мер."""
	results = []
	for run, label in RUNS:
		path = files.run_path(run, "generations.jsonl")
		if not os.path.exists(path):
			print(f"Нет {path} — пропускаю")
			continue
		data = load_run(run)
		for lang in LANGS:
			rows = list(data.get(lang, {}).values())
			if not rows:
				continue
			for measure in MEASURES:
				mean_values = means(rows, measure)
				point = decompose(mean_values)
				intervals = bootstrap(rows, measure, config.SEED)
				results.append({
					"run": run, "label": label, "lang": lang, "measure": measure, "n": len(rows),
					"base": mean_values["base"], "control": mean_values["control"],
					"neologism": mean_values["neologism"], "target": mean_values["target"],
					"init_part": point[0], "init_low": intervals[0][0], "init_high": intervals[0][1],
					"trained_part": point[1], "trained_low": intervals[1][0], "trained_high": intervals[1][1],
					"total": point[2], "total_low": intervals[2][0], "total_high": intervals[2][1],
					"diff": point[3], "diff_low": intervals[3][0], "diff_high": intervals[3][1],
				})
	return results


def pct(value):
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return "—"
	return f"{value * 100:.0f}".replace("-", "−")


def interval(low, high):
	if math.isnan(low) or math.isnan(high):
		return ""
	return f"[{pct(low)}…{pct(high)}]"


def present_runs(results):
	"""Пробеги из RUNS, для которых есть результаты."""
	runs = []
	for run, label in RUNS:
		for row in results:
			if row["run"] == run:
				runs.append((run, label))
				break
	return runs


def print_results(results):
	for run, label in present_runs(results):
		print(f"\n===== {run} ({label}) =====")
		print(f"  {'язык':5}{'мера':22}{'без':>6}{'контр.':>8}{'~card':>7}{'инстр.':>8}"
			f"   {'вклад старта':>20}   {'вклад обучения':>20}   {'~card − контроль, п.п.':>24}")
		for row in results:
			if row["run"] != run:
				continue
			if row["measure"] == "words":
				levels = f"{row['base']:6.0f}{row['control']:8.0f}{row['neologism']:7.0f}{row['target']:8.0f}"
				diff_text = f"{row['diff']:.0f} [{row['diff_low']:.0f}…{row['diff_high']:.0f}] слов"
			else:
				levels = f"{pct(row['base']):>6}{pct(row['control']):>8}{pct(row['neologism']):>7}{pct(row['target']):>8}"
				diff_text = f"{pct(row['diff'])} {interval(row['diff_low'], row['diff_high'])}"
			init_text = f"{pct(row['init_part'])}% {interval(row['init_low'], row['init_high'])}"
			trained_text = f"{pct(row['trained_part'])}% {interval(row['trained_low'], row['trained_high'])}"
			print(f"  {row['lang']:5}{MEASURE_NAMES[row['measure']]:22}{levels}   {init_text:>20}   {trained_text:>20}   {diff_text:>24}")


def write_csv(results, path):
	columns = ["run", "lang", "measure", "n", "base", "control", "neologism", "target",
		"init_part", "init_low", "init_high", "trained_part", "trained_low", "trained_high",
		"total", "total_low", "total_high", "diff", "diff_low", "diff_high"]
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(columns)
		for row in results:
			line = []
			for column in columns:
				value = row[column]
				if isinstance(value, float):
					value = round(value, 4)
				line.append(value)
			writer.writerow(line)


def plot(results, path):
	"""По панели на пробег: вклад старта и вклад обучения в долю разрыва по баллу, по языкам и условиям."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	shown = ["count", "rest", "plain", "score"]
	shown_names = ["ровно 4\nпредл.", "остальные\n≤ 15 слов", "без\nразметки", "балл"]
	colors = {"init": "#a3a29c", "trained": "#2a78d6"}
	runs = present_runs(results)
	fig, axes = plt.subplots(len(runs), len(LANGS), figsize=(12, 3.1 * len(runs)), facecolor="#fcfcfb", sharey=True, squeeze=False)
	for i, (run, label) in enumerate(runs):
		for j, lang in enumerate(LANGS):
			ax = axes[i][j]
			ax.set_facecolor("#fcfcfb")
			for side in ["top", "right"]:
				ax.spines[side].set_visible(False)
			ax.axhline(0, color="#52514e", linewidth=0.8)
			ax.axhline(100, color="#52514e", linewidth=0.8, linestyle="--")
			for g, measure in enumerate(shown):
				row = None
				for candidate in results:
					if candidate["run"] == run and candidate["lang"] == lang and candidate["measure"] == measure:
						row = candidate
				if row is None or math.isnan(row["init_part"]):
					continue
				for k, (key, low_key, high_key, name) in enumerate([
					("init_part", "init_low", "init_high", "init"),
					("trained_part", "trained_low", "trained_high", "trained"),
				]):
					x = g + (k - 0.5) * 0.38
					value = row[key] * 100
					low = row[low_key] * 100
					high = row[high_key] * 100
					ax.bar(x, value, 0.36, color=colors[name], zorder=2)
					ax.errorbar(x, value, yerr=[[max(0.0, value - low)], [max(0.0, high - value)]], fmt="none",
						ecolor="#52514e", elinewidth=0.9, capsize=2.5, zorder=3)
			ax.set_xticks(range(len(shown)))
			ax.set_xticklabels(shown_names, fontsize=8)
			if i == 0:
				ax.set_title(lang, fontsize=11, loc="left")
			if j == 0:
				ax.set_ylabel(f"{label}\nдоля разрыва, %", fontsize=9)
	handles = [
		plt.Rectangle((0, 0), 1, 1, color=colors["init"]),
		plt.Rectangle((0, 0), 1, 1, color=colors["trained"]),
	]
	fig.legend(handles, ["вклад стартового слова (контроль − без приписки)", "вклад обучения (~card − контроль)"],
		loc="upper right", frameon=False, ncol=2, fontsize=9)
	fig.suptitle("~card: что даёт стартовое слово, а что обучение", x=0.01, ha="left", fontsize=12)
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	fig.savefig(path, dpi=150, facecolor="#fcfcfb")
	plt.close(fig)


def main():
	files.setup_console()
	os.makedirs(OUT_DIR, exist_ok=True)
	results = analyse()
	print_results(results)
	write_csv(results, os.path.join(OUT_DIR, "card_decomposition.csv"))
	try:
		plot(results, os.path.join(OUT_DIR, "card_decomposition.png"))
	except ImportError:
		print("matplotlib не установлен — график пропущен")
	print(f"\nСохранено в {OUT_DIR}/: card_decomposition.csv, card_decomposition.png")


if __name__ == "__main__":
	main()
