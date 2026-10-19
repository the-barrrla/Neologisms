"""Шаг 6. Итоги эксперимента: таблица, доверительные интервалы и два графика.

Для каждого языка считаем:
	* среднюю и медианную длину ответа в словах по условиям;
	* долю закрытого разрыва = (база − неологизм) / (база − явная просьба), с 95% интервалом;
	то же для контроля (необученный вектор): там должно быть около нуля;
	* долю ответов на нужном языке и долю ответов, обрезанных по длине.

Запуск: python 06_analyze.py --run short_en
Результат в runs/{run}/: summary.csv, gap_closure.png, words.png
"""

import argparse
import csv
import math
import os

import config
from neolib import files, metrics

CONDITIONS = ["base", "neologism", "target", "target_en", "short_en", "short_native", "control"]
CONDITION_NAMES = {
	"base": "без приписки",
	"neologism": "с ~short",
	"target": "явная просьба",
	"target_en": "англ. просьба",
	"short_en": "англ. «short»",
	"short_native": "«короткий» на языке вопроса",
	"control": "необученный вектор",
}
# Условия, для которых считается доля закрытого разрыва (относительно явной просьбы на том же языке).
CLOSURE_CONDITIONS = ["neologism", "target_en", "short_en", "short_native", "control"]

# Цвета (проверенная палитра: синий, оранжевый, бирюзовый, жёлтый) и цвета текста.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CONTROL_COLOR = "#a3a29c"
WORD_COLORS = {
	"base": SERIES_COLORS[0],
	"neologism": SERIES_COLORS[1],
	"target": SERIES_COLORS[2],
	"target_en": "#7a6fd0",
	"short_en": "#c2569b",
	"short_native": "#8a6d3b",
	"control": SERIES_COLORS[3],
}
CLOSURE_COLORS = {
	"neologism": SERIES_COLORS[0],
	"target_en": "#7a6fd0",
	"short_en": "#c2569b",
	"short_native": "#8a6d3b",
	"control": CONTROL_COLOR,
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def group_rows(rows):
	"""{язык: {условие: {id вопроса: строка}}}"""
	grouped = {}
	for row in rows:
		by_lang = grouped.setdefault(row["lang"], {})
		by_condition = by_lang.setdefault(row["condition"], {})
		by_condition[row["id"]] = row
	return grouped


def closure_triples(by_condition, condition):
	"""Тройки длин (база, условие, цель) по вопросам, у которых есть все три ответа."""
	triples = []
	needed = ["base", condition, "target"]
	for question_id in by_condition.get("base", {}):
		present = True
		for name in needed:
			if question_id not in by_condition.get(name, {}):
				present = False
		if present:
			triples.append((
				by_condition["base"][question_id]["words"],
				by_condition[condition][question_id]["words"],
				by_condition["target"][question_id]["words"],
			))
	return triples


def fmt_percent(value):
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return "—"
	return f"{value * 100:.0f}%"


def summarize(grouped):
	"""По языкам и условиям: длина ответов, доля обрезанных и ответов на нужном языке, доля разрыва."""
	summary = []
	langs = []
	for lang in config.LANGS:
		if lang in grouped:
			langs.append(lang)
	for lang in grouped:
		if lang not in langs:
			langs.append(lang)
	for lang in langs:
		by_condition = grouped[lang]
		entry = {"lang": lang, "conditions": {}}
		for condition in CONDITIONS:
			rows = list(by_condition.get(condition, {}).values())
			if not rows:
				continue
			words = []
			truncated = 0
			right_lang = 0
			for row in rows:
				words.append(row["words"])
				if row["truncated"]:
					truncated += 1
				if row["detected_lang"] == lang:
					right_lang += 1
			entry["conditions"][condition] = {
				"n": len(rows),
				"mean_words": metrics.mean(words),
				"median_words": metrics.median(words),
				"truncated_share": truncated / len(rows),
				"right_lang_share": right_lang / len(rows),
			}
		for condition in CLOSURE_CONDITIONS:
			triples = closure_triples(by_condition, condition)
			if triples:
				value = metrics.closure_from_triples(triples)
				low, high = metrics.bootstrap_closure(triples, config.N_BOOTSTRAP, config.SEED)
			else:
				value, low, high = float("nan"), float("nan"), float("nan")
			entry[f"{condition}_closure"] = (value, low, high, len(triples))
		summary.append(entry)
	return summary


def print_summary(summary):
	for entry in summary:
		print(f"\n===== {entry['lang']} =====")
		print(f"  {'условие':<30}{'n':>5}{'среднее':>10}{'медиана':>10}{'обрезано':>10}{'нужный язык':>13}")
		for condition in CONDITIONS:
			stats = entry["conditions"].get(condition)
			if stats is None:
				continue
			print(
				f"  {CONDITION_NAMES[condition]:<30}{stats['n']:>5}{stats['mean_words']:>10.0f}"
				f"{stats['median_words']:>10.0f}{fmt_percent(stats['truncated_share']):>10}"
				f"{fmt_percent(stats['right_lang_share']):>13}"
			)
		for condition in CLOSURE_CONDITIONS:
			value, low, high, n = entry[f"{condition}_closure"]
			if n == 0 and condition != "neologism":
				continue
			print(f"  доля закрытого разрыва, {CONDITION_NAMES[condition]}: {fmt_percent(value)} "
				f"(95% интервал {fmt_percent(low)} … {fmt_percent(high)}, вопросов: {n})")


def write_summary_csv(summary, path):
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			"lang", "condition", "n", "mean_words", "median_words", "truncated_share",
			"right_lang_share", "gap_closure", "ci_low", "ci_high",
		])
		for entry in summary:
			for condition in CONDITIONS:
				stats = entry["conditions"].get(condition)
				if stats is None:
					continue
				closure = ("", "", "")
				key = f"{condition}_closure"
				if key in entry:
					closure = entry[key][:3]
				writer.writerow([
					entry["lang"], condition, stats["n"],
					round(stats["mean_words"], 1), stats["median_words"],
					round(stats["truncated_share"], 3), round(stats["right_lang_share"], 3),
					closure[0], closure[1], closure[2],
				])


def style_axis(ax):
	ax.set_facecolor(SURFACE)
	for side in ["top", "right", "left"]:
		ax.spines[side].set_visible(False)
	ax.spines["bottom"].set_color(GRID_COLOR)
	ax.tick_params(colors=TEXT_SECONDARY, length=0)
	ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
	ax.set_axisbelow(True)


def plot_closure(summary, path, plt):
	"""Один график — одна мера: доля закрытого разрыва по языкам, с 95% интервалом."""
	langs = []
	for entry in summary:
		langs.append(entry["lang"])
	fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
	style_axis(ax)
	series = []
	for condition in CLOSURE_CONDITIONS:
		color = CLOSURE_COLORS[condition]
		has_value = False
		for entry in summary:
			if not math.isnan(entry[f"{condition}_closure"][0]):
				has_value = True
		if has_value:
			series.append((condition, color))
	width = 0.72 / len(series)
	top = 100.0
	bottom = 0.0
	for k, (condition, color) in enumerate(series):
		xs = []
		values = []
		errors_low = []
		errors_high = []
		for i, entry in enumerate(summary):
			value, low, high, n = entry[f"{condition}_closure"]
			if math.isnan(value):
				continue
			if math.isnan(low):
				low = value
			if math.isnan(high):
				high = value
			xs.append(i + (k - (len(series) - 1) / 2) * width)
			values.append(value * 100)
			errors_low.append(max(0.0, (value - low) * 100))
			errors_high.append(max(0.0, (high - value) * 100))
		ax.bar(xs, values, width=width * 0.92, color=color, edgecolor=SURFACE, linewidth=2,
			label=CONDITION_NAMES[condition], zorder=2)
		ax.errorbar(xs, values, yerr=[errors_low, errors_high], fmt="none", ecolor=TEXT_SECONDARY,
			elinewidth=1.2, capsize=4, zorder=3)
		# Подпись ставим за концом «уса», чтобы она не наезжала на интервал.
		for x, value, err_low, err_high in zip(xs, values, errors_low, errors_high):
			if value >= 0:
				anchor = value + err_high
				offset = 4
				vertical = "bottom"
			else:
				anchor = value - err_low
				offset = -4
				vertical = "top"
			top = max(top, anchor)
			bottom = min(bottom, anchor)
			ax.annotate(f"{value:.0f}%", (x, anchor), xytext=(0, offset), textcoords="offset points",
				ha="center", va=vertical, fontsize=9 if len(series) <= 3 else 7, color=TEXT_PRIMARY, zorder=4)
	ax.axhline(100, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1, label="100% = как явная просьба")
	ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=1)
	ax.set_ylim(min(-10.0, bottom - 12), top + 15)
	ax.set_xticks(list(range(len(langs))))
	ax.set_xticklabels(langs, color=TEXT_PRIMARY)
	ax.set_ylabel("доля закрытого разрыва, %", color=TEXT_SECONDARY)
	ax.set_title("Насколько каждая приписка сокращает ответы по сравнению с явной просьбой на языке вопроса", color=TEXT_PRIMARY,
		fontsize=11, loc="left")
	ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
	fig.tight_layout()
	fig.savefig(path, dpi=160, facecolor=SURFACE)
	plt.close(fig)


def plot_words(summary, path, plt):
	"""Средняя длина ответа в словах: условия рядом внутри каждого языка."""
	fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
	style_axis(ax)
	present = []
	for condition in CONDITIONS:
		for entry in summary:
			if condition in entry["conditions"] and condition not in present:
				present.append(condition)
	width = 0.8 / len(present)
	for k, condition in enumerate(present):
		xs = []
		values = []
		for i, entry in enumerate(summary):
			stats = entry["conditions"].get(condition)
			if stats is None:
				continue
			xs.append(i + (k - (len(present) - 1) / 2) * width)
			values.append(stats["mean_words"])
		if not xs:
			continue
		ax.bar(xs, values, width=width * 0.92, color=WORD_COLORS[condition], edgecolor=SURFACE, linewidth=2,
			label=CONDITION_NAMES[condition], zorder=2)
		for x, value in zip(xs, values):
			ax.annotate(f"{value:.0f}", (x, value), xytext=(0, 3), textcoords="offset points",
				ha="center", va="bottom", fontsize=8 if len(present) <= 5 else 6.5, color=TEXT_PRIMARY)
	langs = []
	for entry in summary:
		langs.append(entry["lang"])
	ax.set_xticks(list(range(len(langs))))
	ax.set_xticklabels(langs, color=TEXT_PRIMARY)
	ax.set_ylabel("средняя длина ответа, слов", color=TEXT_SECONDARY)
	ax.set_title("Длина ответов по условиям и языкам", color=TEXT_PRIMARY, fontsize=11, loc="left")
	ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
	fig.tight_layout()
	fig.savefig(path, dpi=160, facecolor=SURFACE)
	plt.close(fig)


def print_examples(grouped, n=2):
	print("\nПримеры ответов со словом:")
	for lang, by_condition in grouped.items():
		rows = list(by_condition.get("neologism", {}).values())
		for row in rows[:n]:
			print(f"  [{lang}, {row['words']} слов] {row['question'][:70]!r}")
			print(f"      {row['response'][:250]!r}")


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--run", required=True)
	parser.add_argument("--file", default="generations.jsonl")
	args = parser.parse_args()

	rows = files.read_jsonl(files.run_path(args.run, args.file))
	grouped = group_rows(rows)
	summary = summarize(grouped)
	print_summary(summary)
	print_examples(grouped)

	run_dir = os.path.join(config.RUNS_DIR, args.run)
	write_summary_csv(summary, os.path.join(run_dir, "summary.csv"))
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		print("\nmatplotlib не установлен — графики пропущены (pip install matplotlib)")
		return
	plot_closure(summary, os.path.join(run_dir, "gap_closure.png"), plt)
	plot_words(summary, os.path.join(run_dir, "words.png"), plt)
	print(f"\nСохранено в {run_dir}: summary.csv, gap_closure.png, words.png")


if __name__ == "__main__":
	main()
