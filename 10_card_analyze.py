"""Шаг 6 для ~card: насколько ответы похожи на «ответ-карточку» в каждом условии.

Для каждого ответа считаются 4 условия формата (neolib/card.py) и балл — доля выполненных.
Для каждого языка и условия:
	* доля ответов, выполнивших каждое условие, средний балл, доля «всё выполнено»;
	* доля закрытого разрыва по баллу = (условие − база) / (явная просьба − база), с 95% интервалом;
	то же по доле «всё выполнено»;
	* для неологизма — доля закрытого разрыва по каждому условию отдельно
	(какие части понятия переносятся в другой язык, а какие нет).

Запуск: python 10_card_analyze.py --run card_en
Результат в runs/{run}/: summary.csv, criteria.csv, card_closure.png, card_criteria.png
"""

import argparse
import csv
import math
import os

import config
from neolib import card, files, metrics

CONDITIONS = ["base", "neologism", "target", "target_en", "four", "short_en", "short_native", "control"]
CONDITION_NAMES = {
	"base": "без приписки",
	"neologism": "с ~card",
	"target": "явная инструкция",
	"target_en": "англ. инструкция",
	"four": "«в четырёх предложениях»",
	"short_en": "англ. «short»",
	"short_native": "«короткий» на языке вопроса",
	"control": "необученный вектор",
}
CLOSURE_CONDITIONS = ["neologism", "target_en", "four", "short_en", "short_native", "control"]
MEASURES = ["score", "full"] + card.CRITERIA

COLORS = {
	"base": "#2a78d6",
	"neologism": "#eb6834",
	"target": "#1baf7a",
	"target_en": "#7a6fd0",
	"four": "#3fa7c4",
	"short_en": "#c2569b",
	"short_native": "#8a6d3b",
	"control": "#a3a29c",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def measure_values(row):
	"""Все меры одного ответа: балл, «всё выполнено» и каждое условие (1.0 или 0.0)."""
	result = card.check(row["response"])
	values = {"score": result["score"], "n_sentences": result["n_sentences"]}
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


def group_rows(rows):
	"""{язык: {условие: {id: строка с мерами}}}"""
	grouped = {}
	for row in rows:
		row = dict(row)
		row["measures"] = measure_values(row)
		by_lang = grouped.setdefault(row["lang"], {})
		by_condition = by_lang.setdefault(row["condition"], {})
		by_condition[row["id"]] = row
	return grouped


def triples_for(by_condition, condition, measure):
	triples = []
	for question_id in by_condition.get("base", {}):
		if question_id not in by_condition.get(condition, {}):
			continue
		if question_id not in by_condition.get("target", {}):
			continue
		triples.append((
			by_condition["base"][question_id]["measures"][measure],
			by_condition[condition][question_id]["measures"][measure],
			by_condition["target"][question_id]["measures"][measure],
		))
	return triples


def closure_with_ci(by_condition, condition, measure):
	triples = triples_for(by_condition, condition, measure)
	if not triples:
		return float("nan"), float("nan"), float("nan"), 0
	value = card.closure_from_triples(triples)
	low, high = card.bootstrap_closure(triples, config.N_BOOTSTRAP, config.SEED)
	return value, low, high, len(triples)


def ordered_langs(grouped):
	langs = []
	for lang in config.LANGS:
		if lang in grouped:
			langs.append(lang)
	for lang in grouped:
		if lang not in langs:
			langs.append(lang)
	return langs


def summarize(grouped):
	summary = []
	for lang in ordered_langs(grouped):
		by_condition = grouped[lang]
		entry = {"lang": lang, "conditions": {}, "closure": {}}
		for condition in CONDITIONS:
			rows = list(by_condition.get(condition, {}).values())
			if not rows:
				continue
			stats = {"n": len(rows)}
			words = []
			sentences = []
			right_lang = 0
			sums = {}
			for measure in MEASURES:
				sums[measure] = 0.0
			for row in rows:
				words.append(row["words"])
				sentences.append(row["measures"]["n_sentences"])
				if row["detected_lang"] == lang:
					right_lang += 1
				for measure in MEASURES:
					sums[measure] += row["measures"][measure]
			stats["mean_words"] = metrics.mean(words)
			stats["median_sentences"] = metrics.median(sentences)
			stats["right_lang_share"] = right_lang / len(rows)
			for measure in MEASURES:
				stats[measure] = sums[measure] / len(rows)
			entry["conditions"][condition] = stats
		for condition in CLOSURE_CONDITIONS:
			if condition not in entry["conditions"]:
				continue
			entry["closure"][condition] = {}
			for measure in MEASURES:
				entry["closure"][condition][measure] = closure_with_ci(by_condition, condition, measure)
		summary.append(entry)
	return summary


def fmt_percent(value):
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return "—"
	return f"{value * 100:.0f}%"


def fmt_closure(item):
	value, low, high, n = item
	if math.isnan(value):
		return "—"
	return f"{fmt_percent(value)} [{fmt_percent(low)} … {fmt_percent(high)}]"


def print_summary(summary):
	for entry in summary:
		print(f"\n===== {entry['lang']} =====")
		header = f"  {'условие':<30}{'n':>4}{'слов':>7}{'предл.':>8}{'язык':>7}"
		for name in card.CRITERIA:
			header += f"{name:>8}"
		header += f"{'балл':>7}{'всё':>7}"
		print(header)
		for condition in CONDITIONS:
			stats = entry["conditions"].get(condition)
			if stats is None:
				continue
			line = (f"  {CONDITION_NAMES[condition]:<30}{stats['n']:>4}{stats['mean_words']:>7.0f}"
				f"{stats['median_sentences']:>8.0f}{fmt_percent(stats['right_lang_share']):>7}")
			for name in card.CRITERIA:
				line += f"{fmt_percent(stats[name]):>8}"
			line += f"{fmt_percent(stats['score']):>7}{fmt_percent(stats['full']):>7}"
			print(line)
		print("  доля закрытого разрыва по баллу (и по «всё выполнено»):")
		for condition in CLOSURE_CONDITIONS:
			if condition not in entry["closure"]:
				continue
			by_measure = entry["closure"][condition]
			print(f"    {CONDITION_NAMES[condition]:<30}{fmt_closure(by_measure['score'])}   "
				f"всё: {fmt_closure(by_measure['full'])}")
		if "neologism" in entry["closure"]:
			print("  неологизм по отдельным условиям:")
			for name in card.CRITERIA:
				print(f"    {card.CRITERION_NAMES[name]:<30}{fmt_closure(entry['closure']['neologism'][name])}")


def write_csvs(summary, run_dir):
	with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		header = ["lang", "condition", "n", "mean_words", "median_sentences", "right_lang_share"]
		for measure in MEASURES:
			header.append(measure)
		header = header + ["score_closure", "score_ci_low", "score_ci_high", "full_closure", "full_ci_low", "full_ci_high"]
		writer.writerow(header)
		for entry in summary:
			for condition in CONDITIONS:
				stats = entry["conditions"].get(condition)
				if stats is None:
					continue
				line = [entry["lang"], condition, stats["n"], round(stats["mean_words"], 1),
					stats["median_sentences"], round(stats["right_lang_share"], 3)]
				for measure in MEASURES:
					line.append(round(stats[measure], 3))
				closure = entry["closure"].get(condition)
				if closure:
					line = line + list(closure["score"][:3]) + list(closure["full"][:3])
				else:
					line = line + ["", "", "", "", "", ""]
				writer.writerow(line)
	with open(os.path.join(run_dir, "criteria.csv"), "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["lang", "condition", "criterion", "closure", "ci_low", "ci_high", "n"])
		for entry in summary:
			for condition in CLOSURE_CONDITIONS:
				closure = entry["closure"].get(condition)
				if not closure:
					continue
				for name in card.CRITERIA:
					value, low, high, n = closure[name]
					writer.writerow([entry["lang"], condition, name, value, low, high, n])


def style_axis(ax):
	ax.set_facecolor(SURFACE)
	for side in ["top", "right", "left"]:
		ax.spines[side].set_visible(False)
	ax.spines["bottom"].set_color(GRID_COLOR)
	ax.tick_params(colors=TEXT_SECONDARY, length=0)
	ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
	ax.set_axisbelow(True)


def plot_closure(summary, path, plt):
	"""Доля закрытого разрыва по баллу: языки по оси x, условия рядом, с 95% интервалом."""
	fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
	style_axis(ax)
	series = []
	for condition in CLOSURE_CONDITIONS:
		for entry in summary:
			item = entry["closure"].get(condition)
			if item and not math.isnan(item["score"][0]) and condition not in series:
				series.append(condition)
	if not series:
		plt.close(fig)
		return
	width = 0.8 / len(series)
	top = 100.0
	bottom = 0.0
	for k, condition in enumerate(series):
		xs = []
		values = []
		errors_low = []
		errors_high = []
		for i, entry in enumerate(summary):
			item = entry["closure"].get(condition)
			if not item:
				continue
			value, low, high, n = item["score"]
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
			top = max(top, high * 100)
			bottom = min(bottom, low * 100)
		ax.bar(xs, values, width=width * 0.92, color=COLORS[condition], edgecolor=SURFACE, linewidth=2,
			label=CONDITION_NAMES[condition], zorder=2)
		ax.errorbar(xs, values, yerr=[errors_low, errors_high], fmt="none", ecolor=TEXT_SECONDARY,
			elinewidth=1.1, capsize=3, zorder=3)
		for x, value, err_high in zip(xs, values, errors_high):
			ax.annotate(f"{value:.0f}%", (x, value + err_high), xytext=(0, 3), textcoords="offset points",
				ha="center", va="bottom", fontsize=7, color=TEXT_PRIMARY, zorder=4)
	ax.axhline(100, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1, label="100% = как явная инструкция")
	ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=1)
	ax.set_ylim(min(-10.0, bottom - 10), top + 15)
	langs = []
	for entry in summary:
		langs.append(entry["lang"])
	ax.set_xticks(list(range(len(langs))))
	ax.set_xticklabels(langs, color=TEXT_PRIMARY)
	ax.set_ylabel("доля закрытого разрыва по баллу, %", color=TEXT_SECONDARY)
	ax.set_title("Насколько каждая приписка делает ответ «карточкой» по сравнению с явной инструкцией",
		color=TEXT_PRIMARY, fontsize=11, loc="left")
	ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4, fontsize=8)
	fig.tight_layout()
	fig.savefig(path, dpi=160, facecolor=SURFACE)
	plt.close(fig)


def plot_criteria(summary, path, plt):
	"""По панели на язык: доля ответов, выполнивших каждое условие, без приписки / с ~card / с инструкцией."""
	shown = ["base", "neologism", "target", "control"]
	groups = card.CRITERIA + ["full"]
	group_names = []
	for name in card.CRITERIA:
		group_names.append(card.CRITERION_NAMES[name])
	group_names.append("все 4")
	fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 4.4), facecolor=SURFACE, sharey=True)
	if len(summary) == 1:
		axes = [axes]
	width = 0.8 / len(shown)
	for ax, entry in zip(axes, summary):
		style_axis(ax)
		for k, condition in enumerate(shown):
			stats = entry["conditions"].get(condition)
			if stats is None:
				continue
			xs = []
			values = []
			for g, measure in enumerate(groups):
				xs.append(g + (k - (len(shown) - 1) / 2) * width)
				values.append(stats[measure] * 100)
			ax.bar(xs, values, width=width * 0.92, color=COLORS[condition], edgecolor=SURFACE, linewidth=1.5,
				label=CONDITION_NAMES[condition], zorder=2)
		ax.set_xticks(list(range(len(groups))))
		ax.set_xticklabels(group_names, color=TEXT_PRIMARY, fontsize=7, rotation=20, ha="right")
		ax.set_ylim(0, 105)
		ax.set_title(entry["lang"], color=TEXT_PRIMARY, fontsize=11, loc="left")
	axes[0].set_ylabel("доля ответов, %", color=TEXT_SECONDARY)
	axes[0].legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left", fontsize=7)
	fig.suptitle("Какие условия формата выполняются", color=TEXT_PRIMARY, fontsize=11, x=0.01, ha="left")
	fig.tight_layout()
	fig.savefig(path, dpi=160, facecolor=SURFACE)
	plt.close(fig)


def print_examples(grouped, n=2):
	print("\nПримеры ответов со словом:")
	for lang, by_condition in grouped.items():
		rows = list(by_condition.get("neologism", {}).values())
		for row in rows[:n]:
			failed = []
			for name in card.CRITERIA:
				if row["measures"][name] == 0.0:
					failed.append(name)
			label = "карточка OK" if not failed else "не выполнено: " + ", ".join(failed)
			print(f"  [{lang}, {row['measures']['n_sentences']} предл., {label}] {row['question'][:70]!r}")
			print(f"      {row['response'][:300]!r}")


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--run", required=True)
	args = parser.parse_args()

	rows = files.read_jsonl(files.run_path(args.run, "generations.jsonl"))
	grouped = group_rows(rows)
	summary = summarize(grouped)
	print_summary(summary)
	print_examples(grouped)

	run_dir = os.path.join(config.RUNS_DIR, args.run)
	write_csvs(summary, run_dir)
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		print("\nmatplotlib не установлен — графики пропущены (pip install matplotlib)")
		return
	plot_closure(summary, os.path.join(run_dir, "card_closure.png"), plt)
	plot_criteria(summary, os.path.join(run_dir, "card_criteria.png"), plt)
	print(f"\nСохранено в {run_dir}: summary.csv, criteria.csv, card_closure.png, card_criteria.png")


if __name__ == "__main__":
	main()
