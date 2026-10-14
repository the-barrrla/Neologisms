"""Два итоговых графика для README из уже посчитанных результатов (видеокарта не нужна).

	figures/transfer_matrix.png — ~short: на каком языке учили слово и где оно работает
		(доля закрытого разрыва, пробеги b002, from_ru, from_kk);
	figures/card_parts.png      — ~card: какие части формата переносятся в другие языки
		(пробег card_concise_lr003, доля разрыва по каждому условию и по общему баллу).

Запуск: python 11_figures.py   (после 06_analyze.py и 10_card_analyze.py)
"""

import csv
import json
import os

import config

LANGS = ["en", "ru", "kk"]
# Строки матрицы: язык обучения и пробег, на котором он учился.
TRANSFER_RUNS = [("en", "b002"), ("ru", "from_ru"), ("kk", "from_kk")]
CARD_RUN = "card_concise_lr003"
# «Первое предложение ≤ 20 слов» не показываем: без приписки оно и так выполняется в 77–93% ответов,
# разрыв слишком мал, и доля по нему ничего не говорит.
CARD_PARTS = [("count", "ровно 4\nпредложения"), ("rest", "остальные\n≤ 15 слов"), ("plain", "без\nразметки")]

LANG_COLORS = {"en": "#2a78d6", "ru": "#eb6834", "kk": "#1baf7a"}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def read_csv(path):
	with open(path, encoding="utf-8") as f:
		return list(csv.DictReader(f))


def run_file(run, name):
	return os.path.join(config.RUNS_DIR, run, name)


def neologism_closure(run):
	"""{язык проверки: (доля, низ, верх)} для условия neologism из summary.csv пробега ~short."""
	result = {}
	for row in read_csv(run_file(run, "summary.csv")):
		if row["condition"] == "neologism":
			result[row["lang"]] = (float(row["gap_closure"]), float(row["ci_low"]), float(row["ci_high"]))
	return result


def pairs_count(run):
	with open(run_file(run, "settings.json"), encoding="utf-8") as f:
		return json.load(f)["n_pairs"]


def percent_text(value):
	"""Целые проценты с типографским минусом."""
	return f"{value * 100:.0f}".replace("-", "−")


def interval_text(low, high):
	"""[78–91], а при отрицательной границе [−1…2], чтобы минус не путался с тире."""
	if low < 0:
		return f"[{percent_text(low)}…{percent_text(high)}]"
	return f"[{percent_text(low)}–{percent_text(high)}]"


def plot_transfer_matrix(path, plt):
	values = []
	labels = []
	for train_lang, run in TRANSFER_RUNS:
		closure = neologism_closure(run)
		row_values = []
		row_labels = []
		for test_lang in LANGS:
			value, low, high = closure[test_lang]
			row_values.append(value * 100)
			row_labels.append(f"{value * 100:.0f}%\n{interval_text(low, high)}")
		values.append(row_values)
		labels.append(row_labels)

	fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
	ax.set_facecolor(SURFACE)
	image = ax.imshow(values, cmap="Blues", vmin=-10, vmax=110)
	for i in range(len(TRANSFER_RUNS)):
		for j in range(len(LANGS)):
			color = "white" if values[i][j] > 60 else TEXT_PRIMARY
			weight = "bold" if i == j else "normal"
			ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight=weight)
	ax.set_xticks(range(len(LANGS)))
	ax.set_xticklabels(LANGS, color=TEXT_PRIMARY, fontsize=11)
	row_names = []
	for train_lang, run in TRANSFER_RUNS:
		row_names.append(f"{train_lang}  ({pairs_count(run)} пар)")
	ax.set_yticks(range(len(TRANSFER_RUNS)))
	ax.set_yticklabels(row_names, color=TEXT_PRIMARY, fontsize=11)
	ax.set_xlabel("язык проверки", color=TEXT_SECONDARY)
	ax.set_ylabel("язык обучения", color=TEXT_SECONDARY)
	ax.xaxis.set_label_position("top")
	ax.xaxis.tick_top()
	ax.tick_params(length=0)
	for side in ["top", "right", "left", "bottom"]:
		ax.spines[side].set_visible(False)
	bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
	bar.set_label("доля закрытого разрыва, %", color=TEXT_SECONDARY)
	bar.outline.set_visible(False)
	ax.set_title("~short: где работает слово, выученное на одном языке", color=TEXT_PRIMARY,
		fontsize=12, loc="left", pad=40)
	fig.tight_layout()
	fig.savefig(path, dpi=160, facecolor=SURFACE)
	plt.close(fig)


def card_values():
	"""{язык: [(доля, низ, верх) по CARD_PARTS ..., общий балл]}"""
	by_lang = {}
	for lang in LANGS:
		by_lang[lang] = {}
	for row in read_csv(run_file(CARD_RUN, "criteria.csv")):
		if row["condition"] == "neologism" and row["lang"] in by_lang:
			by_lang[row["lang"]][row["criterion"]] = (float(row["closure"]), float(row["ci_low"]), float(row["ci_high"]))
	for row in read_csv(run_file(CARD_RUN, "summary.csv")):
		if row["condition"] == "neologism" and row["lang"] in by_lang:
			by_lang[row["lang"]]["score"] = (float(row["score_closure"]), float(row["score_ci_low"]), float(row["score_ci_high"]))
	return by_lang


def plot_card_parts(path, plt):
	by_lang = card_values()
	groups = []
	for key, name in CARD_PARTS:
		groups.append((key, name))
	groups.append(("score", "общий балл"))

	fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
	ax.set_facecolor(SURFACE)
	for side in ["top", "right", "left"]:
		ax.spines[side].set_visible(False)
	ax.spines["bottom"].set_color(GRID_COLOR)
	ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
	ax.set_axisbelow(True)
	ax.tick_params(colors=TEXT_SECONDARY, length=0)
	width = 0.26
	for k, lang in enumerate(LANGS):
		xs = []
		heights = []
		errors_low = []
		errors_high = []
		for g, (key, name) in enumerate(groups):
			value, low, high = by_lang[lang][key]
			xs.append(g + (k - 1) * width)
			heights.append(value * 100)
			errors_low.append(max(0.0, (value - low) * 100))
			errors_high.append(max(0.0, (high - value) * 100))
		ax.bar(xs, heights, width * 0.92, color=LANG_COLORS[lang], label=lang, zorder=2)
		ax.errorbar(xs, heights, yerr=[errors_low, errors_high], fmt="none", ecolor=TEXT_SECONDARY,
			elinewidth=1, capsize=3, zorder=3)
		for x, height, err_high in zip(xs, heights, errors_high):
			ax.annotate(f"{height:.0f}%", (x, height + err_high), xytext=(0, 2), textcoords="offset points",
				ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
	ax.axhline(100, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
	ax.set_xticks(range(len(groups)))
	names = []
	for key, name in groups:
		names.append(name)
	ax.set_xticklabels(names, color=TEXT_PRIMARY)
	ax.set_ylim(0, 118)
	ax.set_ylabel("доля закрытого разрыва, %", color=TEXT_SECONDARY)
	ax.set_title("~card (обучен на en): какие части формата переносятся", color=TEXT_PRIMARY, fontsize=12, loc="left")
	ax.legend(frameon=False, ncol=3, loc="upper left", labelcolor=TEXT_PRIMARY)
	fig.tight_layout()
	fig.savefig(path, dpi=160, facecolor=SURFACE)
	plt.close(fig)


def main():
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	os.makedirs("figures", exist_ok=True)
	plot_transfer_matrix(os.path.join("figures", "transfer_matrix.png"), plt)
	plot_card_parts(os.path.join("figures", "card_parts.png"), plt)
	print("Сохранено: figures/transfer_matrix.png, figures/card_parts.png")


if __name__ == "__main__":
	main()
