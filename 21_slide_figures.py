"""Крупные графики для презентации (видеокарта не нужна): берёт уже посчитанные результаты и рисует
по одной картинке на слайд, без заголовков (заголовок — на слайде), с прозрачным фоном.

Цвета языков во всех графиках одни и те же: en — синий, ru — оранжевый, kk — бирюзовый.

Запуск: python 21_slide_figures.py
Результат: presentation/*.png
"""

import csv
import os

import config

OUT_DIR = "presentation"
INK = "#1d2433"
MUTED = "#4a5262"
GRID = "#d9d6cc"
GREY = "#a3a29c"
LANG_COLORS = {"en": "#2a78d6", "ru": "#eb6834", "kk": "#1baf7a"}
LANGS = ["en", "ru", "kk"]
# Стартовые слова ~card и их пробеги (несколько — повторы с другим зерном).
START_GROUPS = [
	("concise", ["card_concise", "card_concise_s1", "card_concise_s2"]),
	("brief", ["card_brief"]),
	("accurate", ["card_en", "card_accurate_s1", "card_accurate_s2"]),
	("the", ["card_the"]),
	("random", ["card_random"]),
]
RUN_LABELS = {
	"card_en": "accurate s0",
	"card_accurate_s1": "accurate s1",
	"card_accurate_s2": "accurate s2",
	"card_concise": "concise s0",
	"card_concise_s1": "concise s1",
	"card_concise_s2": "concise s2",
	"card_concise_lr003": "concise, шаг 0,003",
	"card_brief": "brief",
	"card_the": "the",
	"card_random": "random",
}


def setup():
	"""Стиль графиков для слайдов."""
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	plt.rcParams.update({
		"font.family": ["Inter", "DejaVu Sans"],
		"font.size": 17,
		"axes.edgecolor": GRID,
		"axes.labelcolor": MUTED,
		"axes.titlecolor": INK,
		"xtick.color": MUTED,
		"ytick.color": MUTED,
		"text.color": INK,
		"axes.spines.top": False,
		"axes.spines.right": False,
		"axes.grid": True,
		"axes.grid.axis": "y",
		"grid.color": GRID,
		"grid.linewidth": 0.8,
		"axes.axisbelow": True,
		"savefig.transparent": True,
	})
	return plt


def read_csv(path):
	with open(path, encoding="utf-8") as f:
		return list(csv.DictReader(f))


def run_summary(run):
	"""{(язык, условие): строка summary.csv}."""
	path = os.path.join(config.RUNS_DIR, run, "summary.csv")
	if not os.path.exists(path):
		return None
	result = {}
	for row in read_csv(path):
		result[(row["lang"], row["condition"])] = row
	return result


def value(row, key):
	"""Значение из summary.csv в процентах или None."""
	if row is None or not row.get(key):
		return None
	return float(row[key]) * 100


def save(fig, name):
	path = os.path.join(OUT_DIR, name)
	fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
	print(f"  {path}")


def short_transfer(plt):
	"""~short: доля разрыва для каждого языка обучения и проверки."""
	rows = [("en", "b002", 467), ("ru", "from_ru", 591), ("kk", "from_kk", 574)]
	fig, ax = plt.subplots(figsize=(9, 6.2))
	ax.grid(False)
	for side in ["left", "bottom"]:
		ax.spines[side].set_visible(False)
	from matplotlib.colors import LinearSegmentedColormap
	cmap = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#2a78d6", "#0f3f80"])
	for i, (train_lang, run, n_pairs) in enumerate(rows):
		summary = run_summary(run)
		for j, test_lang in enumerate(LANGS):
			row = summary.get((test_lang, "neologism")) if summary else None
			closure = value(row, "gap_closure")
			low = value(row, "ci_low")
			high = value(row, "ci_high")
			if closure is None:
				continue
			share = max(0.0, min(1.0, closure / 105))
			ax.add_patch(plt.Rectangle((j, 2 - i), 0.97, 0.97, color=cmap(share)))
			text_color = "#fbfaf6" if share > 0.55 else INK
			weight = "bold" if train_lang == test_lang else "normal"
			ax.text(j + 0.485, 2 - i + 0.58, f"{closure:.0f}%", ha="center", va="center", fontsize=30,
				color=text_color, fontweight=weight)
			ax.text(j + 0.485, 2 - i + 0.3, f"[{low:.0f}…{high:.0f}]", ha="center", va="center", fontsize=16,
				color=text_color)
	ax.set_xlim(0, 3)
	ax.set_ylim(0, 3)
	ax.set_xticks([0.485, 1.485, 2.485])
	ax.set_xticklabels(["en", "ru", "kk"], fontsize=22)
	ax.xaxis.tick_top()
	ax.set_yticks([2.485, 1.485, 0.485])
	labels = []
	for train_lang, run, n_pairs in rows:
		labels.append(f"{train_lang}\n{n_pairs} пар")
	ax.set_yticklabels(labels, fontsize=20)
	ax.tick_params(length=0)
	ax.set_xlabel("язык проверки", fontsize=18, labelpad=12)
	ax.xaxis.set_label_position("top")
	ax.set_ylabel("язык обучения", fontsize=18, labelpad=12)
	save(fig, "short_transfer.png")
	plt.close(fig)


def bar_labels(ax, xs, ys, offset=2.0, size=15):
	"""Подписи значений над столбцами (под столбцом для отрицательных)."""
	for x, y in zip(xs, ys):
		if y is None:
			continue
		place = y + offset if y >= 0 else y - offset
		ax.text(x, place, f"{y:.0f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=size, color=INK)


def card_first(plt):
	"""Первый результат ~card: старт accurate против старта concise."""
	groups = [("старт accurate", "card_en"), ("старт concise", "card_concise")]
	fig, ax = plt.subplots(figsize=(11, 5.6))
	width = 0.26
	for g, (label, run) in enumerate(groups):
		summary = run_summary(run)
		for k, lang in enumerate(LANGS):
			y = value(summary.get((lang, "neologism")), "score_closure")
			x = g + (k - 1) * width
			ax.bar([x], [y], width=width * 0.92, color=LANG_COLORS[lang], label=lang if g == 0 else None)
			bar_labels(ax, [x], [y], size=18)
	ax.axhline(100, color=MUTED, linewidth=1.5, linestyle="--")
	ax.text(1.42, 101.5, "явная инструкция = 100%", ha="right", va="bottom", fontsize=16, color=MUTED)
	ax.axhline(0, color=GREY, linewidth=1)
	ax.set_xticks([0, 1])
	ax.set_xticklabels([groups[0][0], groups[1][0]], fontsize=20)
	ax.set_ylim(-15, 126)
	ax.set_yticks([0, 20, 40, 60, 80, 100])
	ax.set_ylabel("доля закрытого разрыва, %")
	ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=17)
	save(fig, "card_first.png")
	plt.close(fig)


def start_words(plt):
	"""~card для всех стартовых слов: среднее по зёрнам, точки — отдельные зёрна, черта — без обучения."""
	fig, ax = plt.subplots(figsize=(15, 6.4))
	width = 0.26
	for g, (word, runs) in enumerate(START_GROUPS):
		for k, lang in enumerate(LANGS):
			values = []
			controls = []
			for run in runs:
				summary = run_summary(run)
				if summary is None:
					continue
				values.append(value(summary.get((lang, "neologism")), "score_closure"))
				controls.append(value(summary.get((lang, "control")), "score_closure"))
			if not values:
				continue
			x = g + (k - 1) * width
			mean = sum(values) / len(values)
			ax.bar([x], [mean], width=width * 0.92, color=LANG_COLORS[lang], label=lang if g == 0 else None)
			if len(values) > 1:
				dots_x = []
				for _ in values:
					dots_x.append(x)
				ax.scatter(dots_x, values, s=34, color="#fbfaf6", edgecolor=INK, linewidth=1.2, zorder=4,
					label="отдельные зёрна" if (g == 0 and k == 0) else None)
			control = controls[0]
			ax.plot([x - width * 0.46, x + width * 0.46], [control, control], color=INK, linewidth=2.6,
				solid_capstyle="butt", zorder=5, label="без обучения" if (g == 0 and k == 0) else None)
			if mean >= 0:
				top = max(max(values), mean, control)
				ax.text(x, top + 2.5, f"{mean:.0f}", ha="center", va="bottom", fontsize=15, color=INK)
			else:
				bottom = min(min(values), mean, control)
				ax.text(x, bottom - 2.5, f"{mean:.0f}", ha="center", va="top", fontsize=15, color=INK)
	ax.axhline(0, color=GREY, linewidth=1)
	ax.set_xticks(range(len(START_GROUPS)))
	labels = []
	for word, runs in START_GROUPS:
		labels.append(f"{word}\n({len(runs)} зерна)" if len(runs) > 1 else word)
	ax.set_xticklabels(labels, fontsize=19)
	ax.set_ylim(-20, 112)
	ax.set_ylabel("доля закрытого разрыва, %")
	ax.legend(frameon=False, ncol=5, loc="upper right", fontsize=15)
	save(fig, "start_words.png")
	plt.close(fig)


def start_vs_training(plt):
	"""Что дало само стартовое слово и что добавило обучение (по баллу)."""
	path = os.path.join("analysis", "card_decomposition.csv")
	parts = {}
	for row in read_csv(path):
		if row["measure"] == "score":
			parts[(row["run"], row["lang"])] = (float(row["init_part"]) * 100, float(row["trained_part"]) * 100)
	groups = [("concise", "card_concise"), ("brief", "card_brief"), ("accurate (зерно 1)", "card_accurate_s1")]
	fig, ax = plt.subplots(figsize=(15, 6.2))
	width = 0.14
	ticks = []
	tick_labels = []
	for g, (label, run) in enumerate(groups):
		for k, lang in enumerate(LANGS):
			if (run, lang) not in parts:
				continue
			init_part, trained_part = parts[(run, lang)]
			center = g + (k - 1) * 0.31
			ax.bar([center - width / 2], [init_part], width=width * 0.92, color=GREY,
				label="дало само слово" if (g == 0 and k == 0) else None)
			ax.bar([center + width / 2], [trained_part], width=width * 0.92, color=LANG_COLORS[lang],
				label=f"добавило обучение ({lang})" if g == 0 else None)
			bar_labels(ax, [center - width / 2, center + width / 2], [init_part, trained_part], offset=1.5, size=13)
			ticks.append(center)
			tick_labels.append(lang)
	ax.axhline(0, color=GREY, linewidth=1)
	ax.set_xticks(ticks)
	ax.set_xticklabels(tick_labels, fontsize=16)
	for g, (label, run) in enumerate(groups):
		ax.text(g, -38, label, ha="center", va="top", fontsize=20, color=INK)
	ax.set_ylim(-30, 102)
	ax.set_ylabel("доля разрыва по баллу, %")
	ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=14)
	save(fig, "start_vs_training.png")
	plt.close(fig)


def dose(plt):
	"""Балл при векторе e + α·Δ: пик ровно на обученном векторе."""
	panels = [("concise", "card_concise"), ("accurate, зерно 1", "card_accurate_s1")]
	fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
	for ax, (title, run) in zip(axes, panels):
		rows = read_csv(os.path.join("analysis", f"delta_scale_{run}.csv"))
		for lang in LANGS:
			xs = []
			ys = []
			for row in rows:
				if row["lang"] == lang:
					xs.append(float(row["alpha"]))
					ys.append(float(row["score_closure"]) * 100)
			ax.plot(xs, ys, color=LANG_COLORS[lang], linewidth=3, marker="o", markersize=8, label=lang)
		ax.axvline(1, color=MUTED, linewidth=1.3, linestyle=":")
		ax.text(1.04, 104, "обученный\nвектор", fontsize=14, color=MUTED, va="top")
		ax.axhline(0, color=GREY, linewidth=1)
		ax.set_title(title, loc="left", fontsize=20, pad=10)
		ax.set_xticks([0, 0.25, 0.5, 1, 1.5, 2])
		ax.set_xticklabels(["0", "¼", "½", "1", "1½", "2"])
		ax.set_xlabel("доза добавки α  (вектор = слово + α · добавка)")
	axes[0].set_ylabel("доля разрыва по баллу, %")
	axes[0].set_ylim(-15, 108)
	axes[0].legend(frameon=False, loc="upper left", fontsize=16)
	save(fig, "dose.png")
	plt.close(fig)


def transplant(plt):
	"""Пересадка добавки на другое слово (английский)."""
	rows = read_csv(os.path.join("analysis", "delta_transplant_card_accurate_s1_vs_card_concise.csv"))
	by_name = {}
	for row in rows:
		if row["lang"] == "en":
			by_name[row["variant"]] = float(row["score_closure"]) * 100
	items = [
		("concise + своя добавка", "concise + свой Δ (обученный)", "own"),
		("accurate + своя добавка", "accurate + свой Δ (обученный)", "own"),
		("concise без добавки", "concise (без сдвига)", "none"),
		("accurate без добавки", "accurate (без сдвига)", "none"),
		("concise + добавка от accurate", "concise + Δ от accurate", "other"),
		("accurate + добавка от concise", "accurate + Δ от concise", "other"),
		("the + добавка от concise", "the + Δ от concise", "other"),
		("the + добавка от accurate", "the + Δ от accurate", "other"),
		("только добавка от concise", "только Δ от concise", "other"),
		("только добавка от accurate", "только Δ от accurate", "other"),
	]
	colors = {"own": LANG_COLORS["en"], "none": GREY, "other": "#9cc0ea"}
	fig, ax = plt.subplots(figsize=(13, 6.6))
	ax.grid(axis="x")
	ax.grid(axis="y", visible=False)
	labels = []
	for i, (label, key, kind) in enumerate(items):
		y = len(items) - 1 - i
		x = by_name.get(key)
		if x is None:
			continue
		ax.barh([y], [x], height=0.68, color=colors[kind])
		ax.text(x + (1.5 if x >= 0 else -1.5), y, f"{x:.0f}", va="center", ha="left" if x >= 0 else "right",
			fontsize=15, color=INK)
		labels.append((y, label))
	ys = []
	names = []
	for y, label in labels:
		ys.append(y)
		names.append(label)
	ax.set_yticks(ys)
	ax.set_yticklabels(names, fontsize=16)
	ax.axvline(0, color=GREY, linewidth=1)
	ax.set_xlim(-20, 100)
	ax.set_xlabel("доля разрыва по баллу в английском, %")
	ax.tick_params(axis="y", length=0)
	save(fig, "transplant.png")
	plt.close(fig)


def read_log(run):
	path = os.path.join(config.RUNS_DIR, run, "log.csv")
	xs = []
	ys = []
	for row in read_csv(path):
		xs.append(int(row["pairs_done"]))
		ys.append(float(row["loss"]))
	return xs, ys


def loss_curves(plt):
	"""Потеря при обучении у удачных и провальных пробегов."""
	runs = [
		("card_concise", "concise", True),
		("card_brief", "brief", True),
		("card_accurate_s1", "accurate s1", True),
		("card_en", "accurate s0", False),
		("card_random", "random", False),
	]
	fig, ax = plt.subplots(figsize=(14, 6.2))
	ends = []
	for run, label, good in runs:
		if not os.path.exists(os.path.join(config.RUNS_DIR, run, "log.csv")):
			continue
		xs, ys = read_log(run)
		color = INK if good else "#c4472a"
		style = "-" if good else "--"
		ax.plot(xs, ys, color=color, linewidth=2.4, linestyle=style)
		summary = run_summary(run)
		closure = value(summary.get(("en", "neologism")), "score_closure") if summary else None
		tail = ys[-6:]
		ends.append([sum(tail) / len(tail), f"{label}: поведение {closure:.0f}%", color])
	ends.sort(key=lambda item: item[0])
	last = -1.0
	for item in ends:
		if item[0] < last + 0.075:
			item[0] = last + 0.075
		last = item[0]
		ax.text(1165, item[0], item[1], va="center", fontsize=15, color=item[2])
	ax.axhline(2 * 0.6931, color=GREY, linewidth=1, linestyle=":")
	ax.text(1140, 1.41, "старт: 1,386", fontsize=13, color=MUTED, va="bottom", ha="right")
	ax.axvline(575, color=MUTED, linewidth=1.2, linestyle=":")
	ax.text(590, 1.25, "2-я эпоха:\nте же пары\nвторой раз", fontsize=14, color=MUTED, va="top")
	ax.set_xlim(0, 1150)
	ax.set_ylim(0, 1.62)
	ax.set_xlabel("пар пройдено при обучении")
	ax.set_ylabel("потеря")
	save(fig, "loss_curves.png")
	plt.close(fig)


def first_token(plt):
	"""Вопрос 2: где модель сходит с ответа-карточки — и что она в итоге пишет."""
	rows = read_csv(os.path.join("analysis", "likelihood_vs_generation.csv"))
	offsets = {
		"card_concise": (12, -4),
		"card_concise_s1": (12, 4),
		"card_concise_s2": (-14, 8),
		"card_concise_lr003": (12, -10),
		"card_brief": (12, 2),
		"card_accurate_s1": (12, 0),
		"card_accurate_s2": (-14, 6),
		"card_en": (-14, 8),
		"card_the": (-14, 6),
		"card_random": (-14, -10),
	}
	fig, ax = plt.subplots(figsize=(13, 6.6))
	ax.grid(axis="both")
	for row in rows:
		if row["lang"] != "en" or not row.get("closure"):
			continue
		x = float(row["first_token_miss_share"]) * 100
		y = float(row["closure"]) * 100
		good = y >= 50
		color = INK if good else "#c4472a"
		ax.scatter([x], [y], s=130, color=color, edgecolor="#fbfaf6", linewidth=2, zorder=4)
		dx, dy = offsets.get(row["run"], (10, 0))
		ax.annotate(RUN_LABELS.get(row["run"], row["run"]), (x, y), textcoords="offset points",
			xytext=(dx, dy), ha="left" if dx > 0 else "right", va="center", fontsize=15, color=color)
	ax.text(2, 4, "● удачные пробеги (поведение ≥ 50%)", fontsize=15, color=INK, va="bottom")
	ax.text(2, -2, "● провальные пробеги", fontsize=15, color="#c4472a", va="bottom")
	ax.set_xlim(0, 105)
	ax.set_ylim(-5, 110)
	ax.set_xlabel("доля вопросов, где модель уходит с ответа-карточки уже на первом токене, %")
	ax.set_ylabel("поведение: доля разрыва, %")
	save(fig, "first_token.png")
	plt.close(fig)


def main():
	os.makedirs(OUT_DIR, exist_ok=True)
	plt = setup()
	print("Рисую:")
	short_transfer(plt)
	card_first(plt)
	start_words(plt)
	start_vs_training(plt)
	dose(plt)
	transplant(plt)
	loss_curves(plt)
	first_token(plt)


if __name__ == "__main__":
	main()
