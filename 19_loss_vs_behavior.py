"""Вопрос 2: почему потеря падает, а поведение меняется мало (видеокарта не нужна, только log.csv пробегов).

Потеря при обучении считается «с подсказкой»: модели показывают готовый хороший и плохой ответ
и смотрят, насколько вероятнее каждый из них стал по сравнению со стартом. Поведение же —
это то, что модель пишет сама. Здесь для каждого пробега сопоставляется:
	потеря в конце 1-й эпохи   — на парах, которых модель ещё не видела (как проверка на новых данных);
	потеря в конце 2-й эпохи   — на тех же парах второй раз;
	доля верных                — у какой доли пар хороший ответ обогнал плохой сильнее, чем на старте;
	«хороший − старт» и «плохой − старт» — насколько выросла и упала вероятность ответов
	                             (сумма по ответу, в натах; на слово — делённое на среднюю длину ответа);
	доля разрыва в поведении   — из summary.csv пробега.
Ранговая корреляция (Спирмен) показывает, совпадает ли порядок пробегов по потере с порядком по поведению.

Запуск: python 19_loss_vs_behavior.py
Результат в analysis/: loss_vs_behavior.csv, loss_vs_behavior.png
"""

import csv
import json
import math
import os

from neolib import files, metrics

# (пробег, подпись, понятие). Пробеги, которых нет на диске, пропускаются.
RUNS = [
	("card_en", "card accurate s0", "card"),
	("card_accurate_s1", "card accurate s1", "card"),
	("card_accurate_s2", "card accurate s2", "card"),
	("card_concise", "card concise", "card"),
	("card_concise_lr003", "card concise 0,003", "card"),
	("card_concise_s1", "card concise s1", "card"),
	("card_concise_s2", "card concise s2", "card"),
	("card_brief", "card brief", "card"),
	("card_the", "card the", "card"),
	("card_random", "card random", "card"),
	("b002", "short en", "short"),
	("from_ru", "short ru 591", "short"),
	("from_kk", "short kk 574", "short"),
	("from_ru_467", "short ru 467", "short"),
	("from_kk_467", "short kk 467", "short"),
	("from_ru_467b", "short ru 467b", "short"),
	("from_ru_467s", "short ru 467s", "short"),
]
LANGS = ["en", "ru", "kk"]
OUT_DIR = "analysis"
# Доля эпохи в её конце, по которой усредняется потеря (окна log.csv — по 25 пар).
LATE_SHARE = 0.25
CARD_COLORS = {
	"card_en": "#eb6834",
	"card_accurate_s1": "#c4472a",
	"card_accurate_s2": "#f2a03d",
	"card_concise": "#2a78d6",
	"card_concise_lr003": "#79aef0",
	"card_concise_s1": "#1f5fa8",
	"card_concise_s2": "#a9c8f2",
	"card_brief": "#1baf7a",
	"card_the": "#8a5cc2",
	"card_random": "#7a7974",
}


def read_log(run):
	path = files.run_path(run, "log.csv")
	if not os.path.exists(path):
		return None
	rows = []
	with open(path, encoding="utf-8") as f:
		for row in csv.DictReader(f):
			rows.append({
				"epoch": int(row["epoch"]),
				"pairs_done": int(row["pairs_done"]),
				"loss": float(row["loss"]),
				"chosen_gain": float(row["chosen_gain"]),
				"rejected_gain": float(row["rejected_gain"]),
				"accuracy": float(row["accuracy"]),
				"norm_ratio": float(row["norm_ratio"]),
			})
	return rows


def read_settings(run):
	with open(files.run_path(run, "settings.json"), encoding="utf-8") as f:
		return json.load(f)


def read_closures(run, concept):
	"""{язык: доля разрыва} для условия neologism."""
	path = files.run_path(run, "summary.csv")
	closures = {}
	if not os.path.exists(path):
		return closures
	column = "score_closure" if concept == "card" else "gap_closure"
	with open(path, encoding="utf-8") as f:
		for row in csv.DictReader(f):
			if row["condition"] == "neologism" and row.get(column):
				closures[row["lang"]] = float(row[column])
	return closures


def pair_lengths(settings, concept):
	"""Средняя длина хорошего и плохого ответа (в словах) в тех парах, на которых учили."""
	lang = settings["train_lang"]
	name = f"pairs_card_{lang}.jsonl" if concept == "card" else f"pairs_{lang}.jsonl"
	path = files.data_path(name)
	if not os.path.exists(path):
		return math.nan, math.nan
	pairs = files.read_jsonl(path)
	offset = settings.get("offset") or 0
	pairs = pairs[offset:offset + settings["n_pairs"]]
	chosen = []
	rejected = []
	for pair in pairs:
		if "chosen_words" in pair:
			chosen.append(pair["chosen_words"])
			rejected.append(pair["rejected_words"])
		else:
			chosen.append(len(pair["chosen"].split()))
			rejected.append(len(pair["rejected"].split()))
	return metrics.mean(chosen), metrics.mean(rejected)


def epoch_rows(log, epoch):
	rows = []
	for row in log:
		if row["epoch"] == epoch:
			rows.append(row)
	return rows


def late_mean(rows, key):
	"""Среднее по последней четверти эпохи."""
	if not rows:
		return math.nan
	count = max(1, int(round(len(rows) * LATE_SHARE)))
	values = []
	for row in rows[-count:]:
		values.append(row[key])
	return metrics.mean(values)


def describe_run(run, label, concept):
	"""Потеря, доля верных и рост вероятностей по эпохам вместе с долей разрыва для одного пробега."""
	log = read_log(run)
	if log is None:
		return None
	settings = read_settings(run)
	first = epoch_rows(log, 1)
	second = epoch_rows(log, 2)
	chosen_words, rejected_words = pair_lengths(settings, concept)
	closures = read_closures(run, concept)
	info = {
		"run": run,
		"label": label,
		"concept": concept,
		"train_lang": settings["train_lang"],
		"init_word": settings.get("init_word", "").strip(),
		"seed": settings.get("seed", 0) or 0,
		"lr": settings.get("lr"),
		"n_pairs": settings["n_pairs"],
		"loss_start": log[0]["loss"],
		"loss_ep1_late": late_mean(first, "loss"),
		"loss_ep2_late": late_mean(second, "loss"),
		"accuracy_ep1_late": late_mean(first, "accuracy"),
		"accuracy_ep2_late": late_mean(second, "accuracy"),
		"chosen_gain": late_mean(second, "chosen_gain"),
		"rejected_gain": late_mean(second, "rejected_gain"),
		"chosen_words": chosen_words,
		"rejected_words": rejected_words,
		"norm_ratio": log[-1]["norm_ratio"],
		"log": log,
		"epoch_pairs": len(first) and first[-1]["pairs_done"],
	}
	info["loss_drop_ep2"] = info["loss_ep1_late"] - info["loss_ep2_late"]
	info["chosen_gain_per_word"] = info["chosen_gain"] / chosen_words if chosen_words else math.nan
	info["rejected_gain_per_word"] = info["rejected_gain"] / rejected_words if rejected_words else math.nan
	margin = info["chosen_gain"] - info["rejected_gain"]
	info["rejected_share"] = -info["rejected_gain"] / margin if margin > 0 else math.nan
	for lang in LANGS:
		info[f"closure_{lang}"] = closures.get(lang, math.nan)
	info["closure_train_lang"] = closures.get(settings["train_lang"], math.nan)
	return info


def ranks(values):
	"""Ранги (1 — наименьшее), одинаковым значениям — средний ранг."""
	order = sorted(range(len(values)), key=lambda i: values[i])
	result = [0.0] * len(values)
	position = 0
	while position < len(order):
		end = position
		while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
			end += 1
		rank = (position + end) / 2 + 1
		for k in range(position, end + 1):
			result[order[k]] = rank
		position = end + 1
	return result


def spearman(xs, ys):
	"""Ранговая корреляция по парам, где оба значения — числа."""
	clean_x = []
	clean_y = []
	for x, y in zip(xs, ys):
		if not (math.isnan(x) or math.isnan(y)):
			clean_x.append(x)
			clean_y.append(y)
	if len(clean_x) < 3:
		return math.nan, len(clean_x)
	rx = ranks(clean_x)
	ry = ranks(clean_y)
	mx = metrics.mean(rx)
	my = metrics.mean(ry)
	top = 0.0
	sx = 0.0
	sy = 0.0
	for a, b in zip(rx, ry):
		top += (a - mx) * (b - my)
		sx += (a - mx) ** 2
		sy += (b - my) ** 2
	if sx == 0 or sy == 0:
		return math.nan, len(clean_x)
	return top / math.sqrt(sx * sy), len(clean_x)


def pct(value):
	if value is None or math.isnan(value):
		return "—"
	return f"{value * 100:.0f}%"


def num(value, digits=2):
	if value is None or math.isnan(value):
		return "—"
	return f"{value:.{digits}f}"


def print_table(infos):
	print(f"{'пробег':20}{'старт':>7}{'1-я эп.':>9}{'2-я эп.':>9}{'верных':>8}"
		f"{'хор.−старт':>12}{'плох.−старт':>13}{'на слово':>10}{'en':>7}{'ru':>7}{'kk':>7}")
	for info in infos:
		print(f"{info['label']:20}{num(info['loss_start']):>7}{num(info['loss_ep1_late']):>9}{num(info['loss_ep2_late']):>9}"
			f"{pct(info['accuracy_ep2_late']):>8}{num(info['chosen_gain'], 0):>12}{num(info['rejected_gain'], 0):>13}"
			f"{num(info['chosen_gain_per_word'], 1):>10}"
			f"{pct(info['closure_en']):>7}{pct(info['closure_ru']):>7}{pct(info['closure_kk']):>7}")
	print("\nСтарт — потеря на первых 25 парах (должна быть около 1,386). 1-я и 2-я эп. — средняя потеря за последнюю")
	print("четверть эпохи: в 1-й эпохе пары ещё не виденные, во 2-й — те же второй раз. «Хор.−старт» и «плох.−старт» —")
	print("насколько выросла и упала вероятность хорошего и плохого ответа (сумма по ответу, наты, конец 2-й эпохи);")
	print("«на слово» — рост вероятности хорошего ответа на одно слово (e^x — во сколько раз).")


def print_correlations(infos):
	print("\nСовпадает ли порядок пробегов по потере с порядком по поведению (Спирмен; −1 — чем ниже потеря, тем лучше):")
	for concept, title, closure_key in [
		("card", "~card, доля разрыва в en", "closure_en"),
		("card", "~card, доля разрыва в ru", "closure_ru"),
		("card", "~card, доля разрыва в kk", "closure_kk"),
	]:
		for loss_key, loss_name in [("loss_ep1_late", "1-я эпоха"), ("loss_ep2_late", "2-я эпоха")]:
			xs = []
			ys = []
			for info in infos:
				if info["concept"] == concept:
					xs.append(info[loss_key])
					ys.append(info[closure_key])
			rho, n = spearman(xs, ys)
			print(f"  {title:40} потеря {loss_name}: ρ = {num(rho)} (пробегов: {n})")
	losses = []
	closures = []
	for info in infos:
		if info["concept"] == "short" and not math.isnan(info["closure_train_lang"]):
			losses.append(info["loss_ep2_late"])
			closures.append(info["closure_train_lang"])
	if losses:
		print(f"  ~short, язык обучения: доля разрыва {pct(min(closures))}–{pct(max(closures))} при потере"
			f" {num(min(losses))}–{num(max(losses))} — поведение у всех на потолке, корреляция не считается.")


def write_csv(infos, path):
	columns = [
		"run", "label", "concept", "train_lang", "init_word", "seed", "lr", "n_pairs",
		"loss_start", "loss_ep1_late", "loss_ep2_late", "loss_drop_ep2",
		"accuracy_ep1_late", "accuracy_ep2_late",
		"chosen_gain", "rejected_gain", "rejected_share",
		"chosen_words", "rejected_words", "chosen_gain_per_word", "rejected_gain_per_word",
		"norm_ratio", "closure_en", "closure_ru", "closure_kk", "closure_train_lang",
	]
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(columns)
		for info in infos:
			row = []
			for column in columns:
				value = info[column]
				if isinstance(value, float):
					value = "" if math.isnan(value) else round(value, 4)
				row.append(value)
			writer.writerow(row)


def style(ax):
	ax.set_facecolor("#fcfcfb")
	for side in ["top", "right"]:
		ax.spines[side].set_visible(False)


def plot(infos, path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	cards = []
	for info in infos:
		if info["concept"] == "card":
			cards.append(info)
	fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="#fcfcfb")
	loss_ax = axes[0][0]
	acc_ax = axes[0][1]
	gain_ax = axes[1][0]
	scatter_ax = axes[1][1]
	for ax in [loss_ax, acc_ax, gain_ax, scatter_ax]:
		style(ax)

	for info in cards:
		color = CARD_COLORS.get(info["run"], "#555555")
		xs = []
		losses = []
		accuracies = []
		chosen = []
		rejected = []
		for row in info["log"]:
			xs.append(row["pairs_done"])
			losses.append(row["loss"])
			accuracies.append(row["accuracy"] * 100)
			chosen.append(row["chosen_gain"])
			rejected.append(row["rejected_gain"])
		name = f"{info['label']} — en {pct(info['closure_en'])}"
		loss_ax.plot(xs, losses, color=color, linewidth=1.6, label=name)
		acc_ax.plot(xs, accuracies, color=color, linewidth=1.4)
		gain_ax.plot(xs, chosen, color=color, linewidth=1.6)
		gain_ax.plot(xs, rejected, color=color, linewidth=1.2, linestyle="--")
	if cards and cards[0]["epoch_pairs"]:
		for ax in [loss_ax, acc_ax, gain_ax]:
			ax.axvline(cards[0]["epoch_pairs"], color="#a3a29c", linewidth=1, linestyle=":")
	loss_ax.axhline(2 * math.log(2), color="#a3a29c", linewidth=1, linestyle=":")
	loss_ax.set_title("~card: потеря (до пунктира — новые пары, после — те же второй раз)", loc="left", fontsize=10)
	loss_ax.set_xlabel("пар пройдено")
	loss_ax.set_ylabel("потеря")
	loss_ax.legend(frameon=False, fontsize=8)
	acc_ax.set_title("~card: у какой доли пар хороший ответ обогнал плохой сильнее, чем на старте", loc="left", fontsize=10)
	acc_ax.set_xlabel("пар пройдено")
	acc_ax.set_ylabel("%")
	acc_ax.set_ylim(40, 103)
	gain_ax.axhline(0, color="#a3a29c", linewidth=1)
	gain_ax.set_title("~card: насколько вероятнее стал ответ, чем на старте", loc="left", fontsize=10)
	gain_ax.text(0.02, 0.04, "сплошная — хороший ответ, пунктир — плохой", transform=gain_ax.transAxes,
		fontsize=8, color="#3d3c39")
	gain_ax.set_xlabel("пар пройдено")
	gain_ax.set_ylabel("наты (сумма по ответу)")

	for info in infos:
		if info["concept"] == "card":
			color = CARD_COLORS.get(info["run"], "#555555")
			marker = "o"
		else:
			color = "#7a7974"
			marker = "^"
		x = info["loss_ep2_late"]
		y = info["closure_train_lang"] * 100
		if math.isnan(x) or math.isnan(y):
			continue
		scatter_ax.scatter([x], [y], color=color, marker=marker, s=55, zorder=3)
		if info["concept"] == "card":
			scatter_ax.annotate(info["label"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5,
				color="#3d3c39")
	scatter_ax.text(0.98, 0.83, "▲ ~short: все пробеги около 100%", transform=scatter_ax.transAxes, ha="right",
		fontsize=8, color="#3d3c39")
	scatter_ax.axhline(100, color="#a3a29c", linewidth=1, linestyle=":")
	scatter_ax.set_title("потеря в конце обучения и поведение", loc="left", fontsize=10)
	scatter_ax.set_xlabel("потеря, последняя четверть 2-й эпохи")
	scatter_ax.set_ylabel("доля разрыва в языке обучения, %")
	fig.tight_layout()
	fig.savefig(path, dpi=150, facecolor="#fcfcfb")
	plt.close(fig)


def main():
	files.setup_console()
	os.makedirs(OUT_DIR, exist_ok=True)
	infos = []
	for run, label, concept in RUNS:
		info = describe_run(run, label, concept)
		if info is None:
			print(f"Нет runs/{run}/log.csv — пропускаю")
			continue
		infos.append(info)
	print_table(infos)
	print_correlations(infos)
	write_csv(infos, os.path.join(OUT_DIR, "loss_vs_behavior.csv"))
	try:
		plot(infos, os.path.join(OUT_DIR, "loss_vs_behavior.png"))
	except ImportError:
		print("matplotlib не установлен — график пропущен")
	print(f"\nСохранено в {OUT_DIR}/: loss_vs_behavior.csv, loss_vs_behavior.png")


if __name__ == "__main__":
	main()
