"""Геометрия обученных векторов (видеокарта не нужна): насколько каждый вектор ушёл от стартового слова
и похожи ли выученные сдвиги у разных пробегов.

Обученный вектор v = e + Δ, где e — эмбеддинг стартового слова, Δ — выученный сдвиг.
Для каждого пробега считается:
	|Δ| / |e|           — насколько далеко вектор ушёл от старта (в долях длины стартового вектора);
	cos(v, e)           — насколько вектор ещё похож на стартовое слово;
	доля Δ вдоль e      — какая часть сдвига просто растягивает или сжимает стартовый вектор,
	                      а какая уводит его в новом направлении;
	эпоха 1 → 2         — как вырос сдвиг за вторую эпоху и сохранил ли он направление.
Для всех пар пробегов — косинус между их сдвигами Δ: выучивают ли разные пробеги одно и то же направление.

Запуск: python 12_vector_geometry.py
Результат в analysis/: vector_geometry.csv, delta_cosine.csv, delta_cosine.png
"""

import csv
import os

import torch
import torch.nn.functional as F

import config

# Пробеги в порядке запуска и их короткие подписи для графика. Пробеги, которых нет на диске, пропускаются.
RUNS = [
	("short_en", "short en (β=0,2)"),
	("b002", "short en"),
	("from_ru", "short ru 591"),
	("from_kk", "short kk 574"),
	("from_ru_467", "short ru 467"),
	("from_kk_467", "short kk 467"),
	("from_ru_467b", "short ru 467b"),
	("from_ru_467s", "short ru 467s"),
	("card_en", "card accurate"),
	("card_accurate_s1", "card accurate s1"),
	("card_accurate_s2", "card accurate s2"),
	("card_concise", "card concise"),
	("card_concise_s1", "card concise s1"),
	("card_concise_s2", "card concise s2"),
	("card_concise_lr003", "card concise 0,003"),
	("card_brief", "card brief"),
	("card_detailed", "card detailed"),
	("card_table", "card table"),
	("card_the", "card the"),
	("card_random", "card random"),
]
OUT_DIR = "analysis"


def load(run, name):
	"""Сохранённый вектор пробега или None, если файла нет."""
	path = os.path.join(config.RUNS_DIR, run, name)
	if not os.path.exists(path):
		return None
	return torch.load(path, map_location="cpu")


def cosine(a, b):
	return F.cosine_similarity(a.float(), b.float(), dim=0).item()


def describe_run(run):
	"""Длины и косинусы стартового вектора, обученного вектора и добавки Δ."""
	final = load(run, "neologism.pt")
	if final is None:
		return None
	vector = final["vector"].float()
	init = final["init_vector"].float()
	delta = vector - init
	init_unit = init / init.norm()
	along = torch.dot(delta, init_unit).item()
	info = {
		"run": run,
		"init_word": final["meta"].get("init_word", ""),
		"neologism": final["meta"].get("neologism", ""),
		"train_lang": final["meta"].get("train_lang", ""),
		"n_pairs": final["meta"].get("n_pairs", ""),
		"lr": final["meta"].get("lr", ""),
		"seed": final["meta"].get("seed", 0),
		"init_norm": init.norm().item(),
		"vector_norm": vector.norm().item(),
		"delta_ratio": (delta.norm() / init.norm()).item(),
		"cos_vector_init": cosine(vector, init),
		# Доля квадрата длины сдвига, приходящаяся на направление стартового вектора.
		"delta_along_init_share": (along * along) / (delta.norm().item() ** 2),
		"delta": delta,
		"init": init,
		"vector": vector,
	}
	epoch1 = load(run, "neologism_epoch1.pt")
	if epoch1 is not None:
		delta1 = epoch1["vector"].float() - init
		info["delta_ratio_epoch1"] = (delta1.norm() / init.norm()).item()
		info["cos_delta_epoch1_final"] = cosine(delta1, delta)
	else:
		info["delta_ratio_epoch1"] = float("nan")
		info["cos_delta_epoch1_final"] = float("nan")
	return info


def write_geometry(infos, path):
	columns = [
		"run", "neologism", "init_word", "train_lang", "n_pairs", "lr", "seed",
		"init_norm", "vector_norm", "delta_ratio", "delta_ratio_epoch1", "cos_delta_epoch1_final",
		"cos_vector_init", "delta_along_init_share",
	]
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(columns)
		for info in infos:
			row = []
			for column in columns:
				value = info[column]
				if isinstance(value, float):
					value = round(value, 4)
				row.append(value)
			writer.writerow(row)


def delta_cosines(infos):
	"""Матрица косинусов между добавками Δ всех пробегов."""
	matrix = []
	for a in infos:
		row = []
		for b in infos:
			row.append(cosine(a["delta"], b["delta"]))
		matrix.append(row)
	return matrix


def write_cosines(infos, matrix, path):
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		header = ["run"]
		for info in infos:
			header.append(info["run"])
		writer.writerow(header)
		for info, row in zip(infos, matrix):
			line = [info["run"]]
			for value in row:
				line.append(round(value, 3))
			writer.writerow(line)


def plot_cosines(infos, matrix, labels, path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(figsize=(8.6, 7.2), facecolor="#fcfcfb")
	image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
	for i in range(len(infos)):
		for j in range(len(infos)):
			value = matrix[i][j]
			color = "white" if abs(value) > 0.6 else "#0b0b0b"
			ax.text(j, i, f"{value:.2f}".replace("-", "−"), ha="center", va="center", fontsize=7.5, color=color)
	ax.set_xticks(range(len(infos)))
	ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
	ax.set_yticks(range(len(infos)))
	ax.set_yticklabels(labels, fontsize=8)
	ax.tick_params(length=0)
	for side in ["top", "right", "left", "bottom"]:
		ax.spines[side].set_visible(False)
	bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
	bar.set_label("косинус между выученными сдвигами Δ", color="#52514e")
	bar.outline.set_visible(False)
	ax.set_title("Выучивают ли разные пробеги одно и то же направление", loc="left", fontsize=11, color="#0b0b0b")
	fig.tight_layout()
	fig.savefig(path, dpi=160, facecolor="#fcfcfb")
	plt.close(fig)


def print_summary(infos, matrix):
	print(f"{'пробег':22}{'старт':12}{'|Δ|/|e|':>9}{'эп.1':>7}{'cos(v,e)':>10}{'Δ вдоль e':>11}")
	for info in infos:
		print(f"{info['run']:22}{info['init_word']!r:12}{info['delta_ratio']:9.3f}{info['delta_ratio_epoch1']:7.3f}"
			f"{info['cos_vector_init']:10.3f}{info['delta_along_init_share']:11.1%}")
	by_run = {}
	for index, info in enumerate(infos):
		by_run[info["run"]] = index
	starts = {}
	for info in infos:
		starts[info["init_word"]] = info["init"]
	if " accurate" in starts and " concise" in starts:
		print(f"\ncos(« accurate», « concise») = {cosine(starts[' accurate'], starts[' concise']):.3f}")
	pairs = [
		("card_en", "card_concise", "card: accurate против concise (шаг 0,01)"),
		("card_concise", "card_concise_lr003", "card concise: шаг 0,01 против 0,003"),
		("b002", "card_en", "short en против card accurate (тот же старт)"),
		("b002", "card_concise", "short en против card concise"),
		("b002", "from_ru", "short: en против ru"),
		("b002", "from_kk", "short: en против kk"),
		("from_ru", "from_kk", "short: ru против kk"),
		("from_ru_467", "from_ru_467s", "short ru 467: зерно 0 против зерна 1"),
		("from_ru_467", "from_ru_467b", "short ru: первые 467 против последних 467"),
		("card_en", "card_accurate_s1", "card accurate: зерно 0 против 1"),
		("card_en", "card_accurate_s2", "card accurate: зерно 0 против 2"),
		("card_accurate_s1", "card_accurate_s2", "card accurate: зерно 1 против 2"),
		("card_concise", "card_concise_s1", "card concise: зерно 0 против 1"),
		("card_concise", "card_concise_s2", "card concise: зерно 0 против 2"),
		("card_concise_s1", "card_concise_s2", "card concise: зерно 1 против 2"),
		("card_concise", "card_brief", "card: concise против brief"),
		("card_en", "card_detailed", "card: accurate против detailed"),
		("card_the", "card_random", "card: the против случайного"),
	]
	dimension = infos[0]["delta"].shape[0]
	chance = 2.0 / (dimension ** 0.5)
	print(f"\nКосинус между сдвигами Δ (у двух случайных направлений в {dimension} измерениях"
		f" он в 95% случаев в пределах ±{chance:.3f}):")
	for a, b, name in pairs:
		if a in by_run and b in by_run:
			print(f"  {name:48}{matrix[by_run[a]][by_run[b]]:7.3f}")
	if "card_en" in by_run and "card_concise" in by_run:
		accurate = infos[by_run["card_en"]]
		concise = infos[by_run["card_concise"]]
		print("\nКуда ушли векторы ~card относительно чужого стартового слова:")
		print(f"  cos(v accurate-старта, « concise») = {cosine(accurate['vector'], concise['init']):.3f}"
			f" (у самого « accurate»: {cosine(accurate['init'], concise['init']):.3f})")
		print(f"  cos(v concise-старта, « accurate») = {cosine(concise['vector'], accurate['init']):.3f}"
			f" (у самого « concise»: {cosine(concise['init'], accurate['init']):.3f})")


def main():
	os.makedirs(OUT_DIR, exist_ok=True)
	infos = []
	labels = []
	for run, label in RUNS:
		info = describe_run(run)
		if info is None:
			print(f"Нет runs/{run}/neologism.pt — пропускаю")
			continue
		infos.append(info)
		labels.append(label)
	matrix = delta_cosines(infos)
	write_geometry(infos, os.path.join(OUT_DIR, "vector_geometry.csv"))
	write_cosines(infos, matrix, os.path.join(OUT_DIR, "delta_cosine.csv"))
	print_summary(infos, matrix)
	try:
		plot_cosines(infos, matrix, labels, os.path.join(OUT_DIR, "delta_cosine.png"))
	except ImportError:
		print("matplotlib не установлен — график пропущен")
	print(f"\nСохранено в {OUT_DIR}/: vector_geometry.csv, delta_cosine.csv, delta_cosine.png")


if __name__ == "__main__":
	main()
