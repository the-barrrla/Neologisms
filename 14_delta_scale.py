"""Проверка: дозировка выученного сдвига. Какие условия формата растут вместе со сдвигом, в каком языке.

Обученный вектор v = e + Δ (e — стартовое слово, Δ — выученный сдвиг). Здесь берутся векторы e + α·Δ:
	α = 0 — необученный старт (контроль), α = 1 — обученный ~card, α > 1 — усиленный сдвиг.
Если какое-то условие в языке растёт с α, его несёт выученный сдвиг. Если стоит на месте при любом α,
этот язык сдвиг «не слышит». Ответы для α = 0 и α = 1 берутся из generations.jsonl пробега (их не генерируем).

Запуск: python 14_delta_scale.py --run card_concise
	--alphas 0 0.25 0.5 1 1.5 2   (по умолчанию)
	--limit 50                    (вопросов на язык; для α = 0 и 1 используются те же вопросы)
Около 20–30 минут на пробег. Прерванный запуск продолжится с места остановки.
Результат: runs/{run}/delta_scale.jsonl (ответы), analysis/delta_scale_{run}.csv и .png
"""

import argparse
import os

import config
from neolib import common, files, probe

PLOT_MEASURES = ["count", "rest", "plain", "score"]
PLOT_COLORS = {"count": "#eb6834", "rest": "#1baf7a", "plain": "#2a78d6", "score": "#0b0b0b"}


def variant_name(alpha):
	return f"alpha={alpha:g}"


def plot(rows, alphas, run, path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	fig, axes = plt.subplots(1, len(probe.LANGS), figsize=(13, 4), facecolor="#fcfcfb", sharey=True)
	for ax, lang in zip(axes, probe.LANGS):
		ax.set_facecolor("#fcfcfb")
		for side in ["top", "right"]:
			ax.spines[side].set_visible(False)
		ax.axvline(1, color="#a3a29c", linewidth=1, linestyle=":")
		for measure in PLOT_MEASURES:
			xs = []
			ys = []
			for alpha in alphas:
				for row in rows:
					if row["lang"] == lang and row["alpha"] == alpha:
						xs.append(alpha)
						ys.append(row[measure] * 100)
			width = 2.4 if measure == "score" else 1.6
			ax.plot(xs, ys, marker="o", color=PLOT_COLORS[measure], linewidth=width, label=probe.MEASURE_NAMES[measure])
		ax.set_title(lang, loc="left", fontsize=11)
		ax.set_xlabel("α (доля выученного сдвига; 1 = обученный ~card)")
		ax.set_ylim(-3, 103)
	axes[0].set_ylabel("доля ответов, %  (балл — средний, ×100)")
	axes[0].legend(frameon=False, fontsize=8, loc="upper left")
	fig.suptitle(f"{run}: что меняется, если усиливать выученный сдвиг", x=0.01, ha="left", fontsize=12)
	fig.tight_layout(rect=(0, 0, 1, 0.94))
	fig.savefig(path, dpi=150, facecolor="#fcfcfb")
	plt.close(fig)


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--run", required=True, help="пробег ~card, например card_concise или card_en")
	parser.add_argument("--alphas", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 1.5, 2])
	parser.add_argument("--limit", type=int, default=50)
	parser.add_argument("--max-new-tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
	args = parser.parse_args()
	common.set_seed(config.SEED)
	probe.use_card_token()

	vector, init, meta = probe.load_vectors(args.run)
	if meta.get("neologism") != config.CARD_NEOLOGISM:
		raise SystemExit(f"{args.run} — не пробег ~card")
	delta = vector - init
	print(f"Пробег {args.run}: старт {meta.get('init_word')!r}, |Δ|/|e| = {delta.norm() / init.norm():.3f}")

	saved = {0.0: probe.saved_answers(args.run, "control"), 1.0: probe.saved_answers(args.run, "neologism")}
	cache_path = common.run_path(args.run, "delta_scale.jsonl")
	cache = probe.load_cache(cache_path)
	model = None
	tokenizer = None
	neo_id = None

	rows = []
	for lang in probe.LANGS:
		questions = probe.test_questions(lang, args.limit)
		question_ids = []
		for row in questions:
			question_ids.append(row["id"])
		levels = probe.reference_levels(args.run, lang, question_ids)
		suffix = common.neologism_suffix(lang)
		for alpha in args.alphas:
			answers = None
			if alpha in saved:
				answers = []
				for question_id in question_ids:
					if (lang, question_id) in saved[alpha]:
						answers.append(saved[alpha][(lang, question_id)])
				if len(answers) != len(question_ids):
					answers = None
			if answers is None:
				if model is None:
					model, tokenizer, neo_id = common.load_model_and_tokenizer()
				scaled = init + alpha * delta
				answers = probe.generate_variant(model, tokenizer, neo_id, scaled, variant_name(alpha), lang,
					questions, suffix, cache, cache_path, args.max_new_tokens)
			row = probe.summarize(variant_name(alpha), lang, answers, levels)
			row["alpha"] = alpha
			rows.append(row)

	probe.print_table(rows, f"{args.run}: e + α·Δ")
	os.makedirs("analysis", exist_ok=True)
	csv_path = os.path.join("analysis", f"delta_scale_{args.run}.csv")
	png_path = os.path.join("analysis", f"delta_scale_{args.run}.png")
	probe.write_rows(rows, csv_path)
	try:
		plot(rows, args.alphas, args.run, png_path)
	except ImportError:
		print("matplotlib не установлен — график пропущен")
	print(f"\nСохранено: {cache_path}, {csv_path}, {png_path}")


if __name__ == "__main__":
	main()
