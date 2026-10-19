"""Вопрос 2 напрямую: ответ-карточка стал вероятнее, но почему модель его не пишет.

Генерации нет — только подсчёт вероятностей готовых ответов (как потеря при обучении).
На тестовых вопросах для каждого пробега ~card и языка берутся три готовых ответа из generations.jsonl:
	карточка   — ответ модели на явную инструкцию (условие target): «правильный» ответ;
	свой       — что модель сама пишет с обученным вектором (условие neologism);
	без приписки — обычный длинный ответ (условие base): «неправильный» ответ.
Каждый ответ «подсказывается» модели после промпта «… Give me a ~card answer.» дважды:
со стартовым вектором (e) и с обученным (v). Считается:
	рост вероятности карточки (v против e) — то же, что потеря видит при обучении, но на новых вопросах;
	доля верных — у какой доли вопросов карточка обогнала ответ без приписки сильнее, чем со стартом
	              (так же считается «доля верных» в log.csv);
	совпадение с выбором модели — у какой доли токенов карточки модель сама поставила бы этот же токен первым;
	первый расход — номер токена, на котором модель впервые выбрала бы не тот токен, что в карточке
	                (0 — уже первый токен ответа другой);
	начало ответа — вероятность первых 20 токенов карточки и своего ответа: какое начало модель предпочитает.
Если карточка стала намного вероятнее, «доля верных» высокая, но первый расход — на первых токенах,
то потеря падает, а поведение не меняется: модель с самого начала уходит на свою дорожку.

Запуск: python 20_likelihood_vs_generation.py
	--runs card_en card_accurate_s1 card_accurate_s2 card_concise card_concise_lr003   (по умолчанию)
	--limit 50   (вопросов на язык)
Около 5 минут на пробег. Прерванный запуск продолжится с места остановки.
Результат: analysis/likelihood/{run}.jsonl (по ответам), analysis/likelihood_vs_generation.csv и .png
"""

import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F

import config
from neolib import common, files, metrics, probe

DEFAULT_RUNS = ["card_en", "card_accurate_s1", "card_accurate_s2", "card_concise", "card_concise_lr003"]
CANDIDATES = {"target": "карточка", "neologism": "свой", "base": "без приписки"}
START_TOKENS = 20
OUT_DIR = "analysis"
CACHE_DIR = os.path.join("analysis", "likelihood")
RUN_COLORS = {
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


def token_stats(model, prompt_ids, response_ids, neo_id, vector):
	"""Вероятность каждого токена ответа и совпадение с самым вероятным токеном (ответ «подсказан»)."""
	with torch.no_grad():
		ids = torch.cat([prompt_ids, response_ids])
		embeds = common.embed_ids(model, ids, neo_id, vector).unsqueeze(0)
		n_response = response_ids.shape[0]
		logits = common.last_logits(model, embeds, n_response + 1)[0, :-1, :].float()
		targets = response_ids.to(logits.device)
		token_logprobs = -F.cross_entropy(logits, targets, reduction="none")
		matches = logits.argmax(dim=-1) == targets
		del logits
	return token_logprobs.cpu(), matches.cpu()


def summarize_tokens(token_logprobs, matches):
	"""Сумма лог-вероятностей, доля токенов, совпавших с выбором модели, и первый токен, где модель выбрала бы другой."""
	n = int(token_logprobs.shape[0])
	first_miss = n
	for position in range(n):
		if not bool(matches[position]):
			first_miss = position
			break
	return {
		"n_tokens": n,
		"sum_logprob": float(token_logprobs.sum()),
		"start_logprob": float(token_logprobs[:START_TOKENS].sum()),
		"top1_share": float(matches.float().mean()) if n else math.nan,
		"first_miss": first_miss,
	}


def load_answers(run):
	"""{(язык, id): {условие: ответ}} для нужных условий."""
	answers = {}
	for row in files.read_jsonl(files.run_path(run, "generations.jsonl")):
		if row["condition"] in CANDIDATES:
			answers.setdefault((row["lang"], row["id"]), {})[row["condition"]] = row["response"]
	return answers


def load_done(path):
	"""Уже посчитанные строки кэша, чтобы продолжить прерванный запуск."""
	done = {}
	if os.path.exists(path):
		for row in files.read_jsonl(path):
			done[(row["lang"], row["id"], row["candidate"], row["vector"])] = row
	return done


def score_run(model, tokenizer, neo_id, run, limit):
	"""Лог-вероятности трёх готовых ответов с обученным и стартовым вектором; результат пишется в кэш."""
	vector, init, meta = probe.load_vectors(run)
	answers = load_answers(run)
	os.makedirs(CACHE_DIR, exist_ok=True)
	cache_path = os.path.join(CACHE_DIR, f"{run}.jsonl")
	done = load_done(cache_path)
	vectors = {"trained": vector, "start": init}
	for lang in probe.LANGS:
		questions = probe.test_questions(lang, limit)
		suffix = common.neologism_suffix(lang)
		new_rows = []
		for question in questions:
			by_condition = answers.get((lang, question["id"]), {})
			user_text = common.build_user_text(question["question"], suffix)
			prompt_ids = common.encode(tokenizer, common.chat_prompt(tokenizer, user_text))
			for candidate in CANDIDATES:
				if candidate not in by_condition:
					continue
				response_text = common.chat_response_part(tokenizer, user_text, by_condition[candidate])
				response_ids = common.encode(tokenizer, response_text)[:config.MAX_RESPONSE_TOKENS]
				for vector_name, vector_value in vectors.items():
					key = (lang, question["id"], candidate, vector_name)
					if key in done:
						continue
					token_logprobs, matches = token_stats(model, prompt_ids, response_ids, neo_id, vector_value)
					row = {"lang": lang, "id": question["id"], "candidate": candidate, "vector": vector_name}
					row.update(summarize_tokens(token_logprobs, matches))
					done[key] = row
					new_rows.append(row)
			if len(new_rows) >= 60:
				files.append_jsonl(cache_path, new_rows)
				new_rows = []
		if new_rows:
			files.append_jsonl(cache_path, new_rows)
		print(f"  {run}/{lang}: готово")
	return done


def median(values):
	if not values:
		return math.nan
	return metrics.median(values)


def aggregate(run, lang, done):
	"""Сводка по языку: только вопросы, где посчитаны все три ответа с обоими векторами."""
	by_question = {}
	for (row_lang, question_id, candidate, vector_name), row in done.items():
		if row_lang == lang:
			by_question.setdefault(question_id, {})[(candidate, vector_name)] = row
	gains = []
	gains_per_token = []
	base_gains = []
	correct = []
	top1_trained = []
	top1_start = []
	first_miss = []
	first_token_miss = []
	own_start_better = []
	for question_id, rows in by_question.items():
		needed = [("target", "trained"), ("target", "start"), ("base", "trained"), ("base", "start"), ("neologism", "trained")]
		complete = True
		for key in needed:
			if key not in rows:
				complete = False
		if not complete:
			continue
		target_v = rows[("target", "trained")]
		target_e = rows[("target", "start")]
		base_v = rows[("base", "trained")]
		base_e = rows[("base", "start")]
		own_v = rows[("neologism", "trained")]
		gain = target_v["sum_logprob"] - target_e["sum_logprob"]
		gains.append(gain)
		gains_per_token.append(gain / max(1, target_v["n_tokens"]))
		base_gains.append(base_v["sum_logprob"] - base_e["sum_logprob"])
		margin_v = target_v["sum_logprob"] - base_v["sum_logprob"]
		margin_e = target_e["sum_logprob"] - base_e["sum_logprob"]
		correct.append(1.0 if margin_v > margin_e else 0.0)
		top1_trained.append(target_v["top1_share"])
		top1_start.append(target_e["top1_share"])
		first_miss.append(target_v["first_miss"])
		first_token_miss.append(1.0 if target_v["first_miss"] == 0 else 0.0)
		own_start_better.append(1.0 if own_v["start_logprob"] > target_v["start_logprob"] else 0.0)
	return {
		"run": run,
		"lang": lang,
		"n": len(gains),
		"target_gain": metrics.mean(gains) if gains else math.nan,
		"target_gain_per_token": metrics.mean(gains_per_token) if gains else math.nan,
		"base_gain": metrics.mean(base_gains) if gains else math.nan,
		"correct_share": metrics.mean(correct) if gains else math.nan,
		"top1_trained": metrics.mean(top1_trained) if gains else math.nan,
		"top1_start": metrics.mean(top1_start) if gains else math.nan,
		"first_miss_median": median(first_miss),
		"first_token_miss_share": metrics.mean(first_token_miss) if gains else math.nan,
		"own_start_better_share": metrics.mean(own_start_better) if gains else math.nan,
	}


def read_closure(run, lang):
	"""Доля разрыва ~card по баллу для языка из summary.csv."""
	path = files.run_path(run, "summary.csv")
	if not os.path.exists(path):
		return math.nan
	with open(path, encoding="utf-8") as f:
		for row in csv.DictReader(f):
			if row["lang"] == lang and row["condition"] == "neologism" and row.get("score_closure"):
				return float(row["score_closure"])
	return math.nan


def pct(value):
	if value is None or math.isnan(value):
		return "—"
	return f"{value * 100:.0f}%"


def num(value, digits=1):
	if value is None or math.isnan(value):
		return "—"
	return f"{value:+.{digits}f}"


def print_table(rows):
	print(f"\n{'пробег':22}{'язык':>5}{'n':>4}{'карт.: рост':>13}{'на токен':>10}{'без приписки':>14}{'верных':>8}"
		f"{'совп. v':>9}{'совп. e':>9}{'расход':>8}{'с 1-го':>8}{'своё нач.':>11}{'поведение':>11}")
	for row in rows:
		print(f"{row['run']:22}{row['lang']:>5}{row['n']:>4}{num(row['target_gain'], 0):>13}{num(row['target_gain_per_token'], 2):>10}"
			f"{num(row['base_gain'], 0):>14}{pct(row['correct_share']):>8}{pct(row['top1_trained']):>9}{pct(row['top1_start']):>9}"
			f"{row['first_miss_median']:>8.0f}{pct(row['first_token_miss_share']):>8}{pct(row['own_start_better_share']):>11}"
			f"{pct(row['closure']):>11}")
	print("\nкарт.: рост — на сколько натов карточка стала вероятнее с обученным вектором, чем со стартовым (на токен — то же,")
	print("делённое на длину); без приписки — то же для обычного длинного ответа; верных — у какой доли вопросов карточка")
	print("обогнала длинный ответ сильнее, чем со стартом; совп. v / e — доля токенов карточки, которые модель сама поставила бы")
	print("первыми (с обученным / стартовым вектором); расход — медианный номер токена, где модель впервые уходит с карточки;")
	print("с 1-го — доля вопросов, где расход уже на первом токене; своё нач. — доля вопросов, где первые 20 токенов своего")
	print("ответа вероятнее первых 20 токенов карточки; поведение — доля разрыва по баллу из summary.csv.")


def write_csv(rows, path):
	columns = ["run", "lang", "n", "target_gain", "target_gain_per_token", "base_gain", "correct_share",
		"top1_trained", "top1_start", "first_miss_median", "first_token_miss_share", "own_start_better_share", "closure"]
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(columns)
		for row in rows:
			values = []
			for column in columns:
				value = row[column]
				if isinstance(value, float):
					value = "" if math.isnan(value) else round(value, 4)
				values.append(value)
			writer.writerow(values)


def plot(rows, runs, path):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	panels = [
		("target_gain_per_token", "насколько вероятнее стал каждый токен карточки, наты", 1.0),
		("correct_share", "доля вопросов, где карточка обогнала длинный ответ сильнее, чем со стартом, %", 100.0),
		("top1_trained", "доля токенов карточки, которые модель выбрала бы сама, %", 100.0),
		("closure", "поведение: доля разрыва по баллу, %", 100.0),
	]
	fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.0 * len(panels)), facecolor="#fcfcfb")
	width = 0.8 / max(1, len(runs))
	for ax, (key, title, scale) in zip(axes, panels):
		ax.set_facecolor("#fcfcfb")
		for side in ["top", "right"]:
			ax.spines[side].set_visible(False)
		for index, run in enumerate(runs):
			xs = []
			ys = []
			for lang_index, lang in enumerate(probe.LANGS):
				for row in rows:
					if row["run"] == run and row["lang"] == lang and not math.isnan(row[key]):
						xs.append(lang_index + (index - (len(runs) - 1) / 2) * width)
						ys.append(row[key] * scale)
			ax.bar(xs, ys, width=width * 0.92, color=RUN_COLORS.get(run, "#7a7974"), label=run)
		ax.axhline(0, color="#a3a29c", linewidth=1)
		ax.set_xticks(range(len(probe.LANGS)))
		ax.set_xticklabels(probe.LANGS)
		ax.set_title(title, loc="left", fontsize=10)
	axes[0].legend(frameon=False, fontsize=8, ncol=3)
	fig.tight_layout()
	fig.savefig(path, dpi=150, facecolor="#fcfcfb")
	plt.close(fig)


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
	parser.add_argument("--limit", type=int, default=50)
	args = parser.parse_args()
	common.set_seed(config.SEED)
	probe.use_card_token()

	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	for run in args.runs:
		if not os.path.exists(files.run_path(run, "generations.jsonl")):
			print(f"Нет runs/{run}/generations.jsonl — пропускаю")
			continue
		print(f"\n{run}")
		score_run(model, tokenizer, neo_id, run, args.limit)

	# В сводку идут все посчитанные пробеги, в том числе из прошлых запусков.
	all_runs = []
	for run in DEFAULT_RUNS + args.runs:
		if run not in all_runs:
			all_runs.append(run)
	if os.path.exists(CACHE_DIR):
		for name in sorted(os.listdir(CACHE_DIR)):
			if name.endswith(".jsonl") and name[:-6] not in all_runs:
				all_runs.append(name[:-6])
	rows = []
	for run in all_runs:
		cache_path = os.path.join(CACHE_DIR, f"{run}.jsonl")
		if not os.path.exists(cache_path):
			continue
		done = load_done(cache_path)
		for lang in probe.LANGS:
			row = aggregate(run, lang, done)
			row["closure"] = read_closure(run, lang)
			rows.append(row)

	os.makedirs(OUT_DIR, exist_ok=True)
	print_table(rows)
	csv_path = os.path.join(OUT_DIR, "likelihood_vs_generation.csv")
	png_path = os.path.join(OUT_DIR, "likelihood_vs_generation.png")
	write_csv(rows, csv_path)
	runs_shown = []
	for row in rows:
		if row["run"] not in runs_shown:
			runs_shown.append(row["run"])
	try:
		plot(rows, runs_shown, png_path)
	except ImportError:
		print("matplotlib не установлен — график пропущен")
	print(f"\nСохранено: {csv_path}, {png_path}, ответы по пробегам — в {CACHE_DIR}/")


if __name__ == "__main__":
	main()
