"""Проверка: к каким словам словаря ближе всего обученные векторы и выученные сдвиги.

Для каждого пробега печатаются ближайшие по косинусу токены словаря Gemma для:
	e — стартового слова (самопроверка: первым должно стоять само стартовое слово);
	v — обученного вектора (во что превратилось слово);
	Δ — выученного сдвига (что именно добавило обучение).
Генерации нет, нужна только таблица эмбеддингов модели — около 2 минут.

Запуск: python 16_nearest_tokens.py
	--runs card_en card_concise card_concise_lr003 b002   (по умолчанию)
	--top 15
Результат: analysis/nearest_tokens.md
"""

import argparse
import os

import torch

from neolib import common, files, probe

DEFAULT_RUNS = ["card_en", "card_concise", "card_concise_lr003", "b002"]
CHUNK = 32768


def nearest(weight, vector, top):
	"""Номера и косинусы top ближайших строк таблицы эмбеддингов к вектору."""
	device = weight.device
	query = vector.to(device=device, dtype=torch.float32)
	query = query / query.norm()
	best_values = []
	best_ids = []
	for start in range(0, weight.shape[0], CHUNK):
		block = weight[start:start + CHUNK].float()
		norms = block.norm(dim=1).clamp_min(1e-6)
		scores = (block @ query) / norms
		values, ids = scores.topk(min(top, scores.shape[0]))
		best_values.append(values.cpu())
		best_ids.append(ids.cpu() + start)
	values = torch.cat(best_values)
	ids = torch.cat(best_ids)
	order = values.argsort(descending=True)[:top]
	return ids[order].tolist(), values[order].tolist()


def describe(tokenizer, ids, values):
	"""Токены с косинусами одной строкой."""
	parts = []
	for token_id, value in zip(ids, values):
		token = tokenizer.convert_ids_to_tokens(token_id)
		if token is None:
			# Строки таблицы эмбеддингов за пределами словаря токенизатора (запас модели).
			token = f"<id {token_id}>"
		parts.append(f"`{token}` {value:.2f}")
	return ", ".join(parts)


def main():
	files.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
	parser.add_argument("--top", type=int, default=15)
	args = parser.parse_args()
	probe.use_card_token()

	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	weight = model.get_input_embeddings().weight.detach()
	lines = ["# Ближайшие токены словаря к векторам неологизмов", ""]
	lines.append("Косинус с эмбеддингами словаря Gemma-3-4B. e — стартовое слово, v — обученный вектор, Δ — выученный сдвиг.")
	lines.append("Знак ▁ в начале токена означает пробел перед словом.")
	for run in args.runs:
		vector, init, meta = probe.load_vectors(run)
		delta = vector - init
		lines.append("")
		lines.append(f"## {run}: {meta.get('neologism')}, старт {meta.get('init_word')!r}, язык обучения {meta.get('train_lang')}")
		for name, item in [("e (старт)", init), ("v (обученный)", vector), ("Δ (сдвиг)", delta), ("−Δ (против сдвига)", -delta)]:
			ids, values = nearest(weight, item, args.top)
			text = f"- **{name}:** {describe(tokenizer, ids, values)}"
			lines.append(text)
			print(f"{run} {name}: {describe(tokenizer, ids[:8], values[:8])}")
	os.makedirs("analysis", exist_ok=True)
	path = os.path.join("analysis", "nearest_tokens.md")
	with open(path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines) + "\n")
	print(f"\nСохранено: {path}")


if __name__ == "__main__":
	main()
