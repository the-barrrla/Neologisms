"""Диагностика: почему вектор хорошо учится на парах, но не меняет сгенерированные ответы.

Сравнивает три способа посчитать вероятность ответа:
	A — как при обучении (режим train, gradient checkpointing, мультимодальная модель);
	B — та же модель в режиме eval;
	C — текстовая модель, через которую идёт генерация.
Если A сильно отличается от B и C, обучение и генерация «видят» разное.
Ещё проверяет, не подглядывает ли модель в будущие токены при обучении.

Запуск: python 07_diagnose.py --run short_en   (около минуты после загрузки модели)
"""

import argparse

import torch
import torch.nn.functional as F

import config
from neolib import common


def token_logprobs(model, prompt_ids, response_ids, neo_id, vector):
	"""log p каждого токена ответа отдельно."""
	ids = torch.cat([prompt_ids, response_ids])
	embeds = common.embed_ids(model, ids, neo_id, vector).unsqueeze(0)
	n_response = response_ids.shape[0]
	logits = common.last_logits(model, embeds, n_response + 1)
	logits = logits[0, :-1, :].float()
	targets = response_ids.to(logits.device)
	logp = F.log_softmax(logits, dim=-1)
	return logp.gather(1, targets.unsqueeze(1)).squeeze(1)


def top_next_tokens(model, tokenizer, prompt_ids, neo_id, vector, k=5):
	"""k самых вероятных следующих токенов после промпта, с вероятностями."""
	embeds = common.embed_ids(model, prompt_ids, neo_id, vector).unsqueeze(0)
	logits = common.last_logits(model, embeds, 1)[0, -1].float()
	probs = torch.softmax(logits, dim=-1)
	values, indices = probs.topk(k)
	parts = []
	for value, index in zip(values.tolist(), indices.tolist()):
		parts.append(f"{tokenizer.decode([index])!r} {value:.2f}")
	return ", ".join(parts)


def main():
	common.setup_console()
	parser = argparse.ArgumentParser()
	parser.add_argument("--run", required=True)
	parser.add_argument("--lang", default="en")
	args = parser.parse_args()

	model, tokenizer, neo_id = common.load_model_and_tokenizer()
	saved = common.load_neologism(common.run_path(args.run, "neologism.pt"))
	trained = saved["vector"]
	init = saved["init_vector"]

	print("\n[1] Насколько вектор ушёл от стартового слова")
	cosine = F.cosine_similarity(trained, init, dim=0).item()
	shift = ((trained - init).norm() / init.norm()).item()
	print(f"  косинус с ' accurate': {cosine:.3f}; сдвиг: {shift:.2f} от длины вектора")

	suffix = common.neologism_suffix(args.lang)
	pairs = common.read_jsonl(common.data_path(f"pairs_{args.lang}.jsonl"))[:3]
	examples = []
	for pair in pairs:
		user_text = common.build_user_text(pair["question"], suffix)
		prompt_ids = common.encode(tokenizer, common.chat_prompt(tokenizer, user_text))
		chosen_ids = common.encode(tokenizer, common.chat_response_part(tokenizer, user_text, pair["chosen"]))
		rejected_ids = common.encode(tokenizer, common.chat_response_part(tokenizer, user_text, pair["rejected"]))
		examples.append((prompt_ids, chosen_ids[:config.MAX_RESPONSE_TOKENS], rejected_ids[:config.MAX_RESPONSE_TOKENS]))

	text_model = common.generation_model(model)
	paths = [("A обучение", model, True), ("B eval", model, False), ("C генерация", text_model, False)]

	print("\n[2] log p(хороший) − log p(плохой) на трёх обучающих парах")
	print("    (у правильного вектора разница большая и одинаковая во всех трёх путях)")
	for name, path_model, training in paths:
		if training:
			common.enable_training_mode(model)
		else:
			model.eval()
		line = []
		with torch.no_grad():
			for prompt_ids, chosen_ids, rejected_ids in examples:
				for label, vector in (("старт", init), ("обуч", trained)):
					lc = token_logprobs(path_model, prompt_ids, chosen_ids, neo_id, vector).sum().item()
					lr = token_logprobs(path_model, prompt_ids, rejected_ids, neo_id, vector).sum().item()
					line.append(f"{label} {lc - lr:+.0f}")
		print(f"  {name}: " + " | ".join(line))

	print("\n[3] Не подглядывает ли модель вперёд (первые 5 токенов: полный ответ против обрезанного)")
	prompt_ids, chosen_ids, _ = examples[0]
	for name, path_model, training in paths:
		if training:
			common.enable_training_mode(model)
		else:
			model.eval()
		with torch.no_grad():
			full = token_logprobs(path_model, prompt_ids, chosen_ids, neo_id, trained)[:5].sum().item()
			short = token_logprobs(path_model, prompt_ids, chosen_ids[:5], neo_id, trained).sum().item()
		status = "OK" if abs(full - short) < 0.5 else "ПОДГЛЯДЫВАЕТ"
		print(f"  {name}: {full:.2f} против {short:.2f} [{status}]")

	print("\n[4] Самые вероятные первые слова ответа на тестовый вопрос")
	question = common.read_jsonl(common.data_path(f"questions_test_{args.lang}.jsonl"))[1]["question"]
	prompt_ids = common.encode(tokenizer, common.chat_prompt(tokenizer, common.build_user_text(question, suffix)))
	model.eval()
	with torch.no_grad():
		for label, vector in (("старт", init), ("обуч", trained)):
			print(f"  {label}, путь B: {top_next_tokens(model, tokenizer, prompt_ids, neo_id, vector)}")
			print(f"  {label}, путь C: {top_next_tokens(text_model, tokenizer, prompt_ids, neo_id, vector)}")
		target_prompt = common.encode(tokenizer, common.chat_prompt(tokenizer, common.build_user_text(question, common.target_suffix(args.lang))))
		print(f"  явная просьба, путь C: {top_next_tokens(text_model, tokenizer, target_prompt, neo_id, None)}")


if __name__ == "__main__":
	main()
