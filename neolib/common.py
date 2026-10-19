"""Общий код: загрузка модели, шаблон чата, вставка вектора неологизма,
вероятность ответа, функция потерь и генерация.

Главный приём. Модель заморожена. Токен неологизма (например, ~short) добавлен
в токенизатор, но НЕ в таблицу эмбеддингов модели. Перед подачей в модель мы
сами переводим токены в векторы и на место неологизма ставим свой обучаемый
вектор. Модель получает готовые векторы (inputs_embeds). Выходной словарь модели
не меняется, поэтому сама модель неологизм никогда не напишет, а без неологизма
в промпте работает ровно как исходная.
"""

import math
import os
import random

import torch
import torch.nn.functional as F
from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer

import config
# Работа с файлами живёт в files.py (без PyTorch); здесь просто даём к ней доступ.
from neolib.files import append_jsonl, data_path, read_jsonl, run_path, setup_console, write_jsonl  # noqa: F401


# ---------- мелочи ----------

def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def gpu_memory_report():
	if not torch.cuda.is_available():
		return "GPU нет"
	used = torch.cuda.memory_allocated() / 1024 ** 3
	peak = torch.cuda.max_memory_allocated() / 1024 ** 3
	return f"память GPU: сейчас {used:.1f} ГБ, пик {peak:.1f} ГБ"


# ---------- модель ----------

def pick_dtype(device):
	"""Тип чисел для модели: из config.DTYPE или по возможностям видеокарты."""
	if device != "cuda":
		return torch.float32
	if config.DTYPE == "bfloat16":
		return torch.bfloat16
	if config.DTYPE == "float16":
		return torch.float16
	if config.DTYPE == "float32":
		return torch.float32
	# bfloat16 по-настоящему поддерживают видеокарты с архитектурой Ampere и новее (RTX 30xx/40xx, A100, L4).
	# На T4 он только эмулируется и работает очень медленно, поэтому там берём float16.
	major, _ = torch.cuda.get_device_capability()
	if major >= 8:
		return torch.bfloat16
	print("Видеокарта без быстрого bfloat16 (например, T4): беру float16. Gemma во float16 может работать нестабильно.")
	return torch.float16


def loss_scale(model):
	"""Во float16 маленькие градиенты могут обнулиться, поэтому потерю перед backward умножаем.
	Adam почти не чувствителен к постоянному множителю градиента, так что на шаг это не влияет."""
	if model.get_input_embeddings().weight.dtype == torch.float16:
		return 256.0
	return 1.0


def load_model_and_tokenizer(model_name=None):
	"""Возвращает (model, tokenizer, neo_id). Все веса модели заморожены."""
	if model_name is None:
		model_name = config.MODEL_NAME
	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"
		print("ВНИМАНИЕ: CUDA не найдена, всё будет очень медленно.")
	dtype = pick_dtype(device)

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.padding_side = "left"
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	# lstrip=True: пробел перед словом «приклеивается» к нему, как у обычного слова.
	neo_token = AddedToken(config.NEOLOGISM, lstrip=True, rstrip=False, normalized=False, special=False)
	tokenizer.add_tokens([neo_token])
	neo_id = tokenizer.convert_tokens_to_ids(config.NEOLOGISM)

	try:
		model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device)
	except ValueError:
		# Gemma-3-4B и крупнее — мультимодальные модели, у них другой класс.
		from transformers import AutoModelForImageTextToText
		model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype, device_map=device)
	model.eval()
	for param in model.parameters():
		param.requires_grad_(False)
	return model, tokenizer, neo_id


# Gemma3ForConditionalGeneration (Gemma-3-4B и крупнее) не умеет generate(inputs_embeds=...).
# Для генерации собираем из тех же весов текстовую модель Gemma3ForCausalLM: веса общие, копий нет.
_TEXT_MODELS = {}


def find_text_parts(model):
	"""Текстовая часть мультимодальной модели и её lm_head (в разных версиях transformers лежат по-разному)."""
	inner = getattr(model, "model", None)
	language_model = getattr(inner, "language_model", None)
	if language_model is None:
		language_model = getattr(model, "language_model", None)
	if language_model is None:
		return None, None
	if hasattr(language_model, "lm_head"):
		# старые версии: language_model — уже готовая Gemma3ForCausalLM
		return language_model, None
	return language_model, getattr(model, "lm_head", None)


def generation_model(model):
	"""Модель, которой можно передавать inputs_embeds в generate()."""
	if getattr(model.config, "model_type", "") != "gemma3":
		return model
	key = id(model)
	if key in _TEXT_MODELS:
		return _TEXT_MODELS[key]
	language_model, lm_head = find_text_parts(model)
	if language_model is None:
		raise RuntimeError("Не найдена текстовая часть Gemma 3: версия transformers не поддерживается (нужна версия из requirements.txt)")
	if lm_head is None:
		text_model = language_model
	else:
		from transformers import Gemma3ForCausalLM
		with torch.device("meta"):
			text_model = Gemma3ForCausalLM(model.config.text_config)
		text_model.model = language_model
		text_model.lm_head = lm_head
		text_model.generation_config = model.generation_config
	text_model.eval()
	_TEXT_MODELS[key] = text_model
	return text_model


def model_device(model):
	return model.get_input_embeddings().weight.device


def extra_forward_kwargs(model, seq_len):
	"""Gemma 3 в режиме обучения требует token_type_ids (0 = обычный текст, не картинка)."""
	if getattr(model.config, "model_type", "") == "gemma3":
		ids = torch.zeros((1, seq_len), dtype=torch.long, device=model_device(model))
		return {"token_type_ids": ids}
	return {}


def enable_training_mode(model):
	"""Режим для обучения вектора: включаем gradient checkpointing, чтобы влезть в память.
	Dropout у Gemma 3 и Qwen 2.5 равен нулю, так что режим train() ответы не меняет."""
	if config.GRAD_CHECKPOINTING:
		model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
		model.train()
	else:
		model.eval()


# ---------- текст и шаблон чата ----------

def build_user_text(question, suffix):
	"""Вопрос плюс приписка через пустую строку. Пустая приписка — условие «база»."""
	question = question.strip()
	if suffix:
		return question + "\n\n" + suffix
	return question


def neologism_suffix(lang):
	return config.TEMPLATES[lang]["neologism"].format(neo=config.NEOLOGISM)


def target_suffix(lang):
	return config.TEMPLATES[lang]["target"]


def chat_prompt(tokenizer, user_text):
	"""Промпт в шаблоне чата модели, заканчивается началом реплики ассистента."""
	messages = [{"role": "user", "content": user_text}]
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def chat_response_part(tokenizer, user_text, response):
	"""Ответ так, как его видит модель: текст ответа плюс токен конца реплики.
	Токен конца важен: именно он учит модель вовремя останавливаться."""
	prompt = chat_prompt(tokenizer, user_text)
	messages = [
		{"role": "user", "content": user_text},
		{"role": "assistant", "content": response},
	]
	full = tokenizer.apply_chat_template(messages, tokenize=False)
	if not full.startswith(prompt):
		raise ValueError("Шаблон чата устроен неожиданно: полный диалог не начинается с промпта.")
	return full[len(prompt):]


def encode(tokenizer, text):
	"""Текст в id токенов. Служебные токены не добавляем: шаблон чата уже содержит их сам."""
	ids = tokenizer(text, add_special_tokens=False)["input_ids"]
	return torch.tensor(ids, dtype=torch.long)


def count_token(ids, token_id):
	count = 0
	for t in ids.tolist():
		if t == token_id:
			count += 1
	return count


# ---------- векторы ----------

def embed_ids(model, ids, neo_id, neo_vector=None):
	"""Переводит id токенов (T,) в векторы (T, d). На место неологизма ставит neo_vector."""
	emb_layer = model.get_input_embeddings()
	device = emb_layer.weight.device
	ids = ids.to(device)
	is_neo = ids == neo_id
	# id неологизма нет в таблице модели, поэтому временно меняем его на 0.
	safe_ids = torch.where(is_neo, torch.zeros_like(ids), ids)
	embeds = emb_layer(safe_ids.unsqueeze(0)).squeeze(0)
	if bool(is_neo.any()):
		if neo_vector is None:
			raise ValueError("В тексте есть неологизм, но вектор для него не передан.")
		vec = neo_vector.to(device=device, dtype=embeds.dtype).unsqueeze(0)
		embeds = torch.where(is_neo.unsqueeze(-1), vec, embeds)
	return embeds


def embedding_stats(model, n_sample=4096, seed=0):
	"""Типичный размер обычных эмбеддингов: RMS одной координаты и медианная норма вектора."""
	emb_layer = model.get_input_embeddings()
	vocab_size = emb_layer.weight.shape[0]
	generator = torch.Generator().manual_seed(seed)
	ids = torch.randint(0, vocab_size, (n_sample,), generator=generator)
	with torch.no_grad():
		vectors = embed_ids(model, ids, neo_id=-1).float()
	rms = vectors.pow(2).mean().sqrt().item()
	median_norm = vectors.norm(dim=-1).median().item()
	return rms, median_norm


def initial_vector(model, tokenizer, init_word):
	"""Стартовый вектор: копия эмбеддинга обычного нейтрального слова."""
	ids = encode(tokenizer, init_word)
	if ids.shape[0] != 1:
		print(f"ВНИМАНИЕ: стартовое слово {init_word!r} режется на {ids.shape[0]} токена, беру последний.")
	with torch.no_grad():
		vector = embed_ids(model, ids[-1:], neo_id=-1)
	return vector[0].float().clone()


class Neologism(torch.nn.Module):
	"""Обучаемый вектор нового слова.

	Храним его в нормированном виде u, настоящий вектор v = u * scale, где scale —
	типичный размер координаты обычных эмбеддингов этой модели. Так один и тот же
	шаг обучения подходит разным моделям (у Gemma эмбеддинги в десятки раз крупнее, чем у Qwen).
	"""

	def __init__(self, init_vector, scale):
		super().__init__()
		self.scale = scale
		self.u = torch.nn.Parameter(init_vector.float() / scale)

	def vector(self):
		return self.u * self.scale


def save_neologism(path, vector, init_vector, meta):
	"""Сохраняет обученный и стартовый вектор вместе с настройками пробега."""
	folder = os.path.dirname(path)
	if folder:
		os.makedirs(folder, exist_ok=True)
	payload = {
		"vector": vector.detach().float().cpu(),
		"init_vector": init_vector.detach().float().cpu(),
		"meta": meta,
	}
	torch.save(payload, path)


def load_neologism(path):
	return torch.load(path, map_location="cpu")


# ---------- вероятность ответа и потеря ----------

def last_logits(model, embeds, n_last):
	"""Логиты только для последних n_last позиций: так не нужно хранить логиты всей
	последовательности по огромному словарю (у Gemma 3 — 262 тысячи токенов)."""
	kwargs = extra_forward_kwargs(model, embeds.shape[1])
	try:
		out = model(inputs_embeds=embeds, use_cache=False, logits_to_keep=n_last, **kwargs)
	except TypeError:
		out = model(inputs_embeds=embeds, use_cache=False, **kwargs)
	return out.logits[:, -n_last:, :]


def response_logprob(model, prompt_ids, response_ids, neo_id, neo_vector):
	"""Сумма log p(токены ответа | промпт). Градиент идёт только в neo_vector."""
	ids = torch.cat([prompt_ids, response_ids])
	embeds = embed_ids(model, ids, neo_id, neo_vector).unsqueeze(0)
	n_response = response_ids.shape[0]
	logits = last_logits(model, embeds, n_response + 1)
	# Позиция перед каждым токеном ответа предсказывает этот токен; последнюю позицию отбрасываем.
	logits = logits[0, :-1, :].float()
	targets = response_ids.to(logits.device)
	return -F.cross_entropy(logits, targets, reduction="sum")


def apo_up_loss(lc, lr, lc0, lr0, beta):
	"""APO-up (D'Oosterlinck et al.) = DPO + «якорь».

	lc, lr   — log p(хороший ответ), log p(плохой ответ) с текущим вектором;
	lc0, lr0 — то же со стартовым вектором (точка отсчёта).
	Первая часть: хороший ответ должен обгонять плохой сильнее, чем в начале.
	Вторая часть (якорь): хороший ответ должен стать вероятнее сам по себе.
	В самом начале обе части равны ln 2, сумма — 1.386.
	"""
	dpo = -F.logsigmoid(beta * ((lc - lr) - (lc0 - lr0)))
	anchor = -F.logsigmoid(beta * (lc - lc0))
	return dpo + anchor


# ---------- генерация ----------

def eos_token_ids(model, tokenizer):
	"""Все токены конца ответа (у Gemma их несколько)."""
	ids = set()
	gen_eos = model.generation_config.eos_token_id
	if isinstance(gen_eos, int):
		ids.add(gen_eos)
	elif gen_eos is not None:
		for token_id in gen_eos:
			ids.add(int(token_id))
	if tokenizer.eos_token_id is not None:
		ids.add(tokenizer.eos_token_id)
	return ids


def penalty_kwargs(repetition_penalty):
	"""Штраф за повторы. None — как в настройках модели (у Qwen 2.5 это 1.05).
	С inputs_embeds штраф действует только на новые токены, а с input_ids — ещё и на промпт,
	поэтому для проверки «пути совпадают» его выключаем (1.0)."""
	if repetition_penalty is None:
		return {}
	return {"repetition_penalty": repetition_penalty}


@torch.no_grad()
def generate_batch(model, tokenizer, user_texts, neo_id, neo_vector, max_new_tokens, repetition_penalty=None):
	"""Жадная генерация через векторы (inputs_embeds) с выравниванием слева.
	Возвращает список {"text": ..., "truncated": True/False}."""
	rows = []
	for text in user_texts:
		ids = encode(tokenizer, chat_prompt(tokenizer, text))
		rows.append(embed_ids(model, ids, neo_id, neo_vector))
	max_len = 0
	for row in rows:
		if row.shape[0] > max_len:
			max_len = row.shape[0]
	hidden = rows[0].shape[-1]
	pad_vector = embed_ids(model, torch.tensor([tokenizer.pad_token_id]), neo_id)[0]
	batch = pad_vector.expand(len(rows), max_len, hidden).clone()
	mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=batch.device)
	for i, row in enumerate(rows):
		start = max_len - row.shape[0]
		batch[i, start:] = row
		mask[i, start:] = 1
	out = generation_model(model).generate(
		inputs_embeds=batch,
		attention_mask=mask,
		max_new_tokens=max_new_tokens,
		do_sample=False,
		pad_token_id=tokenizer.pad_token_id,
		**penalty_kwargs(repetition_penalty),
	)
	eos_ids = eos_token_ids(model, tokenizer)
	results = []
	for row in out.tolist():
		has_eos = False
		for token_id in row:
			if token_id in eos_ids:
				has_eos = True
				break
		text = tokenizer.decode(row, skip_special_tokens=True).strip()
		results.append({"text": text, "truncated": not has_eos})
	return results


def generate_many(model, tokenizer, user_texts, neo_id, neo_vector, max_new_tokens, batch_size, desc=""):
	"""Генерация для многих промптов: сортируем по длине, чтобы меньше выравнивать."""
	order = list(range(len(user_texts)))
	order.sort(key=lambda i: len(user_texts[i]))
	results = [None] * len(user_texts)
	n_batches = math.ceil(len(order) / batch_size)
	for b in range(n_batches):
		indices = order[b * batch_size:(b + 1) * batch_size]
		texts = []
		for i in indices:
			texts.append(user_texts[i])
		outputs = generate_batch(model, tokenizer, texts, neo_id, neo_vector, max_new_tokens)
		for i, output in zip(indices, outputs):
			results[i] = output
		if desc:
			print(f"  {desc}: пачка {b + 1}/{n_batches}", flush=True)
	return results


@torch.no_grad()
def generate_plain(model, tokenizer, user_text, max_new_tokens, repetition_penalty=None):
	"""Обычная генерация через id токенов — для проверки, что путь через векторы ничего не меняет."""
	ids = encode(tokenizer, chat_prompt(tokenizer, user_text)).unsqueeze(0).to(model_device(model))
	mask = torch.ones_like(ids)
	out = model.generate(
		input_ids=ids,
		attention_mask=mask,
		max_new_tokens=max_new_tokens,
		do_sample=False,
		pad_token_id=tokenizer.pad_token_id,
		**penalty_kwargs(repetition_penalty),
	)
	new_tokens = out[0, ids.shape[1]:]
	return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
