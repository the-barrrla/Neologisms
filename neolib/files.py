"""Работа с файлами и консолью. Без PyTorch, поэтому годится и для анализа."""

import json
import os
import sys

import config


def setup_console():
	"""Чтобы кириллица нормально печаталась в консоли Windows."""
	try:
		sys.stdout.reconfigure(encoding="utf-8")
		sys.stderr.reconfigure(encoding="utf-8")
	except AttributeError:
		pass


def read_jsonl(path):
	rows = []
	with open(path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def _ensure_folder(path):
	folder = os.path.dirname(path)
	if folder:
		os.makedirs(folder, exist_ok=True)


def write_jsonl(path, rows):
	_ensure_folder(path)
	with open(path, "w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, rows):
	_ensure_folder(path)
	with open(path, "a", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def data_path(name):
	return os.path.join(config.DATA_DIR, name)


def run_path(run, name):
	return os.path.join(config.RUNS_DIR, run, name)
