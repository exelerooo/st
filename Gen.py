import random
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
dataset = load_dataset('json', data_files='emotions_traits.json')
import nltk
from nltk.corpus import wordnet
import torch

# Установка необходимых ресурсов
nltk.download('wordnet')
nltk.download('omw-1.4')

# Список слов для генерации заданий
vocab_list = ["happy", "run", "big", "eat", "sleep"]

# Инициализация токенизатора и модели
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Пример данных для тонкой настройки модели (имитация дистилляции)
def create_training_data(vocab_list):
    data = []
    for word in vocab_list:
        # Пример предложений для обучения
        data.append({"text": f"I feel {word} today.", "label": 1})
        data.append({"text": f"I don’t feel {word} at all.", "label": 0})
    return Dataset.from_dict({"text": [d["text"] for d in data], "label": [d["label"] for d in data]})

# Токенизация данных
def tokenize_data(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True)

# Настройка модели (упрощенная дистилляция)
def distill_model():
    dataset = create_training_data(vocab_list)
    tokenized_dataset = dataset.map(tokenize_data, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./distilled_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()
    trainer.save_model("./distilled_model")
    print("Модель дистиллирована и сохранена.")

# Функция генерации предложений
def generate_sentence(word):
    templates = [
        f"I {word} every day.",
        f"She feels {word} when she’s with friends.",
        f"They didn’t {word} yesterday."
    ]
    return random.choice(templates)

# Генерация заданий разных типов
def generate_tasks(vocab_list):
    tasks = {}

    # 1. Close Test (заполнение пропусков)
    word = random.choice(vocab_list)
    sentence = generate_sentence(word).replace(word, "___")
    tasks["close_test"] = {
        "task": f"Заполните пропуск: {sentence}",
        "answer": word
    }

    # 2. True/False
    word = random.choice(vocab_list)
    sentence = generate_sentence(word)
    tasks["true_false"] = {
        "task": f"Верно ли утверждение: {sentence}",
        "answer": "True"  # Для простоты считаем, что все сгенерированные предложения верны
    }

    # 3. Synonyms
    word = random.choice(vocab_list)
    syns = [syn.lemmas()[0].name() for syn in wordnet.synsets(word) if syn.pos() == 'v' or syn.pos() == 'a'][:3]
    tasks["synonyms"] = {
        "task": f"Выберите синоним к слову '{word}': {', '.join(syns)}",
        "answer": syns[0] if syns else "Нет синонимов"
    }

    # 4. Paraphrasing
    word = random.choice(vocab_list)
    sentence = generate_sentence(word)
    paraphrased = sentence.replace(word, random.choice([syn.lemmas()[0].name() for syn in wordnet.synsets(word)][:1] or [word]))
    tasks["paraphrasing"] = {
        "task": f"Перефразируйте: {sentence}",
        "answer": paraphrased
    }

    # 5. Matching
    words = random.sample(vocab_list, min(3, len(vocab_list)))
    meanings = [wordnet.synsets(w)[0].definition() if wordnet.synsets(w) else f"Definition of {w}" for w in words]
    shuffled_meanings = random.sample(meanings, len(meanings))
    tasks["matching"] = {
        "task": f"Сопоставьте слова и их значения:\nСлова: {', '.join(words)}\nЗначения: {', '.join(shuffled_meanings)}",
        "answer": {w: m for w, m in zip(words, meanings)}
    }

    # 6. Antonyms
    word = random.choice(vocab_list)
    ants = [ant.lemmas()[0].name() for syn in wordnet.synsets(word) for ant in syn.lemmas()[0].antonyms()][:1]
    tasks["antonyms"] = {
        "task": f"Назови антоним к слову '{word}':",
        "answer": ants[0] if ants else "Нет антонимов"
    }

    return tasks

# Основной процесс
if __name__ == "__main__":
    # Дистилляция модели (можно закомментировать после первого запуска)
    distill_model()

    # Генерация заданий
    tasks = generate_tasks(vocab_list)
    
    # Вывод заданий
    for task_type, content in tasks.items():
        print(f"\nТип задания: {task_type}")
        print(f"Задание: {content['task']}")
        print(f"Ответ: {content['answer']}")
