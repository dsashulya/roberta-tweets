# Определение расистских/сексистских твитов с помощью RoBERTa

Содержание:

- [Процесс дообучения модели и PR-Curve](https://github.com/dsashulya/roberta-tweets/blob/main/training.ipynb)
- [CLI](https://github.com/dsashulya/roberta-tweets/blob/main/cli.py)
- [Файл с моделью](https://drive.google.com/file/d/1-Kk-_nUfy7krp0UBv1vck7yEpvD8zFPq/view?usp=sharing)

Задание состоит из 3 частей:

- Возьмите предобученую модель [RoBERTa](https://huggingface.co/transformers/model_summary.html#roberta) из библиотеки transformers от 🤗. Дообучите модель определять является ли твит расистским или сексистким с использованием соответствующего [датасета](https://huggingface.co/datasets/tweets_hate_speech_detection). Не забудьте поделить датасет на тренировочную и тестовую выборку.

- Оцените качество тестовой модели с использованием метрики Accuracy до и после дообучения. Проанализируйте, как выбор порога классификации влияет на точность с помощью PR-curve.

- (Опционально) Реализуйте простой CLI, который принимает на вход предложение и выводит в консоль результат оценки модели, а также время, которое понадобилось модели на обработку этого предложения.


### Примеры работы CLI:

<img src="imgs/1.png">
<img src="imgs/2.png">
<img src="imgs/4.png">
