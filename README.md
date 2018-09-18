# Отчет по лабораторной работе "Машинное обучение 2"
## по курсу "Искусственый интеллект"

### Выполнил:
    Казаков Григорий
## Задание (Вариант 2)

Построить классификатор выявляющий плохие ответы пользователя на языке Python(реализовать в виде класса). Подобрать или создать датасет и обучить модель. Продемонстрировать зависимость качества классификации от объема и качества выборки. Продемонстрировать работу вашего алгоритма. Обосновать выбор данного алгоритма машинного обучения.

## Отчет по ходу работы

## Используемый алгоритм
В качестве алгоритма классификации возьмем логистическую регрессию, где в качестве функции выступает сигмоида.

Идея работы логистической регресси в том, чтобы получить на вход число(качество объекта)
и выдать вероятность того, что объект принадлежит классу.

## Ход работы

Загрузка данных из csv файла (источник https://www.kaggle.com/stackoverflow/rquestions):
```python
df = pandas.read_csv(path)
```


Основной метод это solve.
В нем происходит разбиение данных на обучающие и тестовые, удаляются html теги с помощью метода delete_html.
Затем преобразованная строка добавляется в список преобразованных строк, таким же образом формируется список из нулей и единиц
"полезности" ответа, если рейтинг ответа больше заданного числа (min_rate), то ответ считается полезным.

```python
for index, row in train_df.iterrows():
    if row["Score"] > self.min_goodness:
        train_goodness.append(1)
    else:
        train_goodness.append(0)
    train_bodies.append(self.transform(row["Body"]))
```

Полученные данные отправляются в модель 

```python
self.classifier = LogisticRegression()
        self.classifier.fit(X_train, train_goodness)
```

После чего такие же преобразование выполняются для тестовой выборки(без создания новой модели).
Преобразованная выборка отправляется в модель, после чего сравнивается количество предсказанных значений
с реальными даннами и выводится точность
```python
def predict(self, test_df):
    test_bodies = []
    test_goodness = []
    for index, row in test_df.iterrows():
        test_bodies.append(self.transform(row["Body"]))
        if row["Score"] > self.min_goodness:
            test_goodness.append(1)
        else:
            test_goodness.append(0)
    X_test = self.vectorizer.transform(test_bodies)
    x_predict = self.classifier.predict(X_test)
    cnt = 0
    for i in range(0, len(x_predict)):
        if test_goodness[i] == x_predict[i]:
            cnt += 1
    print("Predicted values : ", cnt / len(test_goodness), sep=" ")
    return cnt / len(test_goodness)
```

В качестве примера возьмем значение полезности ответа равное трем и разделение выборок 80:20. Проверим точность на тестовой выборке:
```python

```

Получим:
```python

```
