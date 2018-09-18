﻿# Отчет по лабораторной работе "Машинное обучение 2"
## по курсу "Искусственый интеллект"

### Выполнил:
    Казаков Григорий
## Задание (Вариант 2)

Построить классификатор выявляющий плохие ответы пользователя на языке Python(реализовать в виде класса). Подобрать или создать датасет и обучить модель. Продемонстрировать зависимость качества классификации от объема и качества выборки. Продемонстрировать работу вашего алгоритма. Обосновать выбор данного алгоритма машинного обучения.

## Отчет по ходу работы

## Используемый алгоритм
В качестве алгоритма классификации возьмем логистическую регрессию, где в качестве функции выступает сигмоида.

Идея работы логистической регресси в том, чтобы получить на вход число и выдать вероятность того, что объект принадлежит классу.

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
    for index, row in train.iterrows():
        if row["Score"] > min_rate:
            bin_rate.append(1)
        else:
            bin_rate.append(0)
        text_rate.append(delete_http(row["Body"]))
```

Полученные данные отправляются в модель 

```python
    classifier = LogisticRegression()
    classifier.fit(fit_train, bin_rate)
```

Далее соответствующее преобразование выполняется для тестовой выборки.
Преобразованная выборка отправляется в модель, после чего сравнивается количество предсказанных значений с реальными даннами и выводится точность:
```python
test_text_rate = []
    test_bin_rate = []
    for index, row in test.iterrows():
        test_text_rate.append(delete_http(row["Body"]))
        if row["Score"] > min_rate:
            test_bin_rate.append(1)
        else:
            test_bin_rate.append(0)
   
```

В качестве примера возьмем значение полезности ответа равное трем и разделение выборок 80:20. Проверим точность на тестовой выборке:
```python
 fit_test = vectorizer.transform(test_text_rate)
    test_predict = classifier.predict(fit_test)
    cnt = 0
    for i in range(0, len(test_predict)):
        if test_bin_rate[i] == test_predict[i]:
            cnt += 1
    print("Accuracy : ", cnt / len(test_bin_rate), sep=" ")
    return cnt / len(test_bin_rate)
```

Получим:
```python
runfile('C:/Users/now20/Desktop/ml2/app.py', wdir='C:/Users/gkazakov/Desktop/ml2')
Predicted values :  0.9312771641612505
0.9312771641612505
```
Таким образом, для нашего примера точность составила 93%.
