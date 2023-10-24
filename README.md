# Команда 6

<b>Наилучшее значение метрики составило: 0,9131 </b>

## Описание шагов работы

1. Загружаем папку с изображениями, проверяем соответсвие расширений разрешенному набору для TensorFlow, иначе заменяем старое изображение на новое, с допустимым расширением;
2. Делим датасет на обучающую и тестовую выборку;
3. Выполняем предобработку входных изображений;
4. Задаем параметры модели - количество слоев, параметры свертки, пуллинга и т.д.;
5. Осуществляем обучение модели;
6. Измеряем среднее значение метрики f1-score по всем 5 классам и время инференса модели при обработке одного изображения в секундах.

Были протестированы различные модели, в том числе кастомная модель со следующей архитектурой,:

```
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same'),
  layers.BatchNormalization(axis=3),
  layers.Activation('relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same'),
  layers.BatchNormalization(axis=3),
  layers.Activation('relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same'),
  layers.BatchNormalization(axis=3),
  layers.Activation('relu'),
  layers.AveragePooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation="softmax")
```

Также была построена и обучена модель, созданная по принципу AlexNet, с небольшой модификацией последних полносвязных слоев:

```
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(96, 11, 4, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(3, 2),
    layers.Conv2D(256, 5, 1, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(3, 2),
    layers.Conv2D(384, 3, 1, activation='relu'),
    layers.Conv2D(384, 3, 1, activation='relu'),
    layers.Conv2D(256, 3, 1, activation='relu'),
    layers.MaxPooling2D(3, 2),
    layers.Flatten(),
    layers.Dense(150, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(70, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='sigmoid'),
```

Среднее значение метрики f1_score на обоих данных моделях было равно примерно ```0,8```.

Помимо этого была также построена модель по архитектуре VGG-19, на ней, при использовании оптимизоватора SGD, удалось добиться значения метрики равного: ```0.9131```.

```
VGG = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(64, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(128, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(256, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.MaxPooling2D(2, 2),

    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2D(512, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),

    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(num_classes, activation='sigmoid'),
])
```

Также были получены результаты ```0.8835``` при использовании AdaGrad и ```~0,7``` с оптимизатором Adam.

Время инференса для одного изображения в построенной модели составляет: ```5.0625 мс```.

На схеме представлена архитетура получившейся модели.

<img src="assets/model.png" width="500"> 

## Результаты

Построенна нейронная сеть для классификации изображений по архитектуре VGG-19, с оптимизатором SGD.
Среднее значение метрики f1-score: ```0.9131```.
Время инференса: ```5.0625 мс``` на одно изображение.

