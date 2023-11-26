# Вычисление оптического потока

## Описание задачи

Язык программирования - Python.
Разрешено использовать любую библиотеку для машинного обучения (PyTorch, TensorFlow, Keras и др.). Запрещено использование готовых архитектур "из коробки" одной строчкой, в коде должны быть прописаны слои модели.

Необходимо создать модель для вычисления оптического потока изображения, обучить ее на предложенном датасете, а также измерить время работы модели на тестовом датасете и посчитать метрику.

## Датасет

Набор данных содержит 1500 наборов данных, состоящих из 2 изображений и соответсвующего им оптического потока.<br/>
[Ссылка на датасет](https://drive.google.com/file/d/1CI6Qv-bo--vredyq3JpSSRftOklSzQRj/view?usp=sharing).

Предложенный датасет представляет из себя выборку из синтетического датасета 'Flying Chairs'. Изображения показывают рендеры 3D-моделей кресел, движущихся перед случайным фоном, при этом движения как стульев и фона являются чисто плоскостными.

```
@InProceedings{DFIB15,
  author    = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz{\i}rba{\c{s}} and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title     = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle = "IEEE International Conference on Computer Vision (ICCV)",
  month     = " ",
  year      = "2015",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15"
}
```

## Метрики

Для оценки качества модели используется следующая метрика:
* EPE - End-point error.
Метрика рассчитывается как Евклидово расстояние между истинным оптическим потоком и полученным в результате вычислений.


```math
 EPE = ||V_{gt} - V_{calc}||_2 = \sqrt{(\Delta x_{gt} - \Delta x_{calc})^2 + (\Delta y_{gt} - \Delta y_{calc})^2}
```

## Baseline

Модель FlowNet S, оптимизатор Adam, loss - EPE (совпадает с метрикой), batch_size = 8, количество эпох 20.

Результаты:
|     EPE (train-val set)     |     EPE (test set)     |     Time per image, sec    |
|-----------------------------|------------------------|----------------------------|
|           8.04              |           7.0          |           0.0107           |
