# Генерация изображений

## Описание задачи

Язык программирования - Python.
Разрешено использовать любую библиотеку для машинного обучения (PyTorch, TensorFlow, Keras и др.). Запрещено использование готовых архитектур "из коробки" одной строчкой, в коде должны быть прописаны слои модели.

Необходимо создать модель для генерации изображений рукописных чисел, обучить ее на предложенном датасете, а также визуально оценить качество выбранной модели

## Датасет

Предложенный датасет для данной задачи - MNIST. Датасет представляет набор написанных от руки цифр в диапазоне от 0 до 9. Он содержит обучающий набор из 60 000 изображений и 10 000 тестовых изображений.
Для того, чтобы использовать данный датасет, рекомендуется воспользоваться API библиотеки PyTorch или TensorFlow (смотря на то, какой библиотекой вы воспользуетесь)

![mnist_img](https://github.com/VladislavEpifanow/CV-Lab-7/blob/main/img/MNIST.png)

PyTorch:
[Ссылка на датасет в PyTorch](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

Tensorflow/Keras:
[Ссылка на датасет в Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist).


## Baseline

В качестве baseline используется модель VAE+GAN, batch_size = 64, epoch = 20, предобработок и аугментаций нет. 

Пример полученных изображений с помощью VAE+GAN:

![mnist_img](https://github.com/VladislavEpifanow/CV-Lab-7/blob/main/img/VAE%2BGAN%20results.png)