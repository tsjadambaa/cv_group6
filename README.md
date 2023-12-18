# Решение
## Описание задачи
* Задача - генерация изображений на основе датасета MNIST. <br/>
Полное описание представлено в файле [task.md](task.md)


## Результаты
### Описание архитектуры решения
В качестве способа решения поставленной задачи было решено построить модель на основе архитектуры VAE+GAN. Данная сеть является трехкомпонентной и состоит из следующих основных частей, которые, по сути, обучаются отдельно друг от друга - энкодер, декодер и дискриминатор. Задача энкодера - научиться переводить изображения в скрытое пространство признаков, декодера - восстанавливать исходное изображение по вектору скрытых признаков. Дискриминатор используется в данной модели только на этапе обучение, и его задача - отличать сгенерированное изображение от реального. Применение дискриминатора для обучения VAE позволяет получить более качественную генерацию изображений, т.к. иначе тяжело подобрать хорошую метрику для оценки качества работы VAE. 

Нами было решено использовать не просто VAE, а CVAE - conditional variation autoencoder, на вход которого, помимо самого изображения также подаются метки классов. Добавление меток классов позволяет более качественно моделировать многообразие данных.

При обучении использовались три отдельных лосса для каждой из частей моделей, а также три оптимизатора - RMSProp, с разными значениями learning rate.

```
opt_enc = tf.train.RMSPropOptimizer(0.001)
opt_dec = tf.train.RMSPropOptimizer(0.0001)
opt_dis = tf.train.RMSPropOptimizer(0.001)
```

Обучение проводилос на 50 эпохах.
Также, поскольку для модели VAE+GAN зачастую характерно неравномерное обучение, было выбрано разное количество шагов для VAE и дискриминатора. Так, на 1 шаг обучения дискриминатора приходилось 4 шага обучения для VAE, поскольку в противном случае дискриминатор быстро начинал переобучаться.


## Сравнение результатов восстановления данных на тестовых данных в процессе обучения
Пример 1:
![img1.png](assets/img1.png)
Пример 2:
![img2.png](assets/img2.png)
Пример 3:
![img3.png](assets/img3.png)
Пример 4:
![img4.png](assets/img4.png)
Пример 5:
![img5.png](assets/img5.png)
Пример 6:
![img6.png](assets/img6.png)
Пример 7:
![img7.png](assets/img7.png)
Пример 8:
![img8.png](assets/img8.png)



## Структура сети
### Энкодер
```python
x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(img)
x = LeakyReLU()(x)
x = MaxPool2D((2, 2), padding='same')(x)

x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
x = LeakyReLU()(x)

x = Flatten()(x)
x = concatenate([x, lbl])

h = Dense(64)(x)
h = LeakyReLU()(h)

z_mean    = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(K.clip(z_log_var/2, -2, 2)) * epsilon
l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model([img, lbl], [z_mean, z_log_var, l], name='Encoder')
```

### Декодер
```python
x = concatenate([z, lbl])
x = Dense(7*7*128)(x)
x = LeakyReLU()(x)
x = Reshape((7, 7, 128))(x)
x = UpSampling2D(size=(2, 2))(x)

x = Conv2D(64, kernel_size=(5, 5), padding='same')(x)
x = LeakyReLU()(x)

x = Conv2D(32, kernel_size=(3, 3), padding='same')(x)
x = UpSampling2D(size=(2, 2))(x)
x = LeakyReLU()(x)

decoded = Conv2D(1, kernel_size=(5, 5), activation='sigmoid', padding='same')(x)
decoder = Model([z, lbl], decoded, name='Decoder')
```

### Дискриминатор
```python
x = Conv2D(128, kernel_size=(7, 7), strides=(2, 2), padding='same')(img)
x = MaxPool2D((2, 2), padding='same')(x)
x = LeakyReLU()(x)

repeat = RepeatVector(int(x.shape[1]) * int(x.shape[2]))(lbl)
repeat = Reshape((x.shape[1], x.shape[2], lbl.shape[1]))(repeat)
x = concatenate([x, repeat])

x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = LeakyReLU()(x)

l = Conv2D(16, kernel_size=(3, 3), padding='same')(x)
x = LeakyReLU()(x)

h = Flatten()(x)
d = Dense(1, activation='sigmoid')(h)
discrim = Model([img, lbl], [d, l], name='Discriminator')
```
## Результаты генерации картинок нужного лейбла из шума


![img9.png](assets/img9.png) ![img10.png](assets/img10.png) ![img11.png](assets/img11.png)
![img12.png](assets/img12.png) ![img13.png](assets/img13.png) ![img14.png](assets/img14.png)
![img15.png](assets/img15.png) ![img16.png](assets/img16.png) ![img17.png](assets/img17.png)
![img18.png](assets/img18.png)
