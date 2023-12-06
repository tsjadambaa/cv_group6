# Вычисление оптического потока

## Описание задачи

Язык программирования - Python.
Разрешено использовать любую библиотеку для машинного обучения (PyTorch, TensorFlow, Keras и др.). Запрещено использование готовых архитектур "из коробки" одной строчкой, в коде должны быть прописаны слои модели.

Необходимо создать модель для вычисления оптического потока изображения, обучить ее на предложенном датасете, а также измерить время работы модели на тестовом датасете и посчитать метрику.

## Датасет

Датасет содержит 1485 примеров, каждый из которых состоит из 2 изображений и соответсвующего им оптического потока (<sample_id>_img1.ppm <sample_id>_img2.ppm <sample_id>_flow.flo).<br/>
[Ссылка на датасет](https://drive.google.com/file/d/1ipbk2nGpVlTHHY7i4GfTf-cJ0VHfvURd/view?usp=sharing).

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
Для загрузки можно использовать функции, представленные в файлах [utils.py](./utils.py) и [loader.py](./loader.py).

## Метрики

Для оценки качества модели используется следующая метрика:
* EPE - End-point error. <br/>

Метрика рассчитывается как Евклидово расстояние между истинным оптическим потоком и полученным в результате вычислений.


```math
 EPE = ||V_{gt} - V_{calc}||_2 = \sqrt{(\Delta x_{gt} - \Delta x_{calc})^2 + (\Delta y_{gt} - \Delta y_{calc})^2}
```
Реализация расчета метрики на python с использованием pytorch представлена в файле [utils.py](./utils.py) в функции _mean_epe_
## Baseline

Модель FlowNet S, оптимизатор Adam, loss - EPE (совпадает с метрикой), batch_size = 8, количество эпох 20.

Результаты:
|     EPE (train-val set)     |     EPE (test set)     |     Time per image, sec    |
|-----------------------------|------------------------|----------------------------|
|           8.04              |           7.0          |           0.0107           |


## Описание решения

Решение построено на основе архитектуры FlowNet S, струтура используемой сети:
```
class FlowNetSimple(pl.LightningModule):

    def __init__(self):
        super().__init__()
        '''
        One could use sequential here too, but since we will be needing the output of some of the conv layers in encoder
        , as an input to decoder layers, it is better to specify each layer seperately.
        '''
        self.conv1 = conv(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = conv(256, 256)
        self.conv4 = conv(256, 512, stride=2, padding=1)
        self.conv4_1 = conv(512, 512)
        self.conv5 = conv(512, 512, stride=2, padding=1)
        self.conv5_1 = conv(512, 512)
        self.conv6 = conv(512, 1024, stride=2, padding=1)

        self.deconv5 = transposedConv(1024, 512)
        self.deconv4 = transposedConv(1024, 256)
        self.deconv3 = transposedConv(770, 128)
        self.deconv2 = transposedConv(386, 64)


        self.predict_flow5 = predict_flow(1024)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 5, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 5, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 5, 2, 1, bias=False)

        self.upsample_bilinear = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3_1 = self.conv3_1(out_conv3)
        out_conv4 = self.conv4(out_conv3_1)
        out_conv4_1 = self.conv4_1(out_conv4)
        out_conv5 = self.conv5(out_conv4_1)
        out_conv5_1 = self.conv5_1(out_conv5)
        out_conv6 = self.conv6(out_conv5_1)

        out_deconv5 = self.deconv5(out_conv6)
        input_to_deconv4 = torch.cat((crop_like(out_deconv5, out_conv5_1), out_conv5_1), 1)
        flow5 = self.predict_flow5(input_to_deconv4)

        upsampled_flow5_to_4 = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4_1)
        out_deconv4 = self.deconv4(input_to_deconv4)
        input_to_deconv3 = torch.cat((crop_like(out_deconv4, out_conv4_1), out_conv4_1, upsampled_flow5_to_4), 1)
        flow4 = self.predict_flow4(input_to_deconv3)

        upsampled_flow4_to_3 = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3_1)
        out_deconv3 = self.deconv3(input_to_deconv3)
        input_to_deconv2 = torch.cat((crop_like(out_deconv3, out_conv3_1), out_conv3_1, upsampled_flow4_to_3), 1)
        flow3 = self.predict_flow3(input_to_deconv2)

        upsampled_flow3_to_2 = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = self.deconv2(input_to_deconv2)
        input_to_upsamling = torch.cat((crop_like(out_deconv2, out_conv2), out_conv2, upsampled_flow3_to_2), 1)

        flow2 = self.predict_flow2(input_to_upsamling)
        output = self.upsample_bilinear(flow2)

        return output

```

Для обучения и тестирования модели использовался фреймворк Pytorch Lighting.

## Визуализация результата

![image](assets/img1.png)  <br/>
![image](assets/img2.png)  
