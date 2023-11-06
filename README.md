# Решение
## Описание задачи
* Задача - улучшение яркости изображения

## Метрики
Для оценки качества используются следующие метрики:
* [PSNR](https://ru.wikipedia.org/wiki/Пиковое_отношение_сигнала_к_шуму) - Пиковое отношение сигнала к шуму;
* [SSIM](https://ru.wikipedia.org/wiki/SSIM) - Индекс структурного сходства;
* [LPIPS](https://github.com/richzhang/PerceptualSimilarity#c-about-the-metric) - Learned Perceptual Image Patch Similarity.  

## Результаты
* Наше решение реализовано на базе метода Zero-Reference Deep Curve Estimation (ZeroDCE). В его основе лежит архитектура сети DCE Net, которая позволяет оценить попиксельные кривые и кривые высокого порядка, для настройки динамического диапазона изображения. Преимуществом метода является независимость от парных или непарных тренировочных данных, что позволяет получать хорошие результаты независимо от освещенности сцены. Также в данном подходе используется набор функций потерь, позволяющих оценить качество изображения без привязки к эталонным изображениям.
* Оптимизатор - Adam, 100 эпох, значение batch size указан в таблице. 
* Тренировочная и тестовая выборки составляют 80% и 20% от исходного набора.

## Значения для валидационной выборки

|  model   | batch size | PSNR(🠕) | SSIM(🠕) | LPIPS(🠗) | Time (s) |
|:--------:|:----------:|:--------:|:--------:|:---------:|:--------:|
| Zero-DCE |     8      |  31.73   |   0.84   |   0.217   |  0.006   |
| Zero-DCE |     14     |  36.84   |   0.88   |   0.211   |   0.01   |
| Zero-DCE |     18     |  38.37   |   0.89   |   0.214   |   0.02   |
| Zero-DCE |     20     |  39.96   |  0.90   |   0.215   |   0.02   |

## Значения для тестовой выборки

|  model   | batch size | PSNR(🠕) | SSIM(🠕) | LPIPS(🠗) | Time (s) |
|:--------:|:----------:|:--------:|:--------:|:---------:|:--------:|
| Zero-DCE |     14     |  35.41   |   0.89   |   0.143   |  0.008   |
| Zero-DCE |     18     |  38.06   |   0.90   |   0.143   |   0.01   |
| Zero-DCE |     20      |  37.30   |   0.90   |   0.143   |   0.01   |

# Структура сети
```
  (e_conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_conv3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_conv4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_conv5): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_conv6): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (e_conv7): Conv2d(64, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (upsample): UpsamplingBilinear2d(scale_factor=2.0, mode='bilinear')
```
