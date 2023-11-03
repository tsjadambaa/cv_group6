# Улучшение яркости изображения
## Описание задачи
* Задача - улучшение яркости изображения
* Язык программирования Python
* Ограничений по использованию библиотек и сторонних функций нет

## Датасет
Набор данных содержит содержит 745 примеров тёмных картинок. В каталоге low хранятся тёмные изображения, в каталоге high - соответсвующие им яркие изображения. Пример загрузки датасета в PyTorch приведён в файле dataset.py       
[Ссылка на датасет](https://drive.google.com/file/d/1ThoPb1flnfXDpRIytgBd7_e9Kv_lPnbo/view) 

## Метрики
Пример расчёта метрик представлен в файле evaluation.py. Для оценки качества используются следующие метрики:
* [PSNR](https://ru.wikipedia.org/wiki/Пиковое_отношение_сигнала_к_шуму) - Пиковое отношение сигнала к шуму;
* [SSIM](https://ru.wikipedia.org/wiki/SSIM) - Индекс структурного сходства;
* [LPIPS](https://github.com/richzhang/PerceptualSimilarity#c-about-the-metric) - Learned Perceptual Image Patch Similarity.  

## Результаты
* Наше решение реализовано на базе архитектуры DCE Net - это архитектура типа Кодер-Декодер. Оптимизатор - Adam, 200 эпох, batch size = 8. Тренировочная и тестовая выборки составляют 70% и 30% от исходного набора, соответсвенно.
* На обработку одного изображения у нашего решения уходит 1,5 (мс).
* Таблица содержит результаты вычисления метрик при разном размере батча.

|batch size|PSNR(🠕)|SSIM(🠕)|LPIPS(🠗)|
:---:|:---:|:---:|:---:
1|13.07|0.63|0.24
2|18.25|0.70|0.24
4|23.97|0.77|0.24
8|29.71|0.82|0.24