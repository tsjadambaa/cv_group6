### Команда 6

## Описание

Шаги работы:
1. Считываем исходное изображение и шаблон. Применяем детектор ключевых точек SIFT для поиска ключевых точек на исходном изображении и на шаблоне.
2. Для наглядности результата отметим ключевые точки на обоих изобрадениях и сохраним изображения в папку.
3. Используя ``FlannBasedMatcher`` из библиотеки openCV и его метод ``knnMatch``, и сравнивая расстояния, найдем соответсвия между ключевыми точками исходного изображения и изображения-шаблона, чтобы определить общие точки. Тем самым мы сможем отобрать только те ключевые точки исходного изображения, которые соответсвуют искомому объекту.
4. В процессе обхода ключевых точек будем запоминать координаты x для самой левой и самой правой точек, координаты y для самой верхней и самой нижней точек. При этом округляя в меньшую или большую сторону, для того, чтобы выделить координаты "с запасом" и весь объект попал в рамку, в случае, когда координаты ключевых точек являются нецелыми числами.
5. Для наглядности используем ``drawMatches``, который склеивает исходное изображение и шаблон и соединяет линиями идентичные ключевые точки на обоих изображениях.
6. В качестве координат рамки берем ``(x1, y1) и (x2, y2)``, где ``x1`` - самая левая координата, ``y1`` - самая верняя, ``x2`` - самая правая и ``y2`` - самая нижняя соответсвенно.
