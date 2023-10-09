# Команда 6

## Описание
Шаги работы:
1. Выделены ключевые точки на изображении и на шаблоне;
2. Выполнено попарное сравнение и определены те, которые имеют соответствия;
3. Определены самые крайние точки по всем четырем направлениям, затем по ним - координаты левого верхнего и правого нижнего углов рамки;
4. Выполнен расчет метрики.

<b>Наилучшее значение метрики составило: 0,5656 </b>

## Подробный разбор шагов

1. Считываем исходное изображение и шаблон. Применяем детектор ключевых точек SIFT для поиска ключевых точек на исходном изображении и на шаблоне.<br/>

```python
sift = cv2.SIFT_create()
input_kp, input_desc = sift.detectAndCompute(input_image, None)
pattern_kp, pattern_desc = sift.detectAndCompute(pattern_template,None)
```

2. Для наглядности результата отметим ключевые точки на обоих изображениях и сохраним изображения в папку.
   
Пример 1 - Выделение ключевых точек на изображении и шаблоне<br/>

<img src="assets/kp_38.jpg" width="500"> <img src="assets/kp_cropped_img_38.jpg" width="250">

3. Используя ``FlannBasedMatcher`` из библиотеки openCV и его метод ``knnMatch``, и сравнивая расстояния, найдем соответсвия между ключевыми точками исходного изображения и изображения-шаблона, чтобы определить общие точки. Тем самым мы сможем отобрать только те ключевые точки исходного изображения, которые соответсвуют искомому объекту.

```python
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(input_desc, pattern_desc, k=2)
```

4. В процессе обхода ключевых точек будем запоминать координаты x для самой левой и самой правой точек, координаты y для самой верхней и самой нижней точек. При этом округляя в меньшую или большую сторону, для того, чтобы выделить координаты "с запасом" и весь объект попал в рамку, в случае, когда координаты ключевых точек являются нецелыми числами.
6. Для наглядности используем ``drawMatches``, который склеивает исходное изображение и шаблон и соединяет линиями идентичные ключевые точки на обоих изображениях.
   
Пример 2 - Выявление соответсвий между ключевыми точками изображения и шаблона<br/>

<img src="assets/result_38.jpg" width="900"> 

6. В качестве координат рамки берем ``(x1, y1) и (x2, y2)``, где ``x1`` - самая левая координата, ``y1`` - самая верхняя, ``x2`` - самая правая и ``y2`` - самая нижняя соответственно.
## Исходный код функции matchTemplate

```python

# поиск по ключевым точкам, получить peak coord

def matchTemplate(path_img, path_pattern):
    
    input_image = cv2.imread(path_img, 0)
    pattern_template = cv2.imread(path_pattern, 0)

    input_name = path_img[path_img.rfind('/') + 1:path_img.rfind('.jpg')]
    pattern_name = path_pattern[path_pattern.rfind('/') + 1:path_pattern.rfind('.jpg')]                                                                     

    path = './key_points'
    is_exist  = os.path.exists('./key_points')
    
    if not os.path.exists(path):
        os.mkdir(path)
        
    if not os.path.exists(path + '/input'):
        os.mkdir(path + '/input')

    if not os.path.exists(path + '/pattern'):
        os.mkdir(path + '/pattern')

    if not os.path.exists('./result'):
        os.mkdir('./result')

    sift = cv2.SIFT_create()
    
    input_kp, input_desc = sift.detectAndCompute(input_image, None)
    
    input_image_kp = cv2.drawKeypoints(input_image, input_kp, input_image)
    kp_input_name = 'key_points/input/kp_' + input_name + '.jpg'
    cv2.imwrite(kp_input_name, input_image_kp)
    
    pattern_kp, pattern_desc = sift.detectAndCompute(pattern_template,None)
    
    pattern_image_kp = cv2.drawKeypoints(pattern_template, pattern_kp, pattern_template)
    kp_pattern_name = 'key_points/pattern/kp_' + pattern_name + '.jpg'
    cv2.imwrite(kp_pattern_name, pattern_image_kp)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(input_desc, pattern_desc, k=2)

    height, width =  input_image.shape
    
    r = 0
    b = 0
    l = width
    t = height
    
    ratio = 0.4
    points = []

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < ratio*n.distance:
                points.append(m)
                img1_idx = m.queryIdx
                (x, y) = input_kp[img1_idx].pt
                if x > r:
                    r = math.floor(x) + 1
                if x < l:
                    l = math.floor(x)
                if y < t:
                    t = math.floor(y)
                if y > b:
                    b = math.floor(y) + 1            

        except ValueError:
            pass
    
        
    result = cv2.drawMatches(input_image, input_kp, pattern_template, pattern_kp, points, None)
    cv2.imwrite('result/' + 'result_' + input_name + '.jpg', result)
    

    highlight_start = (l, t)
    highlight_end = (r, b)
    
    return highlight_start, highlight_end
```

## Результаты
В процессе выполнения работы были сравнены различные алгоритмы детекции ключевых точек, а также их гиперпараметры. Представленные параметры были отмечены, как дающие наилучшие значения метрики: <b>0,5656</b>. 
Несмотря на то, что детекция ключевых точек алгоритмом ORB по скорости значительно опережает алгоритм SIFT, метрика полученная с помощью ORB равна <b>0,1177</b>, при этом для сравнения ключевых точек использовался ```FlannBasedMatcher```, при сравнении похожести точек использовался коэффициент ```ratio = 0,4```, такой же как и в случае со SIFT. При этом были использованы следующие параметры:

```python

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
   table_number = 11, # 12
   key_size = 20,     # 20
   multi_probe_level = 1) #2
   search_params = dict(checks=50)
```

При использовании ```ratio = 0,6``` точность повысилась до <b>0.1226</b>, однако время работы многократно возросло и превысило даже время у SIFT, несмотря на незначительный прирост значения метрики. Таким образом точность алгоритма SIFT на данном датасете оказалась приближительно в 5 раз выше, чем у ORB, хотя значение времени выполнения также выше у SIFT.
