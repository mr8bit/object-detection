# Learning TensorFlow models for object detection  

Для начала скопируйте этот репозиторий 

```bash
git clone ....
```
Выполните установку:
```bash
pip install tensoflow==1.5.0
pip install pandas
pip install pillow
```
Cкопируйте 

```bash
git clone https://github.com/tensorflow/models.git
```
Выполните [установку Object Detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) 

Для обучения необходимо: 
1.Раскинуть dataset в папки 'images/train' - выборки для обучения и 'images/test' - выборки для проверки нейросети

2.Обозначить области для обучения нейронной сети с помощью [LabelImg](https://github.com/tzutalin/labelImg) 

3.Преобразовать XML формат в CSV при помощи `xml_to_csv.py`

```bash
python xml_to_csv.py
``` 
4.Преобразовать в формат для обучения нейросети tfRecord

```bash
  # Создать dataset для обучения:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record --type=train
  # Создать dataset для тестов:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record --type=test

```
Если объектов для обнаружения много (2 и более) необходимо внести изменения в файл `generate_tfrecord.py`
```python
#generate_tfrecord.py:30
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == '1_object':
        return 1
    if tow_label == '2_object':
    	return 2
    # ..... etc.
    else:
        None

```

5.Скачать [нейросеть для обучения](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models)  
Файл `.config` перенести в папку `training`, в ней так же создать  файл `object-detection.pbtxt`
```bash
#object-detection.pbtxt
item {
  id: 1
  name: '1_object'
}
```
Где `name`: название объекта для обнаружения, а `id` его номер. Если объектов для поиска много, необходимо добавить название всех объектов, в соответстви  с `generate_tfrecord.py:30`

Ex.

```bash
#object-detection.pbtxt
item {
  id: 1
  name: '1_object'
}
item {
  id: 2
  name: '3_object'
}
```

6.Перенесите репоситорий в tensoflow-models/research/object-detection

```bash
sudo mv ../* tensoflow-models/research/object-detection
```

7.Начать обучение 
```bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/MODEL_NAME.config
```
8.Выполнить экспорт обученной модели: 
```bash
python export_inference_graph.py     --input_type=image_tensor     --pipeline_config_path=training/ssd_mobilenet_v1_pets.config     --trained_checkpoint_prefix=training/model.ckpt-1673     --output_directory=car_graph
```


