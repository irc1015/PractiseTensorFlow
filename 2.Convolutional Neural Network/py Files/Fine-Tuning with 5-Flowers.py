import tensorflow as tf
import numpy as np
import os,math,re,sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
from keras import backend
import matplotlib.pylab as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
AUTO = tf.data.experimental.AUTOTUNE
print("Tensorflow version " + tf.__version__)

IMG_SIZE = [224,224]
IMG_CHANNELS = 3
BATCH_SIZE = 10

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3) #JPEG --> Tensor
    #image = tf.reshape(image, [*IMG_SIZE, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return image

def read_tfrecord(example):
    image_feature_dict = {
        'image':tf.io.FixedLenFeature([],tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, image_feature_dict) #return a dict
    image = decode_image(example['image'])
    label = tf.cast(example['label'], tf.int64)
    return image,label   #return a tuple

def load_dataset(filenames, ordered = False):
    ignore_order = tf.data.Options()

    if not ordered:
        ignore_order.deterministic = False # experimental_deterministic has been baned

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def get_training_dataset():
    dataset = load_dataset('Training_Set.tfrecord') #ordered default is False
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) #Prepare later elements
    return dataset

def get_validation_dataset(ordered = False):
    dataset = load_dataset('Evaluation_Set.tfrecord', ordered = ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache() #cache files in memory
    dataset = dataset.prefetch(AUTO)
    return dataset

print("Training data shapes:")
for image, label in get_training_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training label examples:", label.numpy())
labelarray = label.numpy()
print(labelarray[0],type(labelarray[0]))

CLASSES = ['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips']

training_dataset = get_training_dataset()
training_dataset = training_dataset.unbatch().batch(20) #20 images, 20 labels in every batch
train_batch = iter(training_dataset)

for image, label in train_batch:
    print(image.numpy().shape, label.numpy().shape)
    print(type(image.numpy()), type(label.numpy()))
    print(label.numpy())
    break

strategy = tf.distribute.MirroredStrategy(devices=None) #Use all available GPUs or CPU
print("REPLICAS:", strategy.num_replicas_in_sync)

LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_START = 0.00001
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 3.0
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  '''def __call__(self, step):
      if step < LR_RAMPUP_EPOCHS:
          self.initial_learning_rate = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * step + LR_START
      elif step < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
          self.initial_learning_rate = LR_MAX
      else:
          self.initial_learning_rate = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(step - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
      return self.initial_learning_rate'''


  def __call__(self, step):
      current_epoch = step // 330
      print('current epoch is {}, current step is {}'.format(current_epoch, step))

      false_situation = tf.cond(pred= current_epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS,
                       true_fn= lambda: LR_MAX,
                       false_fn= lambda: (LR_MAX - LR_MIN) * LR_EXP_DECAY**(current_epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN )

      self.initial_learning_rate = tf.cond(pred=(current_epoch < LR_RAMPUP_EPOCHS),
                     true_fn= lambda: (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * current_epoch + LR_START,
                     false_fn= lambda: false_situation)
      return self.initial_learning_rate



from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.optimizers import MultiOptimizer

with strategy.scope():
    pretrained_model = tf.keras.applications.MobileNetV2(
        weights = 'imagenet',
        include_top = False,
        input_shape = [*IMG_SIZE,3],
    )
    pretrained_model.trainable = True #Open Fine-Tuning

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Lambda(lambda data: tf.keras.applications.mobilenet.preprocess_input(
                tf.cast(data, tf.float32)), input_shape = [*IMG_SIZE, 3]),
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(16, activation='relu', name='flower_dense'),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax', name='flower_prob')
        ]
    )


    layer_wise_optimizer = {
            # Clasification head
            'flower_prob': AdamW(learning_rate=LR_MAX, weight_decay=0.0, name = 'flower_prob'),
            #AdamW has got 'learning_rate', 'lr' arguments.
            'flower_dense': AdamW(learning_rate=LR_MAX, weight_decay=0.0, name = 'flower_dense'),
            # Pretrained layers
            'block_1_': AdamW(learning_rate=LR_MAX * 0.008, weight_decay=0.0, name = 'b1'),
            'block_2_': AdamW(learning_rate=LR_MAX * 0.016, weight_decay=0.0, name = 'b2'),
            'block_3_': AdamW(learning_rate=LR_MAX * 0.024, weight_decay=0.0, name = 'b3'),
            'block_4_': AdamW(learning_rate=LR_MAX * 0.032, weight_decay=0.0, name = 'b4'),
            'block_5_': AdamW(learning_rate=LR_MAX * 0.04, weight_decay=0.0, name = 'b5'),
            'block_6_': AdamW(learning_rate=LR_MAX * 0.06, weight_decay=0.0, name = 'b6'),
            'block_7_': AdamW(learning_rate=LR_MAX * 0.08, weight_decay=0.0, name = 'b7'),
            'block_8_': AdamW(learning_rate=LR_MAX * 0.10, weight_decay=0.0, name = 'b8'),
            'block_9_': AdamW(learning_rate=LR_MAX * 0.12, weight_decay=0.0, name = 'b9'),
            'block_10_': AdamW(learning_rate=LR_MAX * 0.14, weight_decay=0.0, name = 'b10'),
            'block_11_': AdamW(learning_rate=LR_MAX * 0.16, weight_decay=0.0, name = 'b11'),
            'block_12_': AdamW(learning_rate=LR_MAX * 0.20, weight_decay=0.0, name = 'b12'),
            'block_13_': AdamW(learning_rate=LR_MAX * 0.24, weight_decay=0.0, name = 'b13'),
            'block_14_': AdamW(learning_rate=LR_MAX * 0.28, weight_decay=0.0, name = 'b14'),
            'block_15_': AdamW(learning_rate=LR_MAX * 0.32, weight_decay=0.0, name = 'b15'),
            'block_16_': AdamW(learning_rate=LR_MAX * 0.36, weight_decay=0.0, name = 'b16'),
            # these layers do not have stable identifiers in tf.keras.applications.MobileNetV2
            'conv': AdamW(learning_rate=LR_MAX * 0.2, weight_decay=0.0, name = 'c1'),
            'Conv': AdamW(learning_rate=LR_MAX * 0.2, weight_decay=0.0, name = 'c2')
        }

    optimiers_and_layers = [
        (layer_wise_optimizer['flower_dense'], model.layers[3]),
        (layer_wise_optimizer['flower_prob'], model.layers[4]),
        (layer_wise_optimizer['Conv'], pretrained_model.layers[1:3]),
        (layer_wise_optimizer['conv'], pretrained_model.layers[4:9]),
        (layer_wise_optimizer['block_1_'], pretrained_model.layers[9:18]),
        (layer_wise_optimizer['block_2_'], pretrained_model.layers[18:27]),
        (layer_wise_optimizer['block_3_'], pretrained_model.layers[27:36]),
        (layer_wise_optimizer['block_4_'], pretrained_model.layers[36:45]),
        (layer_wise_optimizer['block_5_'], pretrained_model.layers[45:54]),
        (layer_wise_optimizer['block_6_'], pretrained_model.layers[54:63]),
        (layer_wise_optimizer['block_7_'], pretrained_model.layers[63:72]),
        (layer_wise_optimizer['block_8_'], pretrained_model.layers[72:81]),
        (layer_wise_optimizer['block_9_'], pretrained_model.layers[81:90]),
        (layer_wise_optimizer['block_10_'], pretrained_model.layers[90:98]),
        (layer_wise_optimizer['block_11_'], pretrained_model.layers[98:107]),
        (layer_wise_optimizer['block_12_'], pretrained_model.layers[107:116]),
        (layer_wise_optimizer['block_13_'], pretrained_model.layers[116:125]),
        (layer_wise_optimizer['block_14_'], pretrained_model.layers[125:134]),
        (layer_wise_optimizer['block_15_'], pretrained_model.layers[134:143]),
        (layer_wise_optimizer['block_16_'], pretrained_model.layers[143:151]),
        (layer_wise_optimizer['Conv'], pretrained_model.layers[151:])]

    optimizer = MultiOptimizer(optimiers_and_layers)

NUM_TRAINING_IMG = 3303
STEPS_PER_EPOCH = NUM_TRAINING_IMG // BATCH_SIZE
EPOCHS = 5
NUM_VALIDATION_IMG = 367
VALIDATION_STEPS = -(-NUM_VALIDATION_IMG // BATCH_SIZE) #The "-(- // )" trick rounds up instead of down

model.compile(
    #optimizer='adam',
    optimizer= optimizer,
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
    steps_per_execution=8
)
model.summary()


def scheduler_epoch(epoch, learning_rate):
    if epoch < 3:
        learning_rate = learning_rate + (epoch * 0.01)
    else:
        learning_rate = learning_rate - (epoch * 0.01 * 0.4)
    return learning_rate


class LearningRate_with_MultiOptimizer(tf.keras.callbacks.Callback):
    def __init__(self, schedule, verbose=0):  # 0:quiet  1:update message
        super(LearningRate_with_MultiOptimizer, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        for item in self.model.optimizer.optimizer_specs:
            learning_rate = float(tf.keras.backend.get_value(item['optimizer'].learning_rate))
            learning_rate = self.schedule(epoch, learning_rate)
            tf.keras.backend.set_value(item['optimizer'].learning_rate, tf.keras.backend.get_value(learning_rate))

            if self.verbose > 0:
                print('\nEpoch {}, learning_rate {}'.format((epoch + 1), learning_rate))


lr_callback = LearningRate_with_MultiOptimizer(scheduler_epoch, verbose=1)

history = model.fit(get_training_dataset(),
                    steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
                    validation_data=get_validation_dataset(), validation_steps=VALIDATION_STEPS,
                    callbacks = [lr_callback])


cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.
images_ds = cmdataset.map(lambda image, label: image) #Make (image, label)tuple --> image
labels_ds = cmdataset.map(lambda image, label: label).unbatch() #Make (image, label)tuple --> label
cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMG))).numpy() # get every lable of Validation DataSet  as one batch
cm_probabilities = model.predict(images_ds, steps=VALIDATION_STEPS)
cm_predictions = np.argmax(cm_probabilities, axis=-1)
print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
print("Predicted labels: ", cm_predictions.shape, cm_predictions)

pretrained_weight = []