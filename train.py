#Import Packages
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from deeplab import DeepLabV3Plus



batch_size = 6  #Define Batch Size
H, W = 512, 512 #Crop Size
num_classes = 26 #labels 

#Training Images and Mask
image_list = sorted(glob('dataset/leftImg8bit/train/*/*'))
mask_list = sorted(glob('dataset/gtFine_only_level3Id/train/*/*'))

#Validation Images and Mask
val_image_list = sorted(glob('dataset/leftImg8bit/val/*/*'))
val_mask_list = sorted(glob('gtFine_only_level3Id/val/**/*'))

#Number of Training and Validation Images
print('Found', len(image_list), 'training images')
print('Found', len(val_image_list), 'validation images')

#Checking the Image number with the mask number
for i in range(len(image_list)):
    assert image_list[i].split('/')[-1].split('_leftImg8bit')[0] == mask_list[i].split('/')[-1].split('_gtFine_labellevel3Ids')[0]

for i in range(len(val_image_list)):
    assert val_image_list[i].split('/')[-1].split('_leftImg8bit')[0] == val_mask_list[i].split('/')[-1].split('_gtFine_labellevel3Ids')[0]

#Preprocess (-1,1)
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

#Image and Mask Loader
def get_image(image_path, img_height=1080, img_width=1920, mask=False, flip=0):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])
        #img = tf.image.random_brightness(img, max_delta=50.)
        #img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        #img = tf.image.random_hue(img, max_delta=0.2)
        #img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.clip_by_value(img, 0, 255)
        #img = tf.case([
        #    (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        #], default=lambda: img)
        img = preprocess_input(img)
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
                      img_height, img_width]), dtype=tf.float32)
        img = tf.clip_by_value(img, 0, 25)    
        #img = tf.case([
         #   (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
        #], default=lambda: img)
    return img

#Random Crop 512*512
def random_crop(image, mask, H=512, W=512):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=H,
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=H,
                                         target_width=W)
    return image, mask

#load data 
def load_data(image_path, mask_path, H=512, W=512):
    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip), get_image(
        mask_path, mask=True, flip=flip)
    image, mask = random_crop(image, mask, H=H, W=W)
    return image, mask

#train data loader
train_dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
train_dataset = train_dataset.shuffle(buffer_size=128)
train_dataset = train_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print(train_dataset)

#val data loader
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_list,
                                                  val_mask_list))
val_dataset = val_dataset.apply(
    tf.data.experimental.map_and_batch(map_func=load_data,
                                       batch_size=batch_size,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       drop_remainder=True))
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

#training strategy
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DeepLabV3Plus(H, W, num_classes)    
    model.load_weights('/gdrive/My Drive/Shared with Shubham_Deep Learning/xception_xception_ds16block14/top_weights.h5')
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)                                                            
    model.compile(loss=loss, 
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
                  metrics=[tf.keras.metrics.MeanIoU(num_classes=26)])


tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(mode='min', filepath='top_weights.h5',
                      monitor='val_loss',
                      save_best_only='True',
                      save_weights_only='True', verbose=1)
callbacks = [mc, tb]

model.summary()

#Training the Model
model.fit(train_dataset,
          steps_per_epoch=len(image_list)/batch_size ,
          epochs=40,
          shuffle=True,
          validation_data=val_dataset,
          validation_steps=len(val_image_list)/batch_size,
          callbacks=callbacks)



