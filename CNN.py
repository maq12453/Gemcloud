import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
INIT_LR = 0.0001
EPOCHS = 12
stepPerEpochs = 57

training_set = pd.read_csv("2nd layer/train I.csv")

training_imgs = ["{}".format(x) for x in list(training_set.train)]
print(len(training_imgs))

training_labels_1 = list(training_set['labels'])
print(len(training_labels_1))

training_set = pd.DataFrame({'image_path': training_imgs, 'label': training_labels_1})
training_set.label = training_set.label.astype(str)

# loading training data
train_dataGen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_dataGen.flow_from_dataframe(
    dataframe=training_set,
    validate_filenames=False,
    directory="", x_col="image_path",
    y_col="label",
    class_mode="categorical",
    target_size=(200, 200),
    batch_size=32)

# loading testing data
testing_set = pd.read_csv("2nd layer/test I.csv")

testing_imgs = ["{}".format(x) for x in list(testing_set.test)]
print(len(testing_imgs))

testing_labels_1 = list(testing_set['labels'])
print(len(testing_labels_1))

testing_set = pd.DataFrame({'Images': testing_imgs, 'found': testing_labels_1})
testing_set.found = testing_set.found.astype(str)

test_dataGen = ImageDataGenerator(rescale=1. / 255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_generator = test_dataGen.flow_from_dataframe(
    dataframe=testing_set,
    validate_filenames=False,
    directory="", x_col="Images",
    y_col="found",
    class_mode="categorical",
    target_size=(200, 200),
    batch_size=32)

# load the VGG16 network, ensuring the head FC layer sets are left
baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(200, 200, 3)))
baseModel.summary()
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
BatchNormalization()(headModel)
headModel = Dense(16, activation="relu")(headModel)
# BatchNormalization()(headModel)
# headModel = Dropout(0.2)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
history = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS, steps_per_epoch=stepPerEpochs,
                    validation_steps=14)

model_json = model.to_json()
with open("2nd layer/VGG.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("2nd layer/VGG.h5")
model.save("VGG")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("2nd layer/VGGacc.png")
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("2nd layer/VGGloss.png")
plt.show()
