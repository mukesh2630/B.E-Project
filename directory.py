from scipy import signal
from scipy.io import wavfile
from scipy.misc import imsave
import os, glob

model_path = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/model/'
#convert everything to images
def to_spectrogram_and_save_png(filename):
    sample_rate, samples = wavfile.read(filename)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    filename=filename.replace("/audio/", "/png/")
    filename=filename.replace(".wav", ".png")
    imsave(filename, spectrogram)

paths = ["C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/train/audio/" + x + "/" for x in os.listdir("C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/train/audio")]

# filenames = []
# for path in paths:
#     os.makedirs(path.replace('/audio/', '/png/'))
#     filenames.extend([path + x for x in os.listdir(path)])
#
# for i in filenames:
#     to_spectrogram_and_save_png(i)

#map(to_spectrogram_and_save_png, filenames)

# #move 100 items per class to a validation set (10*30=3000 samples)
# list_to_move=[]
# for path in paths:
#     os.makedirs(path.replace('/audio/', '/png/').replace('/train/','/valid/'))
#     list_to_move.extend(glob.glob(path.replace("/audio/","/png/")+"*.png")[0:100])
#     #print(list_to_move)
#
# for f in list_to_move:
#     #print(f)
#     os.rename(f, f.replace("/train/", "/valid/"))

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    #width_shift_range=0.,
    #height_shift_range=0.,
    #zoom_range=0.01,
    rescale=1./255.,
    data_format='channels_last'
)

valid_datagen = ImageDataGenerator(
    rescale=1./255.,
    data_format='channels_last'
)

train_generator = train_datagen.flow_from_directory(
    "C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/train/png",
    target_size=(129,71),
    color_mode="grayscale",
    batch_size=64
)

valid_generator = valid_datagen.flow_from_directory(
    "C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/valid/png",
    target_size=(129,71),
    color_mode="grayscale",
    batch_size=64
)

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(129,71,1), data_format='channels_last'))
model.add(MaxPooling2D(data_format='channels_last'))
model.add(Conv2D(64, (3,3), activation='relu', data_format='channels_last'))
model.add(MaxPooling2D(data_format='channels_last'))
model.add(Conv2D(64, (3,3), activation='relu', data_format='channels_last'))
model.add(MaxPooling2D(data_format='channels_last'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(30, activation='softmax'))

model.compile(loss='categorical_crossentropy',
    optimizer='nadam',
    metrics=['accuracy']
)

print(model.summary())

callbacks = [TensorBoard(), EarlyStopping(monitor='val_loss', patience=10)]

history = model.fit_generator(
    train_generator,
    validation_data=valid_generator,
    validation_steps=40,
    steps_per_epoch=40,
    epochs=200,
    callbacks=callbacks
)
model.save(os.path.join(model_path, 'speech_commands.h5'))

