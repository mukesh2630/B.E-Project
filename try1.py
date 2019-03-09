import cv2
import os
import category_encoders as ce
import numpy as np 
import pandas as pd
import gc
import glob

legal_labels = 'yes no up down left right on off stop go silence unknown'.split()

output_path = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/output/'
model_path = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/model/'
test_path = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/test/test/audio/'
train_path = 'C:/Users/mukes/Desktop/TensorFlow Speech Recognition Challenge/picts/train/'

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        curr_fold = os.path.join(folder,filename)
        #images = [cv2.imread(i) for i in os.listdir(curr_fold)]
        for i in os.listdir(curr_fold):
            #print(i)
            labels.append(filename)
            file = os.path.join(curr_fold, i)
            img = cv2.imread(file)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            #break
    return labels, images

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))

labels, images = load_images_from_folder(train_path)
print('label is:', len(labels))
print('fnames is:', len(images))

y_train = []
x_train = []

for label, image in zip(labels, images):
        y_train.append(label)
        x_train.append(image)
x_train = np.array(x_train)
#print(x_train.shape)
#x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
#print('xtrain is:', x_train.shape)
y_train = label_transform(y_train)
#print(y_train.shape)
label_index = y_train.columns.values
#print(label_index.shape)
y_train = y_train.values
#print(y_train.shape)
y_train = np.array(y_train)
#print('y_train is:', y_train.shape)
#del labels, images
#print(gc.collect())

#le = ce.OneHotEncoder(return_df=True, impute_missing=False, handle_unknown="ignore")
#y = le.fit_transform(y)
#X = np.asarray(X)
#y = np.asarray(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

model = Sequential()
#input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
#this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, batch_size=128, validation_data=(X_test, y_test), epochs=3, shuffle=True, verbose=1)
#model.fit(X_train, y_train, batch_size=128, epochs=8, verbose=1)
#score = model.evaluate(X_test, y_test, batch_size=128, verbose=1)

model.save(os.path.join(model_path, 'VGG.model'))

# def test_data_generator(batch=16):
#     fpaths = glob(os.path.join(test_path, '*wav'))
#     #print(fpaths)
#     i = 0
#     for path in fpaths:
#         if i == 0:
#             imgs = []
#             fnames = []
#         i += 1
#         rate, samples = wavfile.read(path)
#         samples = pad_audio(samples)
#         resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
#         _, _, spectrogram = log_spectrogram(resampled, sample_rate=new_sample_rate)
#         imgs.append(spectrogram)
#         fnames.append(path.split('\\')[-1])
#         if i == batch:
#             i = 0
#             imgs = np.array(imgs)
#             imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
#             yield fnames, imgs
#     if i < batch:
#         imgs = np.array(imgs)
#         imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
#         yield fnames, imgs
#     raise StopIteration()
#
# #exit() #delete this
# del x_train, y_train
# gc.collect()
#
# index = []
# results = []
# for fnames, imgs in test_data_generator(batch=32):
#      predicts = model.predict(imgs)
#      predicts = np.argmax(predicts, axis=1)
#      predicts = [label_index[p] for p in predicts]
#      index.extend(fnames)
#      results.extend(predicts)
#
# df = pd.DataFrame(columns=['fname', 'label'])
# df['fname'] = index
# df['label'] = results
# df.to_csv(os.path.join(output_path, 'subVGGaudio.csv'), index=False)


def test_data_generator(batch=16):
    fpaths = glob.glob(test_path + '/*.png')
    #print(fpaths)
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        imgs.append(img)
        # print(imgs)
        fnames.append(path.split('\\')[-1].split('.png')[0])
        # print(fnames)
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            #imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
            yield fnames, imgs, globals()
    if i < batch:
        imgs = np.array(imgs)
        #imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
        yield fnames, imgs, locals()
    raise StopIteration()

fnames, imgs = test_data_generator(batch=32)
index = []
results = []
for fnames, imgs in test_data_generator(batch=32):
     predicts = model.predict(imgs)
     predicts = np.argmax(predicts, axis=1)
     predicts = [label_index[p] for p in predicts]
     index.extend(fnames)
     results.extend(predicts)

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
df.to_csv(os.path.join(output_path, 'subVGG.csv'), index=False)