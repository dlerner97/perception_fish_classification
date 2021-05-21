import os
import sys
import numpy as np
from util import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

num_models = 9
models = [str(num) for num in range(1, num_models)]
if len(sys.argv) > 1:
    models = [sys.argv[1:]]

img_arr, img_labels = read_in_images()
train_imgs, test_imgs, train_labels, test_labels = train_test_split(img_arr, img_labels, test_size = .2, random_state = 42)

from sklearn.preprocessing import LabelBinarizer
label_bin = LabelBinarizer().fit(train_labels)
y_test = label_bin.transform(test_labels)

correct_info = []

os.system('cls')
for model_num in models:
    model = tf.keras.models.load_model('model' + str(model_num) + '/model' + str(model_num))

    pred = np.argmax(model.predict(np.array(test_imgs)), axis=-1)

    print(5*"=", model_num, 5*"=")
    result = model.evaluate(x=np.array(test_imgs), y=y_test)
    pred_labels = [label_bin.classes_[i] for i in pred]
    correct_info.append(plot_example_errors(pred_labels, test_labels, test_imgs))
    print(13*"=")

X = np.arange(len(correct_info))
correct_info = np.array(correct_info)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, correct_info[:,0], color='g', width=.25)
ax.bar(X + 0.25, correct_info[:,1], color='r', width=.25)
plt.title("Correct Vs. Incorrect Test Predictions")
plt.legend(['Correct Predictions', "Incorrect Predictions"])
plt.show()
