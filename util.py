import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_in_images():
    print("-----------------")
    print("Reading in images")
    folders = ["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", 
               "Sea Bass", "Shrimp", "Striped Red Mullet", "Trout"]

    img_arr = []
    img_labels = []
    for folder in folders:
        img_rel_folder = "Fish_Dataset/Fish_Dataset/" + folder + '/' + folder + '/'
        for i in range(1, 1000+1):
            str_i = str(i)
            img_name = img_rel_folder + '0'*(5-len(str_i)) + str_i + '.png'
            img = cv2.imread(img_name)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_arr.append(img)
            img_labels.append(folder)

    print("-----------------")
    return img_arr, img_labels

def plot_images(images, cls_true, cls_pred=None, title=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    plt.rcdefaults()
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors(cls_pred, true_cls, test_imgs):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = np.array([pred != true for pred, true in zip(cls_pred, true_cls)])
    test_imgs = np.array(test_imgs)
    cls_pred = np.array(cls_pred)
    true_cls = np.array(true_cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = test_imgs[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = true_cls[incorrect]
    
    num_incorrect = cls_pred.shape[0]
    num_correct = true_cls.shape[0] - num_incorrect
    
    print(f"{num_correct} Correct\n{num_incorrect} Incorrect")
    return num_correct, num_incorrect
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])