
import os
import glob
import cv2
import numpy as np
from models import conv_LSTM, conv_LSTM2
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

list_classes = ['ApplyEyeMakeup', 'Archery','Biking','Fencing', 'Kayaking']

def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
            if we want a list of size 5 and we have a list of size 25, return a new
            list of size five which is every 5th element of the origina list."""
    assert len(input_list) >= size
    # Get the number to skip between iterations.
    skip = len(input_list) // size
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    # Cut off the last one if needed.
    return output[:size]

def process_image(image, target_shape):
    from keras.preprocessing.image import img_to_array, load_img
    h,w,_ = target_shape
    image = load_img(image, target_size = target_shape)
    img_arr = img_to_array(image)
    x = ( (img_arr-127.5)/127.5 ).astype(np.float32)
    return x

def load_data(train_test_val = 'train', seq_length = 40, target_shape_img = (160,160,3)):
    print("target_shape_img : ", target_shape_img)
    X = []
    y = []
    data_path = os.path.join('data/imgs', train_test_val)
    for class_name in list_classes:
        video_names = os.listdir(os.path.join(data_path, class_name))
        for vid_name in video_names:
            frame_names = os.listdir(os.path.join(data_path, class_name, vid_name))
            frame_names = sorted(frame_names, key= lambda x : int(x.split('.')[0]))
            list_frames_path = []
            for name in frame_names:
                path_frame = os.path.join(data_path, class_name, vid_name, name)
                list_frames_path.append(path_frame)
            list_frames_path = rescale_list(list_frames_path, seq_length)
            processed_frames = []
            for frame_path in list_frames_path:
                img_arr = process_image(frame_path, target_shape=target_shape_img)
                processed_frames.append(img_arr)
            X.append(processed_frames)
            y.append(to_categorical(list_classes.index(class_name), len(list_classes)))
    return np.array(X), np.array(y)

def train(seq_length = 40, num_classes = 5, batch_size = 32, epochs = 1000):

    os.makedirs('checkpoint/conv_lstm', exist_ok=True)
    current_checkpoint_subdir = os.listdir('checkpoint/conv_lstm')
    new_checkpoint_subdir = os.path.join("checkpoint/conv_lstm", str(len(current_checkpoint_subdir) + 1))
    os.makedirs(new_checkpoint_subdir, exist_ok=False)

    os.makedirs("logs/conv_lstm", exist_ok=True)
    current_log_subdir = os.listdir("logs/conv_lstm")
    new_log_subdir = os.path.join("logs/conv_lstm", str(len(current_log_subdir) + 1))
    os.makedirs(new_log_subdir, exist_ok=False)

    tensorboard = TensorBoard(log_dir=new_log_subdir)
    early_stopper = EarlyStopping(patience=20)
    checkpointer = ModelCheckpoint(filepath=os.path.join(new_checkpoint_subdir, ".{epoch:03d}-{val_loss:.3f}.hdf5"),
                                   verbose=1, save_best_only=True)

    # load data
    trainX, trainy = load_data(train_test_val='train', target_shape_img = (80,80,3))
    valX, valy = load_data('validation', target_shape_img = (80,80,3))

    #model = conv_LSTM(seq_length= seq_length, num_classes=num_classes, input_image_shape=(80,80,3))
    model = conv_LSTM2(seq_length=seq_length, num_classes=num_classes,input_image_shape=(80,80,3))
    optimizer = Adam(lr=1e-6, decay=1e-6)
    model.compile(loss = 'categorical_crossentropy', optimizer= optimizer, metrics = ['accuracy'])

    model.fit(trainX, trainy, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(valX, valy),
              callbacks=[tensorboard, early_stopper, checkpointer])

if __name__ == "__main__":
    batch_size = 16
    train(batch_size= batch_size)


