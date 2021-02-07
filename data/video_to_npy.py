
import os
import glob
from keras.preprocessing import image
from extractor import Extractor
import cv2
import numpy as np
class information():
    def __init__(self):
        self.seq_length = 40
        self.list_classes = ['ApplyEyeMakeup', 'Archery','Biking','Fencing', 'Kayaking']
        self.parent_videos_path = "videos"
        self.parent_img_path = 'imgs'

    def create_imgs_from_videos(self):
        type_video_names = os.listdir(self.parent_videos_path)
        for type in type_video_names:
            list_videos = os.listdir(self.parent_videos_path + "/" + type)
            num_videos = len(list_videos)
            tmp_path = []
            num_train_videos = int(num_videos * 0.7)
            num_validation_videos = int(num_videos * 0.2)
            num_test_videos = num_videos - num_train_videos - num_validation_videos
            for i in range(num_train_videos):
                tmp_path.append("/train/")
            for i in range(num_validation_videos):
                tmp_path.append("/validation/")
            for i in range(num_test_videos):
                tmp_path.append("/test/")

            # for video_name in list_videos:
            for i in range(len(list_videos)):
                type_img_path = self.parent_img_path + tmp_path[i] + type
                video_name = list_videos[i]
                video_name_no_ext = video_name.split(".")[0]
                new_img_dir = type_img_path + "/" + video_name_no_ext
                os.makedirs(new_img_dir)

                cap = cv2.VideoCapture(self.parent_videos_path + "/" + type + "/" + video_name)
                index = 0
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        try:
                            cv2.imwrite(new_img_dir + "/" + str(index) + ".jpg", frame)
                            index += 1
                        except:
                            pass
                    else:
                        break
    def create_npy(self, sequence_path = "sequences"):
        base_model = Extractor()
        tmp = ['train', 'validation', 'test']
        for folder in tmp:
            for class_name in self.list_classes:
                os.makedirs(os.path.join(sequence_path, folder, class_name), exist_ok=True)

        for folder in tmp:
            for class_name in self.list_classes:
                list_video_name = os.listdir(os.path.join(self.parent_img_path, folder, class_name))
                for video_name in list_video_name:
                    target_npy_path = os.path.join(sequence_path, folder, class_name, video_name + ".npy")
                    video_frames_path = os.path.join(self.parent_img_path, folder, class_name, video_name)
                    frame_names = sorted(os.listdir(video_frames_path), key= lambda x : int(x.split('.')[0]) )
                    frame_path_list = []
                    for name in frame_names:
                        frame_path = os.path.join(video_frames_path, name)
                        frame_path_list.append(frame_path)
                    rescaled_frame_path_list = rescale_list(frame_path_list, self.seq_length)

                    sequence = []
                    for frame_path in rescaled_frame_path_list:
                        features = base_model.extract(frame_path)
                        # features:  [0.1400126  0.43671897 0.01164552 ... 0.09836353 0.43844804 0.7130548 ]
                        sequence.append(features)
                    np.save(target_npy_path, sequence)


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

a = information()
a.create_imgs_from_videos()
a.create_npy()


