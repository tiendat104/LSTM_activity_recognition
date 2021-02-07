# LSTM_activity_recognition
# Abstract 
I implement the task of activity recognition from a video by using convolutional LSTM network. The reason i choose this network is because LSTM can learn the temporal dependencies between frames of a video.


# Data preparation
Please download file data.zip from this link: https://kaistackr-my.sharepoint.com/:u:/g/personal/tiendat_kaist_ac_kr/Edg0Rhj969xBobzYeZ8jxjMBTY0JPicKt5DUCzUzQByLaA?e=NsYHLa  and then unzip into this project. 

For the visualization of data structure, i have upload a folder name "data" inside this project. 
This is how i create this dataset: Create two subfolder "videos" and "imgs" inside folder "data". The folder "data/videos" contains files for videos from 5 classes ["ApplyEyeMakeup", "Archery", "Biking", "Fencing", "Fencing", "Kayaking"]. Then, i generate all frames for each video and save these frames into folder "data/imgs", and use a base model (InceptionV3) to extract features for each frame, and finally stack them together to create a file .npy that contains extracted information for a video. The .npy files are saved into folder "sequences", that would be used for training. 

The file "data/video_to_npy.py" is used to generate .npy files from videos. 

# Environment
- keras==2.3.1
- tensorflow==1.15.0

# Train and evaluation
To train, simply run the file train_conv_lstm.py, and to test, just simply run the file test_conv_lstm.py

