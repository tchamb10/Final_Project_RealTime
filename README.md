# Final_Project_RealTime

Object detection and classification is a technique used to identify objects in images or videos that typically rely on deep learning to produce meaningful results. When humans look at images or videos, we can recognize and locate objects of interest within a matter of moments. The goal of object detection is to replicate this intelligence in humans. There is an infinite number of applications that one can apply to object classifications algorithms that have the potential to increase accessibility, enhance automation, and improve human interactivity. An example where deep learning can improve quality of life is easing translation of sign languages to better communicate with others.

Sign language is a language that uses hand gestures and facial expressions to communicate. These signs mean different things to different people, so it can be difficult to know what the signer is saying or what type of sign language they are speaking. It is commonly used by people who are deaf or have a hearing impairment. This project will strictly focus on the American Sign Language alphabet as ASL is one of the most popular sign languages spoken.

## ASL_YoloV4.ipynb

The notebook was ran on Google Colab making use of GPU's. The code was provided from two main sources, [RoboFlow](https://roboflow.com/), and [AlexeyAB DarkNet](https://github.com/AlexeyAB/darknet). The custom dataset was imported using private code provided from RoboFlow. The dataset consisted of .jpg image files and .txt files. The .txt files are required for each image as they contain information such as class label, and bounding box coordinates for each image. The following [link](https://drive.google.com/drive/folders/1ne3EoMYwUbH3jSYPkXcpb9VCz8EGET3v?usp=sharing) contains results for the trained YOLO-V4 model on the ASL dataset.

## ASL_ResNet_18.ipynb

The notebook was ran on Google Colab making use of GPU's. The untrained ResNet-18 model was imported via torch.models(). The dataset loaded consists of images separated in folders based on label names. The images then needed to be normalized using the mean and standard deviation of the entire dataset. The results came out to 99% validation accuracy, and 97% test accuracy. A confusion matrix was created to give a better idea of what letters were making up this 1%. The total training time took around 1 hour and 15 minutes.

## ASL_DenseNet_161.ipynb

The notebook was ran on Google Colab making use of GPU's. The untrained DenseNet-161 model was imported via torch.models(). The dataset loaded consists of images separated in folders based on label names. The images then needed to be normalized using the mean and standard deviation of the entire dataset. The results came out to 99% validation accuracy, and 98% test accuracy. A confusion matrix was created to give a better idea of what letters were making up this 1%. The total training time took around 3 hours and 30 minutes. Multiple reasons for longer training time include, the increase in epochs to achieve similar accuracy as ResNet, as well as the increase in computational requirements from DenseNet.
