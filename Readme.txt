This is a programm to detect which kind of hand gesture you are perform using Convolution Neural Network.

#HOW TO RUN THE PROGRAMM
1. Make a file dataset and inside it is files contain your gesture images
2. Make a file named TrainedModel to use for Neural Network
3. Run makeDataset.py to make your own Dataset for training
4. Run ResizeImages.py to resize all of your images to 89 x 100 dimension images so it can be suitable for Convolution 
Neural Network
5. Run TrainModel.py to retrain the model using your custom Dataset
6. Run predictGestures.py to predict which kind of gestures you are perform in realtime 
