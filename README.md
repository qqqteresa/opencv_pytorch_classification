This is a classifier which data derives from kaggle, below is the link：
https://www.kaggle.com/competitions/opencv-pytorch-dl-course-classification

1. It includes 16 kinds of Kenyan food images, and task is classified with it.
2. Use classify.py to divide data into a labeled train_set and an unlabeled test_set.
3. Split 20% of the train_set into a valid_set for model evaluation.
4. Gputest.py to test whether is working normally.
5. Use ResNet to train model and safe it in resnet50_model.pth.
6. Use this model to test valid_set to get loss and accuracy.
7. It works certainly and use this model to test test_set.
