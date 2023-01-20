# Load Liraries
import numpy as np
import keras 
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import glob
import cv2


model_path1 = 'Brain_tumor_yes_or_not_predictor.h5'
model1 = keras.models.load_model(model_path1)

#horse image path
N1 = 'test/no 95.jpg'
N2 = 'test/no 96.jpg'
N3 = 'test/no 98.jpg'
N4 = 'test/no 99.jpg'

#human image path
Y1 = 'test/Y242.JPG'
Y2 = 'test/Y243.JPG'
Y3 = 'test/Y244.JPG'
Y4 = 'test/Y246.JPG'
def pred_yes_not(model, yes_or_no):
  test_image = image.load_img(yes_or_no, target_size = (150, 150))
  test_image = image.img_to_array(test_image)/255
  test_image = np.expand_dims(test_image, axis = 0)

  result = model.predict(test_image).round(3)

  pred = np.argmax(result)
  print(result, "--->>>", pred)

  if pred == 0:
    print('Predicted>>> No')
  else:
    print('Predicted>>> Yes')

for yes_or_no in [N1,N2,N3,N4, Y1,Y2,Y3,Y4]:
   print("Image Name :-", yes_or_no)
   pred_yes_not(model1, yes_or_no)

    



    
