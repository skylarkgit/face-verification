from keras.models import load_model
import mtcnn
import face
import os
from matplotlib import pyplot
import facenet.src.facenet as facenet

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load the model
model = load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)

image1 = face.extract_face("./Z.jpg")
image2 = face.extract_face("./k.jpg")
image3 = face.extract_face("./KK.jpg")
embedding1 = face.get_embedding(model, image1)
embedding2 = face.get_embedding(model, image2)
embedding3 = face.get_embedding(model, image3)
print("distance")
print("Z-k")
print(facenet.distance(embedding1, embedding2))
print("Z-KK")
print(facenet.distance(embedding1, embedding3))
print("k-KK")
print(facenet.distance(embedding2, embedding3))
print("DONE")