import numpy as np
import os
import sys

from pathlib import Path
import hashlib

import logging

import cv2
import cognitive_face as CF

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
from keras.applications import VGG19
from keras_vggface.vggface import VGGFace

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pickle
from tensorboard.plugins.beholder.shared_config import IMAGE_WIDTH

#Loads the Python face detection classifiers from their local serialized forms
cnn_face_classifier = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
classifier_uri = "./Face_cascade.xml"
cascade_face_classifier = cv2.CascadeClassifier(classifier_uri)
BASE_CS_URL = 'https://virginia.api.cognitive.microsoft.us/face/v1.0/'  # Replace with your regional Base URL

img_width = 300
img_height = 300

#Acquire vgg face and general models and weights with the top dense layers removed, tuned for the appropriate image size
vgg_19_feature_gen = VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
vgg_face_feature_gen = VGGFace(include_top=False, input_shape=(img_width, img_height, 3), pooling='avg') # pooling: None, avg or max


def detectExtractFace(image_name, image_bytes, cs_account_key):
    #Run the Cascade face detector
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = cascade_face_classifier.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0) 
    
    #Run the CNN face detector
    (h, w) = image.shape[:2] #Save off original image dimensions
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
    cnn_face_classifier.setInput(blob)
    detections = cnn_face_classifier.forward()
    element = 1
    confidence = detections[0,0,element,2]
    if confidence > 0.95:
        detections = 1
    else:
        detections = 0
    
    #Run the Cognition Services Face Recognition
    CF.Key.set(cs_account_key)
    CF.BaseUrl.set(BASE_CS_URL)
    cs_faces = CF.face.detect(image_bytes)


    logging.debug("Detection results for {0} are {1} {2} {3}".format(image_name, len(faces), detections, len(cs_faces)))
    return (image_name, len(faces), detections, len(cs_faces))
    '''
    #If Cascade identifies at least one face, use the cascade results to extract the faces
    roi_images = []
    federatedResult = 0
    if len(faces) > 0:
        federated_result = 2
        for (x,y,w,h) in faces:
            roi_image = image[y:(y+h), x:(x+w)]
            roi_images.append(roi_image)
    #Otherwise
    else:
        #Grab only first element from the CNN classifier
        element = 1
        confidence = detections[0,0,element,2]
        if confidence > 0.95:
            federatedResult = 1
            box = detections[0, 0, element, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            roi_image = image[startY, endY, startX, endX]
            roi_images.append(roi_image)
            
    return roi_images
    '''
def imgPreprocessing(image_file):
    nparr = np.fromstring(image_file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    decoded = cv2.resize(image, (img_width, img_height))
    decoded = decoded.reshape((1,img_width,img_height,3))
    #logging.debug("Converted image from {0} to {1}".format(image.shape, decoded.shape))
    return decoded

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #FORMAT = '%(asctime) %(message)s'
    #logging.basicConfig(format=FORMAT)

    cs_account_name = os.environ['COG_SERV_ACCOUNT']
    cs_account_key = os.environ['COG_SERV_KEY']

    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 1
    sample_size = 1


    def generateTransferFeatures(source_dir):
        face_labels = []
        face_features = []
        image_labels = []
        image_features = []
        decoded_input = []
        for dir_path, dir_names, file_names in os.walk(source_dir, topdown=True):
            for file_name in file_names:
                dir_components = Path(dir_path)
                usage = dir_components.parts[len(dir_components.parts) - 1]
                entity = dir_components.parts[len(dir_components.parts) - 2]

                image_file = open(os.path.join(dir_path, file_name), 'rb').read()
                hash = hashlib.md5(image_file).hexdigest()    
                
                #detectExtractFace(file_name, image_file, cs_account_key)
                
                nparr = np.fromstring(image_file, np.uint8)
                #decoded_input.append(inputs_batch)

                
                
                image_features.append(list(vgg_19_features_gen.predict(inputs_batch, batch_size=1).flat))
                image_labels.append(1 if entity == "Train" else 0)
                face_features.append(list(vgg_face_feature_gen.predict(inputs_batch, batch_size=1).flat))
                face_labels.append(1 if entity == "Train" else 0)
        print("Generated features {0} {1} {2} {3} {4} {5}".format(len(image_features), len(image_features[0]), 
                                                            len(image_labels), len(face_features), 
                                                            len(face_features[0]), len(face_labels)))
        return (image_features, image_labels, face_features, face_labels)
    
    def faceLoader(source_dir):
        #Runs through test images in a standardized file structure with the usage (Train, Validation) or Entity (2+ class labels)
        general_images = []
        general_features = []
        general_labels = []
        
        face_images = []
        face_iso_images = []
        face_iso_features = []
        face_features = []
        face_labels = []
        for dir_path, dir_names, file_names in os.walk(source_dir, topdown=True):
            for file_name in file_names:
                dir_components = Path(dir_path)
                usage = dir_components.parts[len(dir_components.parts) - 2]
                entity = dir_components.parts[len(dir_components.parts) - 1]
                logging.debug(entity)
                logging.debug(usage)
                
                if usage == "Train":
                    image_file = open(os.path.join(dir_path, file_name), 'rb').read()
                    hash = hashlib.md5(image_file).hexdigest()    

                    #Ensure that a single face exists in this set
                    face_list = CF.face.detect(image_file)
                    print(face_list)
                    if len(face_list) == 1:
                        #stuff = CF.person.add_face(image_file, group_name, person_id)
                        face_images.append(image_file)
                        inputs_batch = imgPreprocessing(image_file)
                        face_features.append(vgg_face_feature_gen.predict(inputs_batch, batch_size=1).flatten())
                        face_labels.append(entity)
                        
                        face_rectangle = face_list[0]['faceRectangle']
                        nparr = np.fromstring(image_file, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
                        face_iso_image = image[face_rectangle['top']:face_rectangle['top'] + face_rectangle['height'], face_rectangle['left']:face_rectangle['left'] + face_rectangle['width']]
                        face_iso_images.append(face_iso_image)
                        #cv2.imwrite(os.path.join(".", hash + ".jpg"), face_iso_image)
                        
                        #
                        decoded = cv2.resize(face_iso_image, (img_width,img_height))
                        decoded = decoded.reshape((1,img_width,img_height,3))
                        face_iso_features.append(vgg_face_feature_gen.predict(decoded, batch_size=1).flatten())
                    else:
                        open(os.path.join(".", hash + ".jpg"), 'wb').write(image_file)
                        
                    inputs_batch = imgPreprocessing(image_file)
                    general_images.append(image_file)
                    general_features.append(vgg_19_feature_gen.predict(inputs_batch, batch_size=1).flatten())
                    general_labels.append(entity)
                
        return (general_images, general_features, general_labels, face_images, face_features, face_iso_image, face_iso_features, face_labels)

    def stageCognitiveServiceFaceRecognition(source_dir):
        '''
        
        '''
        
        KEY = 'd7bb7b429dce4441a8f39b9f92088950'  # Replace with a valid subscription key (keeping the quotes in place).
        CF.Key.set(KEY)

        BASE_URL = 'https://eastus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
        CF.BaseUrl.set(BASE_URL)
        
        
        #CF.person_group.
        
        #Checks to see if a "default" person group has been created, and creates one of not
        group_name = 'default'
        if group_name not in map(lambda x: x['personGroupId'], CF.person_group.lists()) :
            CF.person_group.create(group_name, group_name)


    source_dir = './TestImages2'
    KEY = 'd7bb7b429dce4441a8f39b9f92088950'  # Replace with a valid subscription key (keeping the quotes in place).
    CF.Key.set(KEY)

    BASE_URL = 'https://eastus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL
    CF.BaseUrl.set(BASE_URL)

    temp_uri = "./feature_temp_file.pkl"
    (general_images, general_features, general_labels, face_images, face_features, face_iso_image, face_iso_features, face_labels) = faceLoader(source_dir)
    pickle.dump((general_images, general_features, general_labels, face_images, face_features, face_iso_image, face_iso_features, face_labels), open(temp_uri, 'wb'))

    (general_images, general_features, general_labels, face_images, face_features, face_iso_image, face_iso_features, face_labels) = pickle.load(open(temp_uri, 'rb'))
    unique_labels = list(set(face_labels)) #Get list of unique labels
    face_labels = list(map(lambda label: unique_labels.index(label), face_labels))
    general_labels = list(map(lambda label: unique_labels.index(label), general_labels))
    #stageCognitiveServiceFaceRecognition(source_dir)
    #(image_features, image_labels, face_features, face_labels) = pickle.load(open(temp_uri, 'rb'))
    
    '''
    X_train, X_test, y_train, y_test = train_test_split(image_features, image_labels, test_size=0.2, random_state=0)
    
    dec_tree_class = tree.DecisionTreeClassifier()
    model = dec_tree_class.fit(X_train, y_train)
    logging.debug(model.score(X_test, y_test))
    
    knn_class = KNeighborsClassifier(n_neighbors=2)
    model = knn_class.fit(X_train, y_train)
    logging.debug(model.score(X_test, y_test))
    '''

    def testFaceTransferNN(features, labels):
        model = Sequential()
        model.add(Dense(1024, input_dim=512, activation='relu')) #we add dense layers so that the model can learn more complex functions and classify for better results.
        model.add(Dense(1024,activation='relu')) #dense layer 2
        model.add(Dense(512,activation='relu')) #dense layer 3
        model.add(Dense(1,activation='sigmoid')) #final layer with softmax activation
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x=features, y=labels, epochs=5, batch_size=10)
        scores = model.evaluate(features, labels)
        logging.debug("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    features = np.array(face_iso_features)
    labels = np.array(face_labels)
    testFaceTransferNN(features, labels)

    features = np.array(face_features)
    labels = np.array(face_labels)
    testFaceTransferNN(features, labels)
    '''
    features = np.array(general_features)
    labels = np.array(general_labels)
    model = Sequential()
    model.add(Dense(1024, input_dim=4*4*512, activation='relu')) #we add dense layers so that the model can learn more complex functions and classify for better results.
    model.add(Dense(1024,activation='relu')) #dense layer 2
    model.add(Dense(512,activation='relu')) #dense layer 3
    model.add(Dense(1,activation='sigmoid')) #final layer with softmax activation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x=features, y=labels, epochs=5, batch_size=10)
    '''
    
    #generator = datagen.flow(
        #decoded_input,
        #target_size=(150, 150),
        #batch_size=batch_size)
    
    '''
    i = 0
    features = np.zeros(shape=(sample_size, 4, 4, 512))
    labels = np.zeros(shape=(sample_size))
    for inputs_batch, labels_batch in generator:
        features_batch = vgg_19_features_gen.predict(inputs_batch)
        features_batch2 = vgg_face_feature_gen.predict(inputs_batch)
        #features[i * batch_size : (i + 1) * batch_size] = features_batch
        #labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        #i += 1
        #if i * batch_size >= sample_size:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            #break
    '''

if __name__ == '__main__':
    main()