#!/usr/bin/env python
# coding: utf-8

# # Classification of forest change using CSC's Puhti supercomputer

# You can get acces to CSC's services from https://research.csc.fi/

# ### Import neccessary libraries

# In[2]:


import os, time
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump, load
import numpy as np
import rasterio
from rasterio.merge import merge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import autokeras as ak
from tensorflow.keras import models, layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
import sys


# ### Load features and labels

# In[3]:

base_folder = '/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa'
csv_path = os.path.join(base_folder,'uusimaa_signatures_and_features_mean_VH_bitemporal.csv')
model_path = os.path.join(base_folder,'uusimaa_model.pkl')


# In[4]:


def load_signatures(sig_csv_path, sig_datatype=np.int32):
    """
    Extracts features and class labels from a signature CSV
    Parameters
    ----------
    sig_csv_path : str
        The path to the csv
    sig_datatype : dtype, optional
        The type of pixel data in the signature CSV. Defaults to np.int32

    Returns
    -------
    features : array_like
        a numpy array of the shape (feature_count, sample_count)
    class_labels : array_like of int
        a 1d numpy array of class labels corresponding to the samples in features.

    """
    data = np.genfromtxt(sig_csv_path, delimiter=",", dtype=sig_datatype).T
    return (data[1:, :].T, data[0, :])


# In[5]:


features, labels = load_signatures(csv_path,np.float64)


# In[6]:


# Check the shape of the features
features.shape


# ### Set path to input and prediction image 

# In[7]:


base_folder = "/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa"
#inputImage = '/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa/training_data/UM_max_VH_training.tif'
#inputImage = '/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa/UM_max_VH_ratio_db_training2.tif'
#inputImage = '/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa/UM_max_VH_pca_training.tif'
inputImage = '/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa/UM_bitemporal_mean_VH.tif'

outputImageBase = 'forest_loss_VH_mean_bitemporal'
results_folder = "/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa/deep_classification_results"
fullyConnectedModel = os.path.join(results_folder,'fullyConnectedModel_forestloss_yearly_means.json')
fullyConnectedWeights = os.path.join(results_folder,'fullyConnectedWeights_forestloss_yearly_means.h5')
# Predicted .tif image
fullyConnectedImageCropped = os.path.join(results_folder,'uusimaa_fullyConnected.tif')
data_folder = "/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa"


# In[8]:


# Set the number of available cores
n_jobs=4


# You can use random undersampling if the training data is unbalanced. In this study it was skipped

# In[8]:


#rus = RandomUnderSampler(random_state=63)
#pixels_resampled, labels_resampled = rus.fit_resample(features, labels)   
#print ('Dataframe shape after undersampling of majority classes, 2D: ', pixels_resampled.shape)


# In[9]:


#pixels_resampled.dump('pixels_resampled')
#labels_resampled.dump('labels_resampled')
#pixels_resampled = np.load('pixels_resampled', allow_pickle=True)
#labels_resampled = np.load('labels_resampled', allow_pickle=True)


# ### Split the training dataset training and test data 

# In[9]:


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=63)
np.unique(y_train, return_counts=True)


# ### Functions to train the model, estimate it's performance and do the prediction

# In[10]:


# Training the model
# Train the model and see how long it took.
# Credits: https://github.com/csc-training/geocomputing/blob/master/machineLearning/02_shallows/05_classification.ipynb
def trainModel(x_train, y_train, clf, classifierName):
    start_time = time.time()    
    # training the model
    clf.fit(x_train, y_train)
    print('Model training took: ', round((time.time() - start_time), 2), ' seconds')
    
    # Save the model to a file
    modelFilePath = os.path.join(base_folder, ('model_' + classifierName + '.sav'))
    dump(clf, modelFilePath) 
    return clf


# In[11]:


# Predict on test data and see the model accuracy
# Credicts: https://github.com/csc-training/geocomputing/blob/master/machineLearning/02_shallows/05_classification.ipynb
def estimateModel(clf, x_test, y_test):
    test_predictions = clf.predict(x_test)
    print('Confusion matrix: \n', confusion_matrix(y_test, test_predictions))
    print('Classification report: \n', classification_report(y_test, test_predictions))


# In[12]:


# Predict on whole image and save it as .tif file
# Credits: https://github.com/csc-training/geocomputing/blob/master/machineLearning/02_shallows/05_classification.ipynb
def predictImage(modelName, predictImage):
    #Set file paths for input and output files
    predictedClassesFile = outputImageBase + modelName + '_whole_uusimaa.tif'
    predictedClassesPath = os.path.join(base_folder, predictedClassesFile)
    
    # Read the satellite image
    with rasterio.open(predictImage, 'r') as image_dataset:
        start_time = time.time()    
        
        #Reshape data to 1D as we did before model training
        #image_data = image_dataset.read()
        image_data = image_dataset.read()
        image_data2 = np.transpose(image_data, (1, 2, 0))
        #pixels = image_data2.reshape(-1, 22)
        pixels = image_data2.reshape(-1, 2)
        #Load the model from the saved file
        modelFilePath = os.path.join(base_folder, ('model_' + modelName + '.sav'))
        trained_model = load(modelFilePath)
        
        # predict the class for each pixel
        prediction = trained_model.predict(pixels)
        
        # Reshape back to 2D
        print('Prediction shape in 1D: ', prediction.shape)
        prediction2D = np.reshape(prediction, (image_dataset.meta['height'], image_dataset.meta['width']))
        #prediction2D = np.reshape(prediction, ( 705, 1213))
        print('Prediction shape in 2D: ', prediction2D.shape)
        
        # Save the results as .tif file.
        # Copy the coorindate system information, image size and other metadata from the satellite image 
        outputMeta = image_dataset.meta
        
        # Change the number of bands and data type.
        #outputMeta.update(count=1, dtype='uint8')
        outputMeta.update(count=1, dtype='uint8', nodata=255)
        # Writing the image on the disk
        with rasterio.open(predictedClassesPath, 'w', **outputMeta) as dst:
            dst.write(prediction2D, 1)
        plt.imshow(prediction2D)
        print('Predicting took: ', round((time.time() - start_time), 1), ' seconds')
 


# In[16]:


array_job_id = int(sys.argv[1])


if array_job_id == 1:
    classifierName = 'random_forest'
    # Initialize the random forest classifier and give the hyperparameters.
    clf_random_forest = RandomForestClassifier(n_estimators=1000, max_depth=600, random_state=0, n_jobs=4)
    clf_random_forest = trainModel(x_train, y_train, clf_random_forest, classifierName)
    estimateModel(clf_random_forest, x_test, y_test)
    predictImage(classifierName, inputImage)
    print('Feature importances: \n', clf_random_forest.feature_importances_)


# In[ ]:

elif array_job_id == 2:

    classifierName = 'SGD'    
    clf_SGD = SGDClassifier(loss="log", learning_rate='adaptive', alpha=1e-6,  eta0=.01, n_jobs=n_jobs, max_iter=2000, penalty='l1')
    clf_SGD = trainModel(x_train, y_train, clf_SGD, classifierName)
    estimateModel(clf_SGD, x_test, y_test)
    predictImage(classifierName, inputImage)

elif array_job_id == 3:
    classifierName = 'gradient_boost'    
    clf_gradient_boost = GradientBoostingClassifier(n_estimators=500, learning_rate=.02)
    clf_gradient_boost = trainModel(x_train, y_train, clf_gradient_boost, classifierName)
    estimateModel(clf_gradient_boost, x_test, y_test)
    predictImage(classifierName, inputImage)
    print('Feature importances: \n', clf_gradient_boost.feature_importances_)

elif array_job_id == 4:
    classifierName = 'SVM'        
    clf_svc = SVC(kernel='rbf', gamma='auto',  decision_function_shape='ovr')
    clf_svc = trainModel(x_train, y_train, clf_svc, classifierName)
    estimateModel(clf_svc, x_test, y_test)
    predictImage(classifierName, inputImage) 

elif array_job_id == 5:

    # Train the fully connected model and save it.
    # Credits: https://github.com/csc-training/geocomputing/blob/master/machineLearning/03_deep/07_deepClassification.py
    def trainModel(x_train, y_train):
        start_time = time.time()    

        # Initializing a sequential model
        model = models.Sequential()
        # adding the first layer containing 64 perceptrons. 2 is representing the number of bands used for training
        model.add(layers.Dense(64, activation='relu', input_shape=(2,)))
        # add the first dropout layer
        model.add(layers.Dropout(rate=0.2))
        # adding more layers to the model
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(16, activation='relu'))
        # adding the output layer to the model, note:
        # - the activation is 'softmax', should be used for multi-class classification models
        # - size=3, for 3 classes
        model.add(layers.Dense(4, activation='softmax'))
        
        # Compile the model, using:
        # - Adam optimizer, often used, but could be some other optimizer too.
        # -- learning_rate is 0.001
        # - categorical_crossentropy loss function (should be used with multi-class classification)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Use one-hot-encoding to define a class for each pixel
        y_train_categorical = to_categorical(y_train)


        # Train the model
        model.fit(x_train,  y_train_categorical , epochs=2, batch_size=128, verbose=2)	
        
        # Save the model to disk
        # Serialize the model to JSON
        model_json = model.to_json()
        with open(fullyConnectedModel, "w") as json_file:
            json_file.write(model_json)
        # Serialize weights to HDF5
        model.save_weights(fullyConnectedWeights)
        print('Saved model to disk:  \nModel: ', fullyConnectedModel, '\nWeights: ',  fullyConnectedWeights)
        print('Model training took: ', round((time.time() - start_time), 0), ' seconds')
        return model    

    # Predict on test data and see the model accuracy
    def estimateModel(trained_model, x_test, y_test):   
        
        print(y_test)

        
        # Encode the test data labels
        y_test_categorical = to_categorical(y_test)
        
        #print(y_test_categorical)
        
        # Evaluate the performance of the model by the data, it has never seen
        # verbose=0 avoids printing to output a lot of unclear text (good in Puhti)
        test_loss, test_acc = trained_model.evaluate(x_test, y_test_categorical, verbose=0)
        print('Test accuracy:', test_acc)
        
        # Calculate confusion matrix and classification report as we did with shallow classifier.
        # Use scikit-learn functions for that.
        # First predict for the x_test
        test_prediction = trained_model.predict(x_test)	
        # The model returns a 2D array, with:
        # - each row representing one pixel.
        # - each column representing the probablity of this pixel representing each category	
        print ('Test prediction dataframe shape, original 2D: ', test_prediction.shape) 	
        
        # Find which class was most likely for each pixel and select only that class for the output.
        # Output is 1D array, with the most likely class index given for each pixel.
        # Argmax returns the indices of the maximum values 
        predicted_classes = np.argmax(test_prediction,axis=1)
        print ('Test prediction dataframe shape, after argmax, 1D: ', predicted_classes.shape) 	

        print('Confusion matrix: \n', confusion_matrix(y_test, predicted_classes))
        print('Classification report: \n', classification_report(y_test, predicted_classes))


            # Predict on whole image and save it as .tif file
    # Otherwise exactly the same as with shallow classifiers, but:
    # - Load the model from a file.
    # - argmax is used for the prediction results.
    # - Data type is changed to in8, keras returns int64, which GDAL does not support.
    def predictImage(predictedImagePath, predictImage):
        # Load json and create model
        json_file = open(fullyConnectedModel, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # Load weights into new model
        loaded_model.load_weights(fullyConnectedWeights)
        print("Loaded model from disk")
        
        # Read the satellite image
        with rasterio.open(predictImage, 'r') as image_dataset:
            start_time = time.time()    
            
            #Reshape data to 1D as we did before model training
            image_data = image_dataset.read()
            image_data2 = np.transpose(image_data, (1, 2, 0))
            pixels = image_data2.reshape(-1, 2)
            
            # Predict for all pixels
            prediction = loaded_model.predict(pixels)
            print ('Prediction dataframe shape, original 2D: ', prediction.shape) 	
            # Find the most likely class for each pixel.
            predicted_classes = np.argmax(prediction,axis=1)
            print ('Prediction dataframe shape, after argmax, 1D: ', predicted_classes.shape) 	
            
            # Reshape back to 2D as in original raster image
            prediction2D = np.reshape(predicted_classes, (image_dataset.meta['height'], image_dataset.meta['width']))
            print('Prediction shape in 2D: ', prediction2D.shape)
            
            # Change data type to int8
            predicted2D_int8 = np.int8(prediction2D)
            
            # Save the results as .tif file.
            # Copy the coordinate system information, image size and other metadata from the satellite image 
            outputMeta = image_dataset.meta
            # Change the number of bands and data type.
            outputMeta.update(count=1, dtype='int8')
            # Writing the image on the disk
            with rasterio.open(predictedImagePath, 'w', **outputMeta) as dst:
                dst.write(predicted2D_int8, 1)
            
            print('Predicting took: ', round((time.time() - start_time), 0), ' seconds')
    

    def main():
        # Read the input datasets with Rasterio
        #labels_dataset = rasterio.open(labelsImage)
        #image_dataset = rasterio.open(inputImage)  
        
        # Prepare data for the model
        # input_image, input_labels = prepareData(image_dataset, labels_dataset)
        # Divide the data to test and training datasets
        features, labels = load_signatures(csv_path,np.float64)
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=63)
        print(x_train.shape)
        print(y_train.shape)

        # Fit and predict the fully connected deep learning model on the data. Outputs a .tif image with the predicted classification.	
        print("FullyConnected")
        fullyConnectedModel = trainModel(x_train, y_train)	
        estimateModel(fullyConnectedModel, x_test, y_test)
        # Predict image for a small training area first
        predictImage(fullyConnectedImageCropped, inputImage)
        # Predict image then for the whole uusimaa
        fullyConnectedImage = os.path.join(results_folder,'whole_uusimaa_fullyConnected.tif')
        predictImage(fullyConnectedImage, '/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa/UM_bitemporal_mean_VH.tif')

    if __name__ == '__main__':
        ### This part just runs the main method and times it
        print("Script started!")
        start = time.time()
        main()
        end = time.time()
        print("Script completed in " + str(round((end - start),0)) + " seconds")


elif array_job_id == 6:
    # Set the parameters for the ScruturedDataClassifier
    clf = ak.StructuredDataClassifier(overwrite=False, max_trials=50, num_classes=3,seed=14)
    # Perform a so-called one-hot-encoding where the label of the pixel is expressed as a binary-array. For example the label 1 of a one pixel looks like this [0.0, 1.0, 0.0, 0.0]
    y_train_categorical = to_categorical(y_train)
    # Use the autokeras.StructuredDataClassifier.fit() -function to perform the pipeline optimization
    clf.fit(x_train, y_train_categorical, epochs=50)

    # Predict on test data and see the model accuracy
    def estimateModel(trained_model, x_test, y_test):
        
        print(y_test)

        
        # Encode the test data labels
        y_test_categorical = to_categorical(y_test)
        
        #print(y_test_categorical)
        
        # Evaluate the performance of the model by the data, it has never seen
        # verbose=0 avoids printing to output a lot of unclear text (good in Puhti)
        test_loss, test_acc = trained_model.evaluate(x_test, y_test_categorical, verbose=0)
        print('Test accuracy:', test_acc)
        
        # Calculate confusion matrix and classification report as we did with shallow classifier.
        # Use scikit-learn functions for that.
        # First predict for the x_test
        test_prediction = trained_model.predict(x_test)	
        # The model returns a 2D array, with:
        # - each row representing one pixel.
        # - each column representing the probablity of this pixel representing each category	
        print ('Test prediction dataframe shape, original 2D: ', test_prediction.shape) 	
        
        # Find which class was most likely for each pixel and select only that class for the output.
        # Output is 1D array, with the most likely class index given for each pixel.
        # Argmax returns the indices of the maximum values 
        predicted_classes = np.argmax(test_prediction,axis=1)
        print ('Test prediction dataframe shape, after argmax, 1D: ', predicted_classes.shape) 	

        print('Confusion matrix: \n', confusion_matrix(y_test, predicted_classes))
        print('Classification report: \n', classification_report(y_test, predicted_classes))

    # Estimate the model using the test data
    estimateModel(clf, x_test,y_test)
    print(clf.evaluate(x_test, to_categorical(y_test)))

    # Analyse the exported model with the summary() -function
    model = clf.export_model()
    model.summary()
    print(x_train.dtype)

    # Save the model to the disk
    try:
        model.save(os.path.join(results_folder,"model_autokeras_2512"), save_format="tf")
    except Exception:
        model.save(os.path.join(results_folder,"model_autokeras2512.h5"))

    # Saved models can be loaded like this:
    from tensorflow.keras.models import load_model
    loaded_model = load_model(os.path.join(results_folder,"model_autokeras_2512"), custom_objects=ak.CUSTOM_OBJECTS)
    print(loaded_model.summary())
    estimateModel(loaded_model, x_test,y_test)
    print(loaded_model.evaluate(x_test, to_categorical(y_test)))

    input_image = '/scratch/project_2004990/jutilaee/mtk_kehitys/uusimaa/UM_bitemporal_mean_VH.tif'
    output_file_name =  os.path.join(results_folder,'_whole_uusimaa_autokeras_bitemporal_mean_VH.tif')

        # Predict on whole image and save it as .tif file
    def predictImage(clf, input_image,output_image_path):
        #Set file paths for input and output files
        #predictedClassesFile = outputImageBase + modelName + '.tif'
        #predictedClassesPath = os.path.join(results_folder, predictedClassesFile)
        
        # Read the satellite image
        with rasterio.open(input_image, 'r') as image_dataset:
            start_time = time.time()    
            
            #Reshape data to 1D as we did before model training
            #image_data = image_dataset.read()
            image_data = image_dataset.read()
            image_data2 = np.transpose(image_data, (1, 2, 0))
            pixels = image_data2.reshape(-1, 2)
            
            #Load the model from the saved file
            #modelFilePath = os.path.join(base_folder, ('model_' + modelName + '.sav'))
            #trained_model = load(modelFilePath)
            
            # predict the class for each pixel
            prediction = clf.predict(pixels)
            prediction = np.argmax(prediction, axis=1)
            
            # Reshape back to 2D
            print('Prediction shape in 1D: ', prediction.shape)
            prediction2D = np.reshape(prediction, (image_dataset.meta['height'], image_dataset.meta['width']))
            #prediction2D = np.reshape(prediction, ( 705, 1213))
            print('Prediction shape in 2D: ', prediction2D.shape)
            
            # Save the results as .tif file.
            # Copy the coorindate system information, image size and other metadata from the satellite image 
            outputMeta = image_dataset.meta
            # Change the number of bands and data type.
            #outputMeta.update(count=1, dtype='uint8')
            outputMeta.update(count=1, dtype='uint8', height=image_dataset.meta['height'], width=image_dataset.meta['width'])
            # Writing the image on the disk
            with rasterio.open(output_image_path, 'w', **outputMeta) as dst:
                dst.write(prediction2D, 1)
            print('Predicting took: ', round((time.time() - start_time), 1), ' seconds')
    
    predictImage(loaded_model,input_image,output_file_name)

else:
    print('Wrong array job id!')
    print(int(sys.argv[1]))

