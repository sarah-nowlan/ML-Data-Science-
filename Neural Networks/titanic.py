
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')

y = train['Survived']
x = train.drop(labels=['PassengerId','Survived','Name','Ticket','Cabin','Embarked'],axis = 1)


from sklearn.preprocessing import LabelEncoder
encoder1 = LabelEncoder()
x['Sex'] = encoder1.fit_transform(x['Sex'])



x[['Age']] = x[['Age']].fillna(value=x[['Age']].mean())

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#x = sc.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,\
                                                    random_state = 0)

n_samples = x_train.shape[0]
n_featuers = x_train.shape[1]

#creating my model
from keras.layers import Dense, Dropout
from keras.models import Sequential

my_classifier = Sequential()

# Adding the input layer AND the first hidden layer (Pay attention to this)
my_classifier.add(Dense(units = 300, kernel_initializer = 'uniform',
                        activation = 'relu', input_dim = n_featuers))

# Adding the second hidden layer
my_classifier.add(Dense(units = 200, kernel_initializer = 'uniform',
                        activation = 'relu'))


my_classifier.add(Dense(units = 300, kernel_initializer = 'uniform',
                        activation = 'relu'))

my_classifier.add(Dropout(0.3))

my_classifier.add(Dense(units = 100, kernel_initializer = 'uniform',
                        activation = 'relu'))

# Adding the last (output) layer
my_classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
                        activation = 'sigmoid'))

my_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])



history = my_classifier.fit(x_train, y_train, validation_split=0.2,
                            batch_size = 20, epochs = 60)


y_pred_train = my_classifier.predict(x_train)
y_pred_train = (y_pred_train > 0.5)

# Predicting the Test set results
y_pred_test = my_classifier.predict(x_test)
y_pred_test = (y_pred_test > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)


# list all the data in history
print(history.history.keys())


# Plot the accuracy for both train and validation set
plt.subplots() # open a new plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# Plot the loss for both train and validation set
plt.subplots() # open a new plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


print(max(history.history['val_accuracy']))


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred_test,y_test))



