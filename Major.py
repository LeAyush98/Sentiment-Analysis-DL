import numpy as np
import tkinter
import matplotlib.pyplot as plt
import tensorflow as tf
from tkinter import *
import tkinter.font
import tkinter.messagebox
from PIL import ImageTk,Image  
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

window = tkinter.Tk()
window.title("Major Project Group 4")
window.configure(bg = '#F9DED7')
canvas = Canvas(window, width = 300, height = 300 , bg = '#F9DED7')  
canvas.pack()    
img = ImageTk.PhotoImage(Image.open("jims.jpg"))  
canvas.create_image(20, 20, anchor = NW,  image=img) 

tkinter.Label(window, text = "Sentiment Analyser for Social Media", font = ("Ariel Black",50), bg = '#F9DED7').pack()



(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 20000)

X_train[0][:5]

X_train = pad_sequences(X_train, maxlen = 100)
X_test = pad_sequences(X_test, maxlen=100)


vocab_size = 20000
embed_size = 128

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding

model = Sequential()
model.add(Embedding(vocab_size, embed_size, input_shape = (X_train.shape[1],)))
model.add(LSTM(units=60, activation='tanh'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

history.history



k = history.history['accuracy'][-1]

k = (k*100)
NewK = ("%.2f" % k)

#tkinter.Label(window, text = k).pack()
def clicked():
 tkinter.messagebox.showinfo('Accuracy', NewK)
 #tkinter.Label(window, text = k,font = ("Ariel Black",30)).pack()
#window.geometry('800*400') 
tkinter.Button (window, text= "Click for model accuracy",font = ("Ariel Black",40),bg = 'black',fg = "white", command = clicked).pack() 


def plot_learningCurve (history, epochs):
# Plot training & validation accuracy values
 epoch_range = range(1, epochs+1)
 plt.plot(epoch_range, history.history['accuracy'])
 plt.plot(epoch_range, history.history['val_accuracy'])
 plt.title('Model accuracy')
 plt.ylabel('Accuracy')
 plt.xlabel('Epoch')
 plt.legend(['Train', 'Val'], loc = 'upper left')
 plt.show()

# Plot training & validation loss values
 plt.plot(epoch_range, history.history['loss'])
 plt.plot(epoch_range, history.history['val_loss'])
 plt.title('Model loss')
 plt.ylabel('Loss')
 plt.xlabel('Epoch')
 plt.legend (['Train', 'Val' ], loc= 'upper left' )
 plt.show()

plot_learningCurve(history,5) 



window.mainloop()


