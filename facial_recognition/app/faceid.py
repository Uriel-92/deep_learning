# Import Kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


# Build app and layout
class CamApp(App):

    def build(self):
        # Main layout components
        self.img1 = Image(size_hint=(1, .8))
        self.button = Button(text='Verification', on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text='Verification Uninitiaded', size_hint=(1, .1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

		# Load tensorflow/keras model
        self.model = tf.keras.models.load_model('D:\\Documents\\facial_recognition\\siamese_model.h5', custom_objects={'L1Dist':L1Dist})

        # New varible, capture video
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

	# Reed continuosly to ged web cam feed
    def update(self, *args):
        
        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[80:80+250, 200:200+250, :]
        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img1.texture = img_texture

    def preprocess(self, file_path):
		
    	# Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in image
        img = tf.io.decode_jpeg(byte_img)
        # Preprocess resize (100,100,3)
        img = tf.image.resize(img, (100,100))
        # Normalize image 0-1
        img = img/255.0
        return img

    # Verify function
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[80:80+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build result arrays
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
	        input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
	        validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
	        
	        # Make predictions
	        result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
	        results.append(result)
	        
	    # Detection threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # Changue message in verification label
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

         # Log out details
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)
	    
        return results, verified

if __name__ == '__main__':
	CamApp().run()