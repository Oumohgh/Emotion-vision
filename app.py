import os
import time
import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace

# Suppress TensorFlow logging in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
