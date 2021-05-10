import streamlit as st
from detect import detector
import cv2
import numpy as np
import time
import math
import imutils


menu = ['Welcome', 'Social distancing detector', 'Learn more!']
with st.sidebar.beta_expander("Menu", expanded=False):
    option = st.selectbox('Choose your task', menu)
if option == 'Welcome':
    st.image('data/2MCAM_logo.png')
    st.title('2[M]CAM')
    st.subheader('The one stop solution to monitor social ditancing norms in any public environent')
    st.write('Welcome to the 2MCAM web application that can accurately identify human beings that violate social distancing norms in public spaces.')

elif option == 'Social distancing detector':
    st.title('Social Distancing Detection')
    st.write('The practice of social distancing signifies maintaining a distance of 6 feet or more when in public places or simply staying at home and away from others as much as possible to help prevent spread of COVID-19.')
    st.write('A green/red bounding box is drawn over the individuals in the frame to approve or flag the distance between them. ')
    if st.button("Start"):
        cap = cv2.VideoCapture(0)
        ph = st.empty()
        detector(cap,ph)
        cap.release()
        cv2.destroyAllWindows()
elif option=='Learn more!':
    st.title('Why 2MCAM?')
    st.image('data/img1.jpg')
    st.write('2MCAM is a user-friendly ML web application and is designed to detect possible violations of social distancing norms.')
    st.write('Violation of physical distancing norms (1.5 - 2 meters) is looked down upon in the era of the covid19 pandemic. This is why we have a solution for you: 2MCAM, the web application that can instantaneously detect and flag such violations.')
    st.image('data/2feet.jpeg')
    st.write('Our application has a light-weight frontend and heavily tested backend waiting to be used. Discover the working of our application by navigating to the social distancing detector section.')
