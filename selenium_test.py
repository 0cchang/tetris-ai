# importing webdriver from selenium
from selenium import webdriver

from PIL import Image

from time import sleep
 
# Here Chrome  will be used
driver = webdriver.Chrome()
 
# URL of website
url = "https://jstris.jezevec10.com/"
 
# Opening the website
driver.get(url)
sleep(10)
driver.save_screenshot("image5.png")
sleep(10)
driver.save_screenshot("image6.png")
 
# Loading the image
image = Image.open("image6.png")
 
# Showing the image
image.show()