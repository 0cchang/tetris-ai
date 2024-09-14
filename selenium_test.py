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
sleep(5)
driver.save_screenshot("image.png")

box = (439,202,1227,1180) #relative coords
image = Image.open("image.png")
image2 = image.crop(box)

image2.save("cropped_image.png")