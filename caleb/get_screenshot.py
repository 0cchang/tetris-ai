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
sleep(15)
driver.save_screenshot("image.png")

box = (450,200,1250,1200) #relative coords
image = Image.open("image.png")
image2 = image.crop(box)
image2.save("cropped_image.png")

#scan the color of each box, add to vector greyscaled version
img = Image.open("cropped_image.png")
pixels = img.load()

for i in range(20, 450, 45):
    vector = [] 
    for j in range(20, 930, 45):
        r, g, b = pixels[i, j]
        greyscale = 0.299 * r + 0.587 * g + 0.114 * b
        
        vector.append(greyscale)


    print(vector)

