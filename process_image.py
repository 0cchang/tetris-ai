import cv2 as cv

def process_image(file):
    image = cv.imread(file)

    if image is None:
        print("Error: Could not load image.")
        exit()
    
    '''
    cv.imshow('Tetris Screenshot', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    '''
    cv.imshow('Gray SS', gray_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '''
    
    cv.imwrite('grayed' + file, gray_image)

    print('Gray image shape:', gray_image.shape)