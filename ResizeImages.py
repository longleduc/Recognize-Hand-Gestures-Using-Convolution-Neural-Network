from PIL import Image

#-----------------
# Resize the images so that it can be fed into the Convolution Neural Network which accepts 89 x 100 dimensional image.
#-----------------
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

for i in range(0, 300):
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/FistTest/fist_" + str(i) + '.png')
for i in range(0, 300):
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/PalmTest/palm_" + str(i) + '.png')
for i in range(0, 300):
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/SwingTest/swing_" + str(i) + '.png')
for i in range(0, 300):
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/LikeTest/like_" + str(i) + '.png')
for i in range(0, 300):
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/PeaceTest/peace_" + str(i) + '.png')

print("Resize the images done!")


