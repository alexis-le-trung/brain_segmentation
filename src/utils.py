import matplotlib.pyplot as plt

def fast_display(*lst_images):
    plt.figure(figsize=(16,8))
    nb_images = len(lst_images)
    cols = min(9,nb_images)
    rows = (nb_images // cols) + 1
    for ii,image in enumerate(lst_images):
        plt.subplot(rows,cols,1+ii)
        plt.imshow(image)
    plt.show()


def PrintSlices(img):
    sx,sy,sz,_ = img.shape
    fast_display(img[sx//2,:,:,0],img[:,sy//2,:,0],img[:,:,sz//2,0])
