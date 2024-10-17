from  src.feature_extraction  import FIASS_Embedding, Refrence_Image

FIASS = FIASS_Embedding()

refrence_image = Refrence_Image('./images/2016.png', FIASS, 50, True)
img = refrence_image.hexagon_images[0][3]
img.show()
# np.sum(region * template_array) / np.sqrt(np.sum(region**2) * np.sum(template_array**2))