from PIL import Image

from feature_extraction import FIASS_Embedding, Query_Image, Refrence_Image
path = './../images/2016.png'
FAISS = FIASS_Embedding()
Reference_Image = Refrence_Image(path, FAISS)
path = './../images/2018.png'
Input_Image = Query_Image(path)
Input_Image.get_best_guess_of_positon((0,0), Reference_Image, FAISS)