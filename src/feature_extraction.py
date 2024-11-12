import os
from PIL import Image
import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
import math 
import scipy
from numpy.linalg import norm



from helper import crop_hexagon

class HEXAGON_IMAGE():
      def __init__(self, image:Image,hex_image_cropped:Image,bbox,  grid_locatation, pixel_locatation, hex_image_size, save_path = None):
            self.image = image
            self.cropped_image = hex_image_cropped.resize((hex_image_size, hex_image_size))
            self.bbox = bbox
            self.size = image.size
            self.cropped_size = self.cropped_image.size
            self.pixel_locatation= pixel_locatation
            self.grid_locatation= grid_locatation
            self.save_path = save_path
            self.embedding = None
            

class Input_Image():
      def __init__(self, Image_Path, Image, hex_radius, hex_image_size = 100):
            self.image_path = Image_Path
            self.image = Image
            self.hex_image_size = hex_image_size
            self.image_height = self.image.size[0]
            self.image_width = self.image.size[1]
            self.hex_radius = hex_radius
            self.hex_height = math.sqrt(3) * hex_radius
            self.hex_width = 2 * hex_radius
            self.num_hexes_width = math.ceil(self.image_width / self.hex_width )+2
            self.num_hexes_height = math.ceil(self.image_height / self.hex_height)
            self.hexagon_images = self.get_hexagons()

            

      def get_hexagons(self):
            """Get PIL images of each hexagon region"""
            hex_images = []
            hex_image = self.image
            for q in range(self.num_hexes_height):
                  hex_images.append([])
                  for r in range(self.num_hexes_width):
                        x = r * (self.hex_width - (math.cos(1.0472) * self.hex_radius)) 
                        y = q * (self.hex_height) + ((r%2) * math.sin(1.0472) * self.hex_radius) 
                        hex_image_cropped, hex_image_uncropped, bbox = crop_hexagon(self.image, (x, y), self.hex_radius)
                        hex_image = HEXAGON_IMAGE(hex_image_uncropped,hex_image_cropped,bbox, (q,r),(x,y), self.hex_image_size)
                        hex_images[q].append(hex_image)
            return hex_images

      
      
      
            


            
                  
      

      


     
      
