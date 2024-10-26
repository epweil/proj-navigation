import os
from glob import glob
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
import math 
import shutil
import scipy
from numpy.linalg import norm

RESIZE = 150
from helper import crop_hexagon
class HEXAGON_IMAGE():
      def __init__(self, image:Image,hex_image_cropped:Image,bbox,  grid_locatation, pixel_locatation,  save_path = None):
            self.image = image
            self.cropped_image = hex_image_cropped.resize((RESIZE, RESIZE))
            self.bbox = bbox
            self.size = image.size
            self.cropped_size = self.cropped_image.size
            self.pixel_locatation= pixel_locatation
            self.grid_locatation= grid_locatation
            self.save_path = save_path
            self.embedding = None
            
            
            
            
            
class FIASS_Embedding():
      def __init__(self, model_name ='clip-ViT-B-32'):
            self.current_index = None
            self.model = SentenceTransformer(model_name)
            

      def generate_clip_embeddings(self,img:HEXAGON_IMAGE):
            return self.model.encode(img.image)


     





      




class Input_Image():
      def __init__(self, Image_Path, Image, hex_radius, FIASS:FIASS_Embedding):
            self.image_path = Image_Path
            self.image = Image
            self.image_height = self.image.size[0]
            self.image_width = self.image.size[1]
            self.hex_radius = hex_radius
            self.hex_height = math.sqrt(3) * hex_radius
            self.hex_width = 2 * hex_radius
            self.num_hexes_width = math.ceil(self.image_width / self.hex_width )+2
            self.num_hexes_height = math.ceil(self.image_height / self.hex_height)
            self.hexagon_images = self.get_hexagons()
            

            for row in self.hexagon_images:
                  for hex_img in row:
                        hex_img.embedding = FIASS.generate_clip_embeddings(hex_img)
            
            
            
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
                        hex_image = HEXAGON_IMAGE(hex_image_uncropped,hex_image_cropped,bbox, (q,r),(x,y))
                        hex_images[q].append(hex_image)
            return hex_images

      
      
      
            


            
                  
      

      


     
      
