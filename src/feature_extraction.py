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


from helper import crop_hexagon
class HEXAGON_IMAGE():
      def __init__(self, image:Image,hex_image_cropped:Image,bbox,  grid_locatation, pixel_locatation,  save_path = None):
            self.image = image
            self.cropped_image = hex_image_cropped
            self.bbox = bbox
            self.size = image.size
            self.pixel_locatation= pixel_locatation
            self.grid_locatation= grid_locatation
            self.save_path = save_path
            self.embedding = None
            
            
            
            
            





      




class Input_Image():
      def __init__(self, image_path, hex_radius, saved = False):
            self.image = Image.open(image_path)
            self.image_path = image_path
            self.image_height = self.image.size[0]
            self.image_width = self.image.size[1]
            self.hex_radius = hex_radius
            self.hex_height = math.sqrt(3) * hex_radius
            self.hex_width = 2 * hex_radius
            self.num_hexes_width = math.ceil(self.image_width / self.hex_height ) +2
            self.num_hexes_height = math.ceil(self.image_height / self.hex_width) +1
            self.hexagon_images = self.get_hexagons()
            self.embeddings = []
            print("Done with GETTING HEXAGONS")
            self.hexagon_imgages_dir = self.image_path[:-4] + "/hexagon_images"
            if(not saved):
                  self.save_hexagon_images(self.hexagon_imgages_dir)
            print("Done with SAVING HEXAGONS")
            
            
            
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

      
      
      def save_hexagon_images(self, save_dir = None):
            if not save_dir:
                  save_dir = self.image_path[:-4] + "/hexagon_images"

            if(os.path.exists(self.image_path[:-4]) and os.path.isdir(self.image_path[:-4])):
                  shutil.rmtree(self.image_path[:-4],'r')
            os.mkdir(self.image_path[:-4])
            
            if(os.path.exists(save_dir) and os.path.isdir(save_dir)):
                  shutil.rmtree(save_dir,'r')
            os.mkdir(save_dir)


            #Save all hexagons images into local directory
            for row in (self.hexagon_images):
                  for img in row:
                        hex_img = img.image
                        x_pixel, y_pixel = img.pixel_locatation
                        imageBox = hex_img.getbbox()
                        cropped = hex_img.crop(imageBox)
                        cropped.save(f'{save_dir}/{x_pixel}_{y_pixel}.png')
                        img.save_path = f'{save_dir}/{x_pixel}_{y_pixel}.png'
            
class Refrence_Image(Input_Image):
      def __init__(self,image_path,  hex_radius = 50,saved = False):
            super().__init__(image_path, hex_radius, saved = saved)
            print("Done with SUPER")
            self.index_path = self.hexagon_imgages_dir +'/indexes/index'
            
      
            
                 
            
                  
      

      


     
      
class Query_Image(Input_Image):
      def __init__(self,image_path, hex_radius = 50, saved = False):
            super().__init__(image_path,  hex_radius=hex_radius, saved = saved)

            
            