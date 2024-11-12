from pyproj import Proj, transform
from PIL import Image
from owslib.wms import WebMapService
from pyproj import CRS
from cred import SENTENIAL2_TOKEN
import random
import numpy as np
import os
from feature_extraction import Input_Image


outProj = Proj(init='epsg:3857') 
inProj = Proj(init='epsg:4326')
wms = WebMapService(f"https://sh.dataspace.copernicus.eu/ogc/wms/{SENTENIAL2_TOKEN}")


"""
REGION FOR THE AGENT TO MOVE IN 

GRABS SENTENIAL IMAGES OF BOUNDING BOX FOR EACH YEAR AND SEPERATES THEM INTO HEXES OF HEX_SIZE 
"""
class Region():
      def __init__(self, bbox, years = [2016, 2018, 2020], size =(512,512), inProj = 'epsg:4326', outProj = 'epsg:3857', layer = '1_TRUE_COLOR', styles = 'RGB', output_format = 'image/jpeg', hex_size =50):
            self.size = size
            self.inCode = inProj
            self.outCode = outProj 
            self.inProj = Proj(init=inProj)
            self.outProj = Proj(init=outProj)
            self.layer = layer
            self.hex_size = hex_size
            self.output_format = output_format
            top_corner = transform(self.inProj,self.outProj,bbox[0], bbox[1])
            bottom_corner = transform(self.inProj,self.outProj,bbox[2], bbox[3])
            self.bbox = (top_corner[0], top_corner[1], bottom_corner[0], bottom_corner[1])
            self.images = {}
            for year in years:
                  self.images[year] = Year_ENV(self, year)
            
            
      
            
            
"""
EACH YEARS IMAGE 
"""          
class Year_ENV():
      def __init__(self, landscape:Region, year):
            self.year = year
            save_dir = f'./../images/{landscape.bbox[0]}-{landscape.bbox[1]}-{landscape.bbox[2]}-{landscape.bbox[3]}/'
            save_path = f'./../images/{landscape.bbox[0]}-{landscape.bbox[1]}-{landscape.bbox[2]}-{landscape.bbox[3]}/{year}.jpeg'
            #Make Path to save images 
            if(not os.path.exists(save_dir)):
                  os.mkdir(save_dir)
                  
            #If images have been pulled, dont pull them again
            if(os.path.isfile(save_path)):
                  image = Image.open(save_path)
            else:
                  res = wms.getmap( layers=[landscape.layer],
                                    styles =['RGB'],
                                    srs=landscape.outCode,
                                    bbox=landscape.bbox,
                                    size=landscape.size,
                                    format=landscape.output_format,
                                    time=f'{year}-10-20T12:00:00Z'
                                    
                                    )
                  image = Image.open(res)
                  image.save(save_path)
            
            self.Input_Image = Input_Image(save_path,image,  landscape.hex_size)
            
            
            #MOVE NOISE  (Correct, clockwise_1, counter_clockwise_1, clockwise_2, counter_clockwise_2, reverse)
            move_prob = np.array([1, random.random() * 0.1, random.random() * 0.1, random.random() * 0.03 ,random.random() * 0.03,random.random() * 0.01] )
            self.move_prob = move_prob/move_prob.mean()
            
      
      
      """
      MOVE DRIFT FUNCTION TO ADD SOME NOISE TO THE MOVE 
      """
      def move_drift(self, action_taken):
            moves = [0,1,-1,2,-2,3]
            move = random.choices(moves, self.move_prob)[0] + action_taken
            if(move > 5):
                  move -= 5
            elif(move < 0):
                  move +=5
            return move 
                         
                  
                  
                  
                        
                  
            
            
      


