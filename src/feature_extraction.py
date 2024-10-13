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

from helper import crop_hexagon
class HEXAGON_IMAGE():
      def __init__(self, image:Image,grid_locatation, pixel_locatation,  save_path = None):
            self.image = image
            self.size = image.size
            self.pixel_locatation= pixel_locatation
            self.grid_locatation= grid_locatation
            self.save_path = save_path
            
            
            
            
            
class FIASS_Embedding():
      def __init__(self, model_name ='clip-ViT-B-32'):
            self.current_index = None
            self.model = SentenceTransformer(model_name)
            

      def generate_clip_embeddings(self,images:list[HEXAGON_IMAGE]):
            embeddings = []
            for row in images:
                  for img in row:
                        image = img.image
                        embedding = self.model.encode(image)
                        embeddings.append(embedding)
            return embeddings

      def create_faiss_index(self, embeddings, hex_images:list[HEXAGON_IMAGE], output_path = None):
            print(output_path)
            
            if(os.path.exists(output_path) and os.path.isdir(output_path)):
                  shutil.rmtree(output_path,'r')
            os.mkdir(output_path)
            
            output_path = output_path +  'index'
            dimension = len(embeddings[0])
            index = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap(index)
            
            vectors = np.array(embeddings).astype(np.float32)

            # Add vectors to the index with IDs
            index.add_with_ids(vectors, np.array(range(len(embeddings))))
            
            # Save the index
            if(output_path):
                  faiss.write_index(index, output_path)
                  print(f"Index created and saved to {output_path}")
            
                  # Save image locatations
                  with open(output_path + '.paths', 'w') as f:
                        for row in hex_images:
                              for img_path in row:
                                    f.write(img_path.save_path + '\n')
                              
                  # Save image locatations
                  with open(output_path + '.grid_locatation', 'w') as f:
                        for row in hex_images:
                              for img_path in row:
                                    f.write(f'{img_path.grid_locatation[0],img_path.grid_locatation[1]}'  + '\n')
                              
                  with open(output_path + '.pixel_locatation', 'w') as f:
                        for row in hex_images:
                              for img_path in row:
                                    f.write(f'{img_path.pixel_locatation[0],img_path.pixel_locatation[1]}' + '\n')
            
            return index


      def load_faiss_index(self,index_path):
            self.current_index = faiss.read_index(index_path)
            
            with open(index_path + '.paths', 'r') as f:
                  self.image_locatation = [line.strip() for line in f]
                  
            with open(index_path + '.grid_locatation', 'r') as f:
                  self.grid_locatation = [line.strip() for line in f]
                  
            with open(index_path + '.pixel_locatation', 'r') as f:
                  self.pixel_locatation = [line.strip() for line in f]
            print(f"Index loaded from {index_path}")


      def retrieve_similar_images(self, query, top_k=3, VERBOSE = False):
            
            query_features = self.model.encode(query)
            query_features = query_features.astype(np.float32).reshape(1, -1)
            if(self.current_index is None):
                  raise Exception("Must set index of FIASS Object")
            distances, indices = self.current_index.search(query_features, top_k)

            retrieved_locatations = [self.image_locatation[int(idx)] for idx in indices[0]]
            retrieved_grid_locatations = [self.grid_locatation[int(idx)].split(',') for idx in indices[0]]
            retrieved_pixel_locatations = [self.pixel_locatation[int(idx)].split(',') for idx in indices[0]]
            if VERBOSE:
                  self.visualize_results(query, retrieved_locatations, distances)
            return query, retrieved_locatations,retrieved_grid_locatations, retrieved_pixel_locatations,  distances



      def visualize_results(self,query, retrieved_images, distances):
            plt.figure(figsize=(12, 5))

            # If image query
            if isinstance(query, Image.Image):
                  plt.subplot(1, len(retrieved_images) + 1, 1)
                  plt.imshow(query)
                  plt.title("Query Image")
                  plt.axis('off')
                  start_idx = 2

            # If text query
            else:
                  plt.subplot(1, len(retrieved_images) + 1, 1)
                  plt.text(0.5, 0.5, f"Query:\n\n '{query}'", fontsize=16, ha='center', va='center')
                  plt.axis('off')
                  start_idx = 2

            # Display images
            for i, img_path in enumerate(retrieved_images):

                  plt.subplot(1, len(retrieved_images) + 1, i + start_idx)
                  plt.imshow(Image.open(img_path))
                  plt.title(f"Match {i + 1} Sim: {distances[i]} ")
                  plt.axis('off')

            plt.show()






      




class Input_Image():
      def __init__(self, image_path, hex_radius):
            self.image = Image.open(image_path)
            self.image_path = image_path
            self.image_height = self.image.size[0]
            self.image_width = self.image.size[1]
            self.hex_radius = hex_radius
            self.hex_height = math.sqrt(3) * hex_radius
            self.hex_width = 2 * hex_radius
            self.num_hexes_width = math.ceil(self.image_width / self.hex_height ) +2
            self.num_hexes_height = math.ceil(self.image_height / self.hex_width) +2
            self.hexagon_images = self.get_hexagons()
            print("Done with GETTING HEXAGONS")
            self.hexagon_imgages_dir = self.save_hexagon_images()
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
                        hex_image = crop_hexagon(self.image, (x, y), self.hex_radius)
                        hex_image = HEXAGON_IMAGE(hex_image, (r,q),(x,y))
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
            return save_dir
            
class Refrence_Image(Input_Image):
      def __init__(self,image_path, FIASS:FIASS_Embedding,  hex_radius = 50,):
            super().__init__(image_path, hex_radius)
            print("Done with SUPER")
            self.embeddings = FIASS.generate_clip_embeddings(self.hexagon_images)
            self.FAISS_INDEX = FIASS.create_faiss_index(self.embeddings,self.hexagon_images,self.hexagon_imgages_dir + '/indexes/')
            self.index_path = self.hexagon_imgages_dir +'/indexes/index'
            
                  
      

      


     
      
class Query_Image(Input_Image):
      def __init__(self,image_path, hex_radius = 50,):
            super().__init__(image_path,  hex_radius = 50,)
      def get_best_guess_of_positon(self,postion:tuple, refrence:Refrence_Image, FIASS_INSTANCE:FIASS_Embedding, top_k = 5):
            FIASS_INSTANCE.load_faiss_index(refrence.index_path)
            print(postion)
            image = self.hexagon_images[postion[0]][postion[1]].image
            query, retrieved_locatations,retrieved_grid_locatations, retrieved_pixel_locatations,  distances = FIASS_INSTANCE.retrieve_similar_images(image,top_k)
            distances = distances[0]
            distances = distances/(distances).max()
            pred_coord_x = []
            pred_coord_y = []
            for locatation, distance in zip(retrieved_pixel_locatations, distances):
                  retreived_coord_x,retreived_coord_y = locatation
                  retreived_coord_x = float(retreived_coord_x[1:])
                  retreived_coord_y = float(retreived_coord_y[:-1])
                  weight = np.exp(-distance)
                  pred_coord_x.append(retreived_coord_x*weight)
                  pred_coord_y.append(retreived_coord_y*weight)
            pred_coord_x = np.asarray(pred_coord_x).mean()
            pred_coord_y = np.asarray(pred_coord_y).mean()
            
            return (pred_coord_x,pred_coord_y)