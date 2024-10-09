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

class FIASS_Embedding():
      def __init__(self, images, model_name ='clip-ViT-B-32', output_path = None):
            self.index = None
            self.model = SentenceTransformer(model_name)
            self.embedding, self.image_locatation = self.generate_clip_embeddings(images)
            self.index = self.create_faiss_index(self,output_path)
            

      def generate_clip_embeddings(self,images):
                  
            image_paths = glob(os.path.join(images, '**/*.png'), recursive=True)
            
            embeddings = []
            for img_path in image_paths:
                  image = Image.open(img_path)
                  embedding = self.model.encode(image)
                  embeddings.append(embedding)
            
            return embeddings, image_paths

      def create_faiss_index(self, output_path = None):
            
            dimension = len(self.embeddings[0])
            index = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap(index)
            
            vectors = np.array(self.embeddings).astype(np.float32)

            # Add vectors to the index with IDs
            index.add_with_ids(vectors, np.array(range(len(self.embeddings))))
            
            # Save the index
            if(output_path):
                  faiss.write_index(index, output_path)
                  print(f"Index created and saved to {output_path}")
            
                  # Save image locatations
                  with open(output_path + '.paths', 'w') as f:
                        for img_path in self.image_locatation:
                              f.write(img_path + '\n')
            
            return index


      def load_faiss_index(self,index_path):
            self.index = faiss.read_index(index_path)
            with open(index_path + '.paths', 'r') as f:
                  self.image_locatation = [line.strip() for line in f]
            print(f"Index loaded from {index_path}")


      def retrieve_similar_images(self, query, top_k=3, VERBOSE = False):
            
            # query preprocess:
            if query.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                  query = Image.open(query)

            query_features = self.model.encode(query)
            query_features = query_features.astype(np.float32).reshape(1, -1)

            distances, indices = self.index.search(query_features, top_k)

            retrieved_locatations = [self.image_locatation[int(idx)] for idx in indices[0]]
            if VERBOSE:
                  self.visualize_results(query, retrieved_locatations, distances)
            return query, retrieved_locatations, distances



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







def hex_corner(center, size, i):
      """Helper function to calculate hexagon corners."""
      angle_deg = 60 * i
      angle_rad = math.pi / 180 * angle_deg
      return (center[0] + size * math.cos(angle_rad), center[1] + size * math.sin(angle_rad))

def crop_hexagon(hex_image,center):
      """Crop Hexagons from hex image around center """
      
      corners = [hex_corner(center, hex_radius, i) for i in range(6)]
      mask = Image.new("RGBA", (hex_image.size[0],hex_image.size[1]))
      draw = ImageDraw.Draw(mask)
      draw.polygon(list(corners), fill='green', outline='red')
      
      background = Image.new("RGBA", hex_image.size, (0,0,0,0))
      new_img = Image.composite(hex_image, background, mask)
      return new_img

#
def get_hexagons():
      """Get PIL images of each hexagon region"""
      hex_segments = []
      hex_pos = []
      hex_image = img
      for q in range(size_height):
            for r in range(size_width):
                  x = r * (hex_width - (math.cos(1.0472) * hex_radius)) 
                  y = q * (hex_height) + ((r%2) * math.sin(1.0472) * hex_radius) 
                  hex_image = crop_hexagon(img, (x, y))
                  hex_segments.append(hex_image)
                  hex_pos.append([x,y])

      return hex_segments, hex_pos


def extract_features(self,image):
      img = Image.open(f'./../images/{image}.png')

      # Get hexagonal segments and their positions
      hexagons, positions = get_hexagons()


      DIR = f'./../images/{IMAGE}/'
      if(os.path.exists(DIR) and os.path.isdir(DIR)):
            shutil.rmtree(DIR,'r')
      os.mkdir(DIR)


      #Save all hexagons images into local directory
      for (x, y), hex_img in zip(positions, hexagons):
            imageBox = hex_img.getbbox()
            cropped = hex_img.crop(imageBox)
            
            cropped.save(f'./../images/{IMAGE}/{x}_{y}.png')

      


      
      print(f'DONE CROPPING {IMAGE}')
      # Plot each hexagon in its correct position
      if(VERBOSE):
            # Visualization with matplotlib
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            for (x, y), hex_img in zip(positions, hexagons):
                  ax.imshow(hex_img.resize((hex_img.size[0]-10,hex_img.size[1]-10)))

            ax.set_xlim(0, img.size[0])
            ax.set_ylim(img.size[1], 0)
            ax.axis('off')

            plt.show()
      
      """Create IMAGE EMBEDDINGS FROM FIASS
      """
      IMAGES_PATH = f'./../images/{IMAGE}/'
      OUTPUT_INDEX_PATH = f'./../images/{IMAGE}/indexes/'
      
      
      
      
      if(os.path.exists(OUTPUT_INDEX_PATH) and os.path.isdir(OUTPUT_INDEX_PATH)):
            shutil.rmtree(OUTPUT_INDEX_PATH,'r')
      os.mkdir(OUTPUT_INDEX_PATH)
      
      model = SentenceTransformer('clip-ViT-B-32')
      
      
      embeddings, image_paths = generate_clip_embeddings(IMAGES_PATH, model)
      
      index = create_faiss_index(embeddings, image_paths, OUTPUT_INDEX_PATH+'index')
      print(f'DONE INDEXING {IMAGE}')