import math 
from PIL import Image, ImageDraw
import numpy as np



def hex_corner(center, size, i):
      """Helper function to calculate hexagon corners."""
      angle_deg = 60 * i
      angle_rad = math.pi / 180 * angle_deg
      return (center[0] + size * math.cos(angle_rad), center[1] + size * math.sin(angle_rad))

def crop_hexagon(hex_image,center, hex_radius):
      """Crop Hexagons from hex image around center """

      corners = [hex_corner(center, hex_radius, i) for i in range(6)]
      mask = Image.new("RGBA", (hex_image.size[0],hex_image.size[1]))
      draw = ImageDraw.Draw(mask)
      draw.polygon(list(corners), fill='green', outline='red')
      
      background = Image.new("RGBA", hex_image.size, (0,0,0,0))
      new_img = Image.composite(hex_image, background, mask)
      min_x = min([point[0] for point in corners])
      max_x = max([point[0] for point in corners])
      min_y = min([point[1] for point in corners])
      max_y = max([point[1] for point in corners])

      # Crop the image to the bounding box
      bounding_box = (min_x, min_y, max_x, max_y)
      new_img_cropped = new_img.crop(bounding_box)
      return new_img_cropped, new_img, bounding_box



def uncertinity_function(location_expected, location_guessed):
      return math.sqrt((location_expected[0] - location_guessed[0]) ** 2  + (location_expected[1] - location_guessed[1]) ** 2)

