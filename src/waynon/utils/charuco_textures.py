# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict
import cv2
import cv2.aruco as aruco
import marsoom.texture

class CharucoTextures:
    def __init__(self):
        self.textures: Dict[int, marsoom.texture.Texture] = {}
    
    def get_texture(self, aruco_dict: int, aruco_id_offset: int, cols: int, rows: int, square_length: float, marker_length: float):
        texture_key = (aruco_dict, aruco_id_offset, cols, rows, square_length, marker_length)
        if texture_key not in self.textures:
            texture = marsoom.texture.Texture(1280, 720)
            dictionary = aruco.getPredefinedDictionary(aruco_dict)
            dictionary.bytesList = dictionary.bytesList[aruco_id_offset:]

            board = aruco.CharucoBoard((cols, rows), square_length, marker_length, dictionary)

            pixel_size = [int(dim * square_length * 1000)  for dim in (cols, rows)]
            board_img = board.generateImage(tuple(pixel_size), marginSize=0)

            board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2RGB)
            board_img = board_img.astype("float32") / 255.0

            texture.copy_from_host(board_img)
            self.textures[texture_key] = texture
        
        return self.textures[texture_key]

CHARUCO_TEXTURES = CharucoTextures()    