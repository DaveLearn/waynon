from waynon.components.component import Component
from waynon.utils.charuco_textures import CHARUCO_TEXTURES
from waynon.components.aruco_marker import aruco_dict_names, aruco_dict_values
from imgui_bundle import imgui

import cv2.aruco as aruco
import numpy as np

class CharucoBoard(Component):
    marker_length: float = 0.0375
    square_length: float = 0.05
    marker_dict: int = aruco.DICT_4X4_1000
    marker_id_offset: int = 0
    cols: int = 4
    rows: int = 5

    def get_board(self):
        aruco_dict = aruco.getPredefinedDictionary(self.marker_dict)
        aruco_dict.bytesList = aruco_dict.bytesList[self.marker_id_offset:]
        return aruco.CharucoBoard((self.cols, self.rows), self.square_length, self.marker_length, aruco_dict)

    def get_texture(self):
        return CHARUCO_TEXTURES.get_texture(self.marker_dict, self.marker_id_offset, self.cols, self.rows, self.square_length, self.marker_length)

    def draw_property(self, nursery, entity_id):
        imgui.push_id(entity_id)
        imgui.separator_text("Charuco Board")
        t = self.get_texture()
        width = imgui.get_content_region_avail().x
        max_width = 200
        width = min(width, max_width)
        if t.id is not None:
            imgui.image(t.id, (width, width * (float(self.rows) / self.cols)))
        imgui.spacing()
        
        current_item = aruco_dict_values.index(self.marker_dict)
        if imgui.begin_combo("Dict", aruco_dict_names[current_item]):
            for i, (name, value) in enumerate(zip(aruco_dict_names, aruco_dict_values)):
                selected = current_item == i
                res, _ = imgui.selectable(name, selected)
                if res:
                    print(f"Selected {name}, {value}")
                    self.marker_dict = value
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        _, self.marker_id_offset = imgui.input_int("ID Offset", self.marker_id_offset)
        
        self.marker_id_offset = min(max(0, self.marker_id_offset), 1000)

        _, self.rows = imgui.input_int("Rows", self.rows)
        
        _, self.cols = imgui.input_int("Columns", self.cols)
        

        _, self.marker_length = imgui.input_float("Marker Length", self.marker_length)

        _, self.square_length = imgui.input_float("Square Length", self.square_length)


        imgui.pop_id()

    def get_P_MC(self) -> np.ndarray:
        board = self.get_board()
        corners = board.getChessboardCorners()
        return np.array(corners, dtype=np.float64)

    @staticmethod
    def default_name():
        return "Charuco"
