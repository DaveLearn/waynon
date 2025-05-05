# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import cv2.aruco as aruco
import esper
import numpy as np

from imgui_bundle import imgui

from waynon.components.component import Component, ValidityResult
from waynon.components.camera import PinholeCamera
from waynon.components.tree_utils import try_component
from waynon.components.charuco_board import CharucoBoard


class CharucoBoardMeasurement(Component):
    detector_entity_id: int
    camera_entity_id: int
    board_entity_id: int

    corner_ids: list[int]
    corner_pixels: list[list[float]]
    marker_ids: list[int]
    marker_corners: list[list[list[float]]]

    def get_camera(self):
        return try_component(self.camera_entity_id, PinholeCamera)
    
    def get_board(self):
        return try_component(self.board_entity_id, CharucoBoard)
    
    def valid(self):
        if self.get_camera() is None:
            return ValidityResult.invalid("Camera not found")
        
        if self.get_board() is None:
            return ValidityResult.invalid("Charuco board not found")
        
        return ValidityResult.valid()

    def draw_property(self, nursery, entity_id):
        imgui.separator()
        imgui.text(f"Detected corners: {len(self.corner_ids)}")
        imgui.text(f"Detected markers: {len(self.marker_ids)}")

    def _fix_on_load(self, new_to_old_entity_ids):
        self.detector_entity_id = new_to_old_entity_ids.get(self.detector_entity_id, -1)
        self.camera_entity_id = new_to_old_entity_ids.get(self.camera_entity_id, -1)
        self.board_entity_id = new_to_old_entity_ids.get(self.board_entity_id, -1) 