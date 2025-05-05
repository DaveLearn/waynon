# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import esper
import pyglet

from waynon.components.transform import Transform
from waynon.components.renderable import Mesh, ImageQuad, CameraWireframe, ArucoDrawable, StructuredPointCloud, CharucoDrawable
from waynon.components.camera import PinholeCamera
from waynon.components.aruco_marker import ArucoMarker
from waynon.components.charuco_board import CharucoBoard

class RenderProcessor(esper.Processor):

    def process(self):        
        for entity, (transform, drawable) in esper.get_components(Transform, Mesh):
            matrix = transform.get_X_WT()
            drawable.set_X_WT(matrix)

        for entity, (transform, drawable) in esper.get_components(Transform, ImageQuad):
            matrix = transform.get_X_WT()
            drawable.set_X_WT(matrix)

        for entity, (transform, camera, drawable) in esper.get_components(Transform, PinholeCamera, CameraWireframe):
            matrix = transform.get_X_WT()
            drawable.set_X_WT(matrix)
            drawable.update_intrinsics(camera.fl_x, camera.fl_y, camera.cx, camera.cy, camera.width, camera.height)
            texture = camera.get_texture()
            if texture is not None:
                drawable.set_texture_id(texture.id)

        for entity, (transform, sc) in esper.get_components(Transform, StructuredPointCloud):
            X_WT = transform.get_X_WT()
            sc.set_X_WT(X_WT)

        for entity, (transform, marker, drawable) in esper.get_components(Transform, ArucoMarker, ArucoDrawable):
            drawable.set_marker_dict(marker.marker_dict)
            drawable.set_marker_size(marker.marker_length)
            drawable.set_marker_id(marker.id)
            matrix = transform.get_X_WT()
            drawable.set_X_WT(matrix)

        for entity, (transform, board, drawable) in esper.get_components(Transform, CharucoBoard, CharucoDrawable):
            drawable.update_params(
                marker_length=board.marker_length,
                square_length=board.square_length,
                marker_dict=board.marker_dict,
                marker_id_offset=board.marker_id_offset,
                cols=board.cols,
                rows=board.rows
            )
            matrix = transform.get_X_WT()
            drawable.set_X_WT(matrix)
            

        