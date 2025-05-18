# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Tuple, Optional
import cv2
import cv2.aruco as aruco
import numpy as np
import esper
import trio

from waynon.components.tree_utils import *
from .measurement_processor import MeasurementProcessor


class CharucoBoardProcessor(MeasurementProcessor):
    async def run(self, detector_id: int, measurement_id: int):
        from waynon.components.charuco_board_detector import CharucoBoardDetector
        from waynon.components.charuco_board_measurement import CharucoBoardMeasurement
        from waynon.components.measurement import Measurement
        from waynon.components.image_measurement import ImageMeasurement
        from waynon.components.joint_measurement import JointMeasurement
        from waynon.components.charuco_board import CharucoBoard
        from waynon.components.simple import Deletable

        assert esper.entity_exists(detector_id)
        assert esper.entity_exists(measurement_id)
        assert esper.has_component(detector_id, CharucoBoardDetector)
        assert esper.has_component(measurement_id, Measurement)

        image_measurements = find_children_with_component(measurement_id, ImageMeasurement)
        for iid in image_measurements:
            delete_children(iid, lambda id, c: isinstance(c, CharucoBoardMeasurement))

            image_measurement = esper.component_for_entity(iid, ImageMeasurement)

            image = image_measurement.get_image_u()
            all_boards = esper.get_component(CharucoBoard)
            if len(all_boards) == 0:
                print("Warning: No CharucoBoards in system")
                return

            for board_entity_id, board in all_boards:
                res = await trio.to_thread.run_sync(
                    detect_charuco_board, 
                    image, 
                    board.get_board(),
                )
                
                if res:
                    corner_ids, corner_pixels, marker_ids, marker_corners = res
                    
                    if corner_ids is not None and len(corner_ids) > 0:
                        charuco_measurement = CharucoBoardMeasurement(
                            camera_entity_id=image_measurement.camera_id,
                            board_entity_id=board_entity_id,
                            detector_entity_id=detector_id,
                            corner_ids=corner_ids.tolist(),
                            corner_pixels=corner_pixels.tolist(),
                            marker_ids=marker_ids.tolist(),
                            marker_corners=marker_corners.tolist()
                        )
                        create_entity(f"Charuco Board Detection", iid, charuco_measurement, Deletable())


def detect_charuco_board(
    img: np.ndarray,
    board: aruco.CharucoBoard
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Detect a charuco board in an image.
    
    Returns:
        tuple containing (corner_ids, corner_pixels, marker_ids, marker_pixels)
    """
    assert img.dtype == np.uint8
    
    
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    
    detector = aruco.CharucoDetector(board, detectorParams=parameters)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)
    
    if marker_ids is None or len(marker_ids) == 0:
        return None
    
    if charuco_ids is None or len(charuco_ids) == 0:
        return None

    return np.array(charuco_ids).squeeze(1), np.array(charuco_corners).squeeze(1), np.array(marker_ids).squeeze(1), np.array(marker_corners).squeeze(1)


# Create a global instance of the processor
CHARUCO_PROCESSOR = CharucoBoardProcessor() 