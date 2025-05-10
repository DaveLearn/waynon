# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional, Dict, List
import trio

import esper
from imgui_bundle import imgui

from waynon.components.tree_utils import find_nearest_ancestor_with_component, delete_entity, find_children_with_component
from waynon.components.component import Component
from waynon.components.node import Node
from waynon.components.robot import Robot, FrankaManager
from waynon.utils.utils import COLORS
from waynon.components.simple import Deletable, Pose, RobotPose

class PoseGroup(Component):
    color: list[float] = [1.0, 1.0, 1.0]


    def model_post_init(self, __context):
        self._cancel_context = trio.CancelScope()
        self._moving = False
        self._progress = 0
        self._total= 0

    def get_all_robots(self):
        """Get all robot entities in the world."""
        robot_ids = []
        for robot_id, _ in esper.get_component(Robot):
            robot_ids.append(robot_id)
        return robot_ids
    
    def get_robot_manager(self, robot_id):
        """Get the robot manager for a specific robot ID."""
        if robot_id is None:
            return None
        return esper.component_for_entity(robot_id, Robot).get_manager()

    def get_robot_managers(self, entity_id=None):
        """Get all robot managers used in this pose group's RobotPoses."""
        robot_ids = set()
        
        # Find all the robot_ids used in the RobotPoses under this PoseGroup
        if entity_id:
            pose_ids = find_children_with_component(entity_id, Pose)
            for pose_id in pose_ids:
                robot_pose_ids = find_children_with_component(pose_id, RobotPose)
                for robot_pose_id in robot_pose_ids:
                    robot_pose = esper.component_for_entity(robot_pose_id, RobotPose)
                    if robot_pose.robot_id is not None:
                        robot_ids.add(robot_pose.robot_id)
        
        # If no robot_ids found in the PoseGroup, get all robots in the world
        if not robot_ids:
            robot_ids = self.get_all_robots()
        
        robot_managers = {}
        for robot_id in robot_ids:
            manager = self.get_robot_manager(robot_id)
            if manager:
                robot_managers[robot_id] = manager
        
        return robot_managers
    
    def capture_robot_poses(self, entity_id):
        """Capture the current poses of all robots."""
        from waynon.components.scene_utils import create_motion
        from waynon.components.scene_utils import create_entity
        pose_id, _ = create_entity("Pose", entity_id, Pose(), Deletable())
        
        # Capture poses for all robots
        robot_managers = self.get_robot_managers()
        for robot_id, robot_manager in robot_managers.items():
            if robot_manager:
                create_motion(pose_id, robot_manager.read_q(), robot_id)
    
    async def cycle(self, entity_id):
        self._cancel_context.cancel()   
        self._cancel_context = trio.CancelScope()

        # Get all Pose nodes under this PoseGroup
        pose_ids = find_children_with_component(entity_id, Pose)
        if not pose_ids:
            print("No poses found")
            return
        
        # Count total poses for progress bar
        self._total = len(pose_ids)
        
        with self._cancel_context:
            self._moving = True
            
            # Cycle through each Pose node
            for i, pose_id in enumerate(pose_ids):
                pose = esper.component_for_entity(pose_id, Pose)
                # Move all robots to this pose
                await pose.move_to_pose(None, pose_id)
                self._progress = i + 1
                
                # Small delay between poses
                await trio.sleep(0.5)

        self._moving = False


    def draw_property(self, nursery, e):
        assert esper.has_component(e, PoseGroup)
        imgui.separator_text("Pose Group")
        group = esper.component_for_entity(e, PoseGroup)
        imgui.text_wrapped("Press 'circle' on any robot to add poses for all robots and 'cross' to delete the last pose added in this group.")
        imgui.spacing()
        
        robot_managers = self.get_robot_managers(e)
        
        # Disable the cycle button if any robot is not ready to move
        disabled = any(not manager.ready_to_move() for manager in robot_managers.values())
        
        if not self._moving:
            imgui.begin_disabled(disabled)
            imgui.push_style_color(imgui.Col_.button, COLORS["BLUE"])
            if imgui.button("Cycle", (imgui.get_content_region_avail().x, 40)):
                nursery.start_soon(self.cycle, e)
            if disabled:
                imgui.set_item_tooltip("One or more robots are not ready to move")
            imgui.pop_style_color()
            imgui.end_disabled()
        else:
            imgui.push_style_color(imgui.Col_.button, COLORS["RED"])
            if imgui.button("Stop", (imgui.get_content_region_avail().x, 20)):
                self._cancel_context.cancel()
            imgui.pop_style_color()
            imgui.progress_bar(self._progress / self._total, (imgui.get_content_region_avail().x, 40))

    def draw_context(self, nursery, entity_id):
        if imgui.menu_item_simple("Capture robot poses"):
            self.capture_robot_poses(entity_id)
        
    def on_selected(self, nursery, entity_id, just_selected):
        node = esper.component_for_entity(entity_id, Node)
        
        # Check if any robot has the circle button pressed
        circle_pressed = False
        cross_pressed = False
        
        robot_managers = self.get_robot_managers()
        for robot_id, robot_manager in robot_managers.items():
            if isinstance(robot_manager, FrankaManager):
                if robot_manager.is_button_pressed("circle"):
                    circle_pressed = True
                if robot_manager.is_button_pressed("cross"):
                    cross_pressed = True
        
        if circle_pressed:
            self.capture_robot_poses(entity_id)
        
        if cross_pressed:
            # Delete the last pose (not just the last robot pose)
            if node.children:
                for child in reversed(node.children):
                    if esper.has_component(child.entity_id, Pose):
                        delete_entity(child.entity_id)
                        break
    
    @staticmethod
    def default_name():
        return "Pose Group"