#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage  # Add CompressedImage import
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from std_msgs.msg import Bool
from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np
import pyzed.sl as sl
from cv_bridge import CvBridge
from enum import IntEnum
from typing import Tuple, List
import time
import math
from datetime import datetime


class Keypoint(IntEnum):
    PELVIS = 0
    NAVAL_SPINE = 1
    CHEST_SPINE = 2
    NECK = 3
    LEFT_CLAVICLE = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    LEFT_HAND = 8
    LEFT_HANDTIP = 9
    LEFT_THUMB = 10
    RIGHT_CLAVICLE = 11
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 13
    RIGHT_WRIST = 14
    RIGHT_HAND = 15
    RIGHT_HANDTIP = 16
    RIGHT_THUMB = 17
    LEFT_HIP = 18
    LEFT_KNEE = 19
    LEFT_ANKLE = 20     
    LEFT_FOOT = 21
    RIGHT_HIP = 22
    RIGHT_KNEE = 23
    RIGHT_ANKLE = 24
    RIGHT_FOOT = 25
    HEAD = 26
    NOSE = 27
    LEFT_EYE = 28
    LEFT_EAR = 29
    RIGHT_EYE = 30
    RIGHT_EAR = 31
    LEFT_HEEL = 32
    RIGHT_HEEL = 33



ID_COLORS = [(232, 176, 59),
            (175, 208, 25),
            (102, 205, 105),
            (185, 0, 255),
            (99, 107, 252)]

def cvt(pt, scale):
    return [pt[0]*scale[0], pt[1]*scale[1]]

def render_object(object_data, is_tracking_on):
    if is_tracking_on:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
    else:
        return ((object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or 
                (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF))

def generate_color_id_u(idx):
    if idx < 0:
        return [236, 184, 36, 255]
    color_idx = idx % 5
    return [*ID_COLORS[color_idx], 255]

def render_sk(left_display, img_scale, obj, color, BODY_BONES):
    # Draw skeleton bones
    for part in BODY_BONES:
        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
        # Check that the keypoints are inside the image
        if (kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0] 
            and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
            and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0):
            cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), 
                    (int(kp_b[0]), int(kp_b[1])), color, 1, cv2.LINE_AA)

    # Skeleton joints
    for kp in obj.keypoint_2d:
        cv_kp = cvt(kp, img_scale)
        if cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]:
            cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)

def get_body_bbox(keypoints_2d: np.ndarray, margin: float = 0.2) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Calculate bounding box with margin for body keypoints"""
    if len(keypoints_2d) == 0:
        return ((0, 0), (0, 0))
        
    x_coords = keypoints_2d[:, 0]
    y_coords = keypoints_2d[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Add margin
    x_min -= width * margin
    x_max += width * margin
    y_min -= height * margin
    y_max += height * margin
    
    return ((int(max(0, x_min)), int(max(0, y_min))), 
            (int(x_max), int(y_max)))

def is_hand_above_head(keypoints_2d: np.ndarray) -> bool:
    """Check if either hand is above the head"""
    head_y = keypoints_2d[Keypoint.HEAD.value][1]
    left_hand_y = keypoints_2d[Keypoint.LEFT_HAND.value][1]
    right_hand_y = keypoints_2d[Keypoint.RIGHT_HAND.value][1]
    
    # Note: In image coordinates, lower y is higher in the image
    return left_hand_y < head_y or right_hand_y < head_y

class ZedBodyTrackerNode(Node):
    def __init__(self):
        super().__init__('zed_body_tracker')
        self.zed = sl.Camera()
        ip = sl.InitParameters() 
        ip.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT #PERFORMANCE
        ip.coordinate_units = sl.UNIT.METER

        # ip.set_from_svo_file("/home/user/Documents/ZED/HD1200_SN56354199_16-25-22.svo2")
        # ip.svo_real_time_mode = False
        #ip.svo_loop_mode = sl.SVO_LOOP_MODE.SEMI_AUTO
        
        if self.zed.open(ip)!=sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("ZED open failed")

        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        bt_par = sl.BodyTrackingParameters(enable_tracking=True,
                                           enable_body_fitting=True,
                                           detection_model=sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST,
                                           body_format = sl.BODY_FORMAT.BODY_34)
        self.zed.enable_body_tracking(bt_par)
        self.bt_rt = sl.BodyTrackingRuntimeParameters()
        self.bt_rt.detection_confidence_threshold = 0.5
        
        # # Initialize publishers
        self.image_pub = self.create_publisher(CompressedImage, 'zed_image/compressed', 10)
        self.body_data_pub = self.create_publisher(Float32MultiArray, 'body_tracking_data', 10)
        
        # Add new publisher for cropped image
        self.crop_pub = self.create_publisher(CompressedImage, 'zed_image_cropped/compressed', 10)
        
        # Add new publishers
        self.countdown_pub = self.create_publisher(CompressedImage, 'countdown_display/compressed', 10)
        self.final_photo_pub = self.create_publisher(CompressedImage, 'final_photo/compressed', 10)
        self.countdown_pub = self.create_publisher(Bool, 'countdown_starting', 10)
        
        self.bridge = CvBridge()
        
        # Subscribe to processing status
        self.processing_sub = self.create_subscription(
            Bool,
            'processing_status',
            self.processing_status_callback,
            10)
        self.is_processing = False
        
        # Initialize body tracking objects
        self.bodies = sl.Bodies()
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        # Create timer for processing
        self.timer = self.create_timer(0.1, self.timer_callback)  # ~30fps
        
        self.hand_raised_start_time = {}  # Track hand raised time for each person
        self.last_hand_state = {}  # Track previous hand state for each person
        
        # Add state tracking
        self.countdown_state = "idle"  # idle, first_countdown, second_countdown
        self.countdown_start_time = None
        self.last_valid_crop = None
        self.hand_raised_duration = 0.0
        self.last_valid_bbox = None  # Store bbox instead of crop

    def adjust_bbox_aspect_ratio(self, x1: int, y1: int, x2: int, y2: int, 
                               min_aspect_ratio: float = 1.0) -> Tuple[int, int, int, int]:
        """Adjust bounding box to maintain minimum width/height ratio"""
        width = x2 - x1
        height = y2 - y1
        
        if width < height * min_aspect_ratio:
            # Need to increase width
            extra_width = (height * min_aspect_ratio - width) / 2
            x1 = max(0, int(x1 - extra_width))
            x2 = int(x2 + extra_width)
        
        return x1, y1, x2, y2

    def processing_status_callback(self, msg: Bool):
        """Handle processing status updates"""
        self.is_processing = msg.data

    def draw_banner_text(self, image: np.ndarray, text: str) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Create copy of image to avoid modifying original
        result = image.copy()
        
            
        # Get text size for centering
        text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        
        # Draw text at top with dark background for visibility
        text_y = text_size[1] + 20  # 20px padding from top
        cv2.rectangle(result, (0, 0), (image.shape[1], text_y + 10), (0, 0, 0), -1)
        if text is not "empty":
            cv2.putText(result, text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)

        return result
    
    def add_countdown_text(self, image: np.ndarray, seconds_left: float, text_prefix: str = "") -> np.ndarray:
        """Add countdown text at the top of the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Create copy of image to avoid modifying original
        result = image.copy()
        
        # Format countdown text
        countdown_text = f"{text_prefix}{math.ceil(seconds_left)}s"
        if text_prefix == "":
            countdown_text += " - Keep hand above head"
            
        # Get text size for centering
        text_size = cv2.getTextSize(countdown_text, font, 1.5, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        
        # Draw text at top with dark background for visibility
        text_y = text_size[1] + 20  # 20px padding from top
        cv2.rectangle(result, (0, 0), (image.shape[1], text_y + 10), (0, 0, 0), -1)
        cv2.putText(result, countdown_text, (text_x, text_y), font, 1.5, (255, 255, 255), 2)

        return result
        
    def timer_callback(self):
        current_time = time.time()
        
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Get camera image
            image = sl.Mat()
            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            cv_image = image.get_data()
            
            # Convert BGRA to BGR
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
            
            # Create a copy for annotation
            annotated_image = cv_image_bgr.copy()
            
            # Track bodies
            self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)
            
            # Calculate image scale
            img_scale = [1, 1]  # Modify if scaling is needed
            
            # Track bounding box for all bodies and handle tracking
            all_keypoints = []  # Initialize as regular Python list
            hands_above = False
            cropped_image = None
            
            for body in self.bodies.body_list:
                if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    # Check hand position first
                    hand_above = is_hand_above_head(body.keypoint_2d)
                    
                    # Only include keypoints if hand is raised
                    if hand_above:
                        hands_above = True
                        # Convert keypoints to list before extending
                        keypoints_list = [cvt(kp, img_scale) for kp in body.keypoint_2d]
                        all_keypoints.extend(keypoints_list)  # Only add if hand is raised
                        
                        # Draw skeleton
                        color = generate_color_id_u(body.id)
                        render_sk(annotated_image, img_scale, body, color, sl.BODY_34_BONES)
                        
                        if body.id not in self.hand_raised_start_time:
                            self.hand_raised_start_time[body.id] = current_time
                        self.hand_raised_duration = current_time - self.hand_raised_start_time[body.id]
            
            # Convert to numpy array only when needed for bbox calculation
            if all_keypoints:
                all_keypoints_array = np.array(all_keypoints)
                (x1, y1), (x2, y2) = get_body_bbox(all_keypoints_array)
                
                # Adjust bounding box to maintain minimum aspect ratio
                x1, y1, x2, y2 = self.adjust_bbox_aspect_ratio(x1, y1, x2, y2)
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Crop original image to adjusted bounding box
                cropped_image = cv_image_bgr[y1:y2, x1:x2]
            
            # Process body tracking data
            body_data = Float32MultiArray()
            body_data.data = []
            
            for body in self.bodies.body_list:
                if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    keypoints = body.keypoint
                    for kp in keypoints:
                        body_data.data.extend([kp[0], kp[1], kp[2]])
                        
            annotated_image = self.draw_banner_text(annotated_image, "empty")
            # Only check for raised hands if not currently processing
            if not self.is_processing:
                # Track hands above head
                hands_above = False
                for body in self.bodies.body_list:
                    if body.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                        # Add keypoints to list for combined bounding box
                        all_keypoints.extend([cvt(kp, img_scale) for kp in body.keypoint_2d])
                        
                        # Draw skeleton
                        color = generate_color_id_u(body.id)
                        render_sk(annotated_image, img_scale, body, color, sl.BODY_34_BONES)
                        
                        # Check hand position
                        hand_above = is_hand_above_head(body.keypoint_2d)
                        
                        if hand_above:
                            hands_above = True
                            if body.id in self.hand_raised_start_time:
                                self.hand_raised_duration = current_time - self.hand_raised_start_time[body.id]
            else:
                # Reset hand tracking state when processing
                self.hand_raised_duration = 0.0
                self.hand_raised_start_time.clear()
                    
            if self.countdown_state == "idle" and self.hand_raised_duration < 0.5 and not self.is_processing:
                annotated_image = self.draw_banner_text(annotated_image, "Raise arm above head to start")
                
            if self.is_processing:
                annotated_image = self.draw_banner_text(annotated_image, "Generating...")

            # Handle countdown states
            if self.countdown_state == "idle" and self.hand_raised_duration > 0.5:
                self.countdown_state = "first_countdown"
                self.countdown_start_time = current_time
                
            elif self.countdown_state == "first_countdown":
                time_elapsed = current_time - self.countdown_start_time
                seconds_left = 5.0 - time_elapsed
                
                # Reset countdown if hand is lowered
                if not hands_above:
                    self.countdown_state = "idle"
                    self.countdown_start_time = None
                    self.hand_raised_duration = 0.0
                    self.hand_raised_start_time.clear()
                    annotated_image = self.draw_banner_text(annotated_image, "Raise arm above head to start")
                elif seconds_left <= 0:
                    self.countdown_state = "second_countdown"
                    self.countdown_start_time = current_time

                    # Publish countdown start signal to reset media display
                    countdown_msg = Bool()
                    countdown_msg.data = True
                    self.countdown_pub.publish(countdown_msg)

                    if all_keypoints:
                        x1, y1, x2, y2 = self.adjust_bbox_aspect_ratio(x1, y1, x2, y2)
                        self.last_valid_bbox = (x1, y1, x2, y2)
                else:
                    # Add countdown text to image
                    annotated_image = self.add_countdown_text(annotated_image, seconds_left)
                    
            elif self.countdown_state == "second_countdown":
                time_elapsed = current_time - self.countdown_start_time
                seconds_left = 5.0 - time_elapsed
                
                if seconds_left <= 0:
                    # Publish final photo using current frame with stored bbox
                    if self.last_valid_bbox is not None:
                        x1, y1, x2, y2 = self.last_valid_bbox
                        final_crop = cv_image_bgr[y1:y2, x1:x2]
                        
                        final_msg = CompressedImage()
                        final_msg.header.stamp = self.get_clock().now().to_msg()
                        final_msg.format = "jpeg"
                        final_msg.data = np.array(cv2.imencode('.jpg', final_crop, 
                                                [cv2.IMWRITE_JPEG_QUALITY, 99])[1]).tobytes()
                        self.final_photo_pub.publish(final_msg)
                        current_time = self.get_clock().now()
                        self.get_logger().info(f'Sent snapshot image: {final_msg.header.stamp.sec+final_msg.header.stamp.nanosec*1e-9}, current time: {current_time.nanoseconds/1e9}')
        
                    # Reset state
                    self.countdown_state = "idle"
                    self.countdown_start_time = None
                    self.last_valid_crop = None
                    self.hand_raised_duration = 0.0
                    self.hand_raised_start_time.clear()
                    
                    # Add prompt text to image
                    prompt_text = "Generating..." if self.is_processing else "Raise arm above head to start"
                    self.draw_banner_text(annotated_image, prompt_text)
                else:
                    # Add countdown text to image
                    annotated_image = self.add_countdown_text(annotated_image, seconds_left, "Taking photo in: ")
            
            # Publish the final annotated image with all overlays
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', annotated_image, 
                              [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).tobytes()
            
            # Publish data
            self.image_pub.publish(msg)
            self.body_data_pub.publish(body_data)
            
        else:
            self.get_logger().warn("ZED camera grab failed")
    def __del__(self):
        if hasattr(self, 'zed'):
            self.zed.close()

def main(args=None):
    rclpy.init(args=args)
    node = ZedBodyTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
