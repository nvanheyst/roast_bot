#!/usr/bin/env python3

OPENAI_API_KEY = 'sk-proj-...'


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool
import cv2
import numpy as np
from cv_bridge import CvBridge
import base64
from openai import OpenAI  # Updated import
from PIL import Image as PILImage
import io
import os
from datetime import datetime
import time as time


class MediaGenerationNode(Node):
    def __init__(self):
        super().__init__('media_generation')
        
        # Initialize parameters
        api_key = OPENAI_API_KEY #self.get_parameter('openai_api_key').value
        if not api_key:
            self.get_logger().error('OpenAI API key not provided!')
            return
        self.client = OpenAI(api_key=api_key)  # Create OpenAI client
        
        # Create subscriber for snapshot images
        self.snapshot_sub = self.create_subscription(
            CompressedImage,
            '/final_photo/compressed',
            self.snapshot_callback,
            1)
        # Create subscriber for roast and caricature reset
        self.countdown_start_sub = self.create_subscription(
            Bool,
            'countdown_starting',
            self.countdown_starting_callback,
            10)
            
        # Create publishers for generated content
        self.roast_pub = self.create_publisher(
            Image, 'generated_roast', 10)
        self.caricature_pub = self.create_publisher(
            Image, 'generated_caricature', 10)
        self.processing_pub = self.create_publisher(
            Bool, 'processing_status', 10)
            
        self.bridge = CvBridge()
        self.get_logger().info('Media generation node initialized')
        
        # Create base directory for saving content
        self.save_dir = os.path.expanduser('~/roast_bot_media')
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'Media will be saved to: {self.save_dir}')
        
        # Add processing flag and display timeout tracking
        self.is_processing = False
        self.display_timeout = 30.0  # seconds
        self.last_display_time = None
        
        status_msg = Bool()
        status_msg.data = False
        self.processing_pub.publish(status_msg)
        self.last_caricature_time = 0.0
        
        # Show initial waiting images
        waiting_img = self.create_waiting_image("Waiting for image")
        self.roast_pub.publish(self.bridge.cv2_to_imgmsg(waiting_img, "bgr8"))
        self.caricature_pub.publish(self.bridge.cv2_to_imgmsg(waiting_img, "bgr8"))
        
    def encode_image_to_base64(self, cv_image):
        """Convert OpenCV image to base64 string."""
        _, buffer = cv2.imencode('.jpg', cv_image)
        return base64.b64encode(buffer).decode('utf-8')
        
    def create_text_image(self, text, width=1200, height=600):
        """Create an image with text."""
        # Create image with white background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Normalize text encoding
        if isinstance(text, bytes):
            text = text.decode('utf-8')
            

        text = text.replace('“', '\"').replace('”', '\"').replace('‘','\'').replace('’', '\'').replace('—','-').replace('…', '...')
        
        # Split text into lines (max 40 chars per line)
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= 40:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Add text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5 #0.7  # Slightly smaller text
        font_color = (255, 255, 255)  # White text
        thickness = 2  # Thinner text
        line_spacing = 55  # Slightly reduced spacing
        
        # Calculate total text height to center vertically
        total_text_height = len(lines) * line_spacing
        start_y = max(40, (height - total_text_height) // 2)
        
        # Add each line of text
        for i, line in enumerate(lines):
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Calculate x position to center text
            x_position = max(20, (width - text_width) // 2)
            y_position = start_y + i * line_spacing
            
            # Add text with anti-aliasing
            cv2.putText(image, line, (x_position, y_position), font, font_scale, 
                       font_color, thickness, cv2.LINE_AA)
        
        return image

    def create_waiting_image(self, text, width=800, height=400):
        """Create a waiting message image."""
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)
        thickness = 2
        
        # Get text size and center it
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x_position = (width - text_width) // 2
        y_position = (height + text_height) // 2
        
        cv2.putText(image, text, (x_position, y_position), font, font_scale, 
                   font_color, thickness, cv2.LINE_AA)
        return image

    def generate_roast(self, image_base64):
        """Generate a roast using GPT-4."""
        try:
            response = self.client.responses.create(
                model="gpt-4.1",
                instructions="You are a friendly AI roast master. You've been deployed at an academic AI conference to roast the attendees \
                                when they provide you with a photo. Use humor and wit, but keep it light-hearted and fun. When appropriate, incorporate inside jokes about AI, \
                              tech culture, academia, and references to robots in pop culture. Keep your responses concise and engaging, and be \
                              ready to roast groups of users at a time. Keep each roast to one short sentence.", 

                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Generate a roast of the people in this image based on on your instructions."
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        ],
                    }
                ],
            )
            return response.output_text
        except Exception as e:
            self.get_logger().error(f'Error generating roast: {str(e)}')
            return "Error generating roast. Please try again!"

    def generate_caricature(self, image_path):
        """Generate a caricature."""
        try:
            # Generate the caricature using the image file directly
            result = self.client.images.edit(
                model="gpt-image-1",
                image=open(image_path, "rb"),
                quality="medium",
                size="1024x1024",
                prompt="Generate a realistic and fun cartoon caricature, but don't make it unflattering (e.g., don't highlight flaws).\
                    Don't make people fatter or older than they are. Generally try to avoid fat, old or ugly caricatures"
            )
            # Get image data from response
            image_bytes = base64.b64decode(result.data[0].b64_json)
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.last_caricature_time = time.time()
            return image
            
        except Exception as e:
            self.get_logger().error(f'Error generating caricature: {str(e)}')
            return np.zeros((512, 512, 3), dtype=np.uint8)

    def create_session_folder(self):
        """Create a new folder for the current session using timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(self.save_dir, f'session_{timestamp}')
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def snapshot_callback(self, msg):
        # Check if image is too old
        current_time = self.get_clock().now()

        self.get_logger().info(f'Received snapshot image: {msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9}, current time: {current_time.nanoseconds/1e9}')
        
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        time_diff = (current_time - msg_time).nanoseconds / 1e9  # Convert to seconds
        
        if time_diff > 0.5:
            self.get_logger().warning(f'Received old image (age: {time_diff:.2f}s), skipping processing')
            return
            
        if self.is_processing:
            self.get_logger().warning('Still processing previous image, skipping new image')
            return
        
        t1 = time.time()
        if (t1 - self.last_caricature_time) < 10:
            self.get_logger().warning('Caricature drawing on cooldown')
            return
        
        
        # Publish processing status
        status_msg = Bool()
        status_msg.data = True
        self.processing_pub.publish(status_msg)
        self.is_processing = True
        self.get_logger().info(f'Received new image (age: {time_diff:.2f}s), starting processing...')
        
        try:
            # Create a new session folder
            session_dir = self.create_session_folder()
            
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Save original image
            original_path = os.path.join(session_dir, 'original.jpg')
            cv2.imwrite(original_path, cv_image)
            self.get_logger().info(f'Saved original image to: {original_path}')
            
            # Show both waiting messages immediately
            roast_waiting = self.create_waiting_image("Thinking of roast...")
            caricature_waiting = self.create_waiting_image("Drawing caricature...")
            
            self.roast_pub.publish(
                self.bridge.cv2_to_imgmsg(roast_waiting, "bgr8"))
            self.caricature_pub.publish(
                self.bridge.cv2_to_imgmsg(caricature_waiting, "bgr8"))
            
            # Generate and save roast text with timing
            self.get_logger().info('Generating roast text...')
            roast_start_time = time.time()
            
            image_base64 = self.encode_image_to_base64(cv_image)
            roast_text = self.generate_roast(image_base64)
            
            roast_duration = time.time() - roast_start_time
            self.get_logger().info(f'Roast generated in {roast_duration:.2f} seconds: ' + roast_text[:50] + '...')
            
            # Create and publish roast image
            roast_image = self.create_text_image(roast_text)
            self.roast_pub.publish(
                self.bridge.cv2_to_imgmsg(roast_image, "bgr8"))
            self.last_display_time = time.time()
            
            # Save roast content
            roast_path = os.path.join(session_dir, 'roast.jpg')
            cv2.imwrite(roast_path, roast_image)
            
            text_path = os.path.join(session_dir, 'roast.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(roast_text)
            
            # Generate and save caricature with timing
            self.get_logger().info('Generating caricature...')
            caricature_start_time = time.time()
            
            caricature = self.generate_caricature(original_path)
            
            caricature_duration = time.time() - caricature_start_time
            self.get_logger().info(f'Caricature generated in {caricature_duration:.2f} seconds')
            
            # Save and publish caricature
            caricature_path = os.path.join(session_dir, 'caricature.jpg')
            cv2.imwrite(caricature_path, caricature)
            
            self.caricature_pub.publish(
                self.bridge.cv2_to_imgmsg(caricature, "bgr8"))
            self.last_display_time = time.time()  # Update time after both images are published
            
            total_duration = roast_duration + caricature_duration
            self.get_logger().info(f'Processing complete! Total time: {total_duration:.2f} seconds')
            
        except Exception as e:
            self.get_logger().error(f'Error during processing: {str(e)}')
            
        finally:
            # Publish processing status
            status_msg = Bool()
            status_msg.data = False
            self.processing_pub.publish(status_msg)
            self.is_processing = False
    
    def countdown_starting_callback(self, msg: Bool):
        """Clear display only when a new image countdown is starting."""
        if msg.data:  # Only act on True signal
            waiting_img = self.create_waiting_image("Waiting for image")
            self.roast_pub.publish(self.bridge.cv2_to_imgmsg(waiting_img, "bgr8"))
            self.caricature_pub.publish(self.bridge.cv2_to_imgmsg(waiting_img, "bgr8"))
            self.get_logger().info("Resetting display to waiting state.")

def main(args=None):
    rclpy.init(args=args)
    node = MediaGenerationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
