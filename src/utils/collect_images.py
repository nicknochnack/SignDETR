import cv2
import numpy as np
import uuid
import time 
import os 
from setup import get_classes
from logger import logger

classes = get_classes() 

class CaptureImages(): 
    def __init__(self, path: str, classes: dict, camera_id: int) -> None: 
        self.cap = cv2.VideoCapture(camera_id) 
        self.path = path 
        self.classes = classes
        
        # Initialize logger and show banner
        logger.print_banner()
        logger.capture("Image capture system initialized")
        
        # Verify camera connection
        if not self.cap.isOpened():
            logger.capture_error("Camera", f"Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        else:
            logger.success(f"Camera {camera_id} connected successfully")
        
        # Ensure output directory exists
        os.makedirs(self.path, exist_ok=True)
        logger.info(f"Output directory: {self.path}")

    def capture(self, class_name: str) -> bool:     
        try: 
            ret, frame = self.cap.read() 
            raw_frame = frame.copy()
            if not ret:
                raise Exception("Failed to read from camera")
                
            image = cv2.putText(frame, f'Capturing {class_name}', (0,100), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Image Capture', image)
            
            # Generate unique filename
            filename = f'{class_name}-{uuid.uuid1()}.jpg'
            filepath = os.path.join(self.path, filename)
            cv2.imwrite(filepath, raw_frame)
            
            if cv2.waitKey(1) & 0xFF==ord('q'):
                logger.warning("Quit key pressed - stopping capture")
                return False
                
            return True
            
        except Exception as e: 
            logger.capture_error(class_name, str(e))
            return False

    def run(self, sleep_time: int = 1, num_images: int = 10):
        # Display session information
        logger.capture_session_start(self.classes, num_images, sleep_time)
        
        total_captured = 0
        
        for class_idx, img_class in enumerate(self.classes): 
            logger.capture_class_start(img_class, num_images)
            
            # Create progress bar for this class
            with logger.create_capture_progress(num_images, img_class) as progress:
                class_task = progress.add_task(f"Capturing {img_class}", total=num_images)
                
                class_captured = 0
                for idx in range(num_images): 
                    success = self.capture(img_class)
                    
                    if success:
                        class_captured += 1
                        total_captured += 1
                        logger.capture_success(img_class, idx + 1)
                        progress.update(class_task, advance=1)
                    else:
                        logger.capture_error(img_class, f"Image {idx + 1}")
                        # Still advance progress to continue
                        progress.update(class_task, advance=1)
                    
                    time.sleep(sleep_time)
                
                # Show completion for this class
                logger.success(f"Completed {img_class}: {class_captured}/{num_images} images captured")
        
        # Show session completion
        logger.capture_session_complete(total_captured, len(self.classes))
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")

if __name__ == '__main__': 
    cap = CaptureImages('./data/test', classes, 0) 
    cap.run(num_images=25)
