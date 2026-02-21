import platform
import time
import logging
import signal
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Configure Production Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("bedside_monitor.log"), # General system logs
        logging.StreamHandler(sys.stdout)           # Prints to terminal
    ]
)

import pygame
import pygame.camera

# --- Smart TFLite Import ---
try:
    import tflite_runtime.interpreter as tflite
    logging.info("Loaded lightweight tflite-runtime for ARM.")
except ImportError:
    import tensorflow.lite as tflite
    logging.info("Loaded full tensorflow lite interpreter for x86.")

# --- Configuration ---
MODEL_PATH = "ssd_mobilenet_v2.tflite"
PATIENCE_SECONDS = 300  # 300 seconds = 5 minutes absence threshold
CONFIDENCE_THRESHOLD = 0.55
PERSON_CLASS_ID = 0 
MIN_PERSON_AREA = 0.10  # Patient must take up at least 10% of the frame (ignores TVs/Photos)

# Global run flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    logging.info("Shutdown signal received. Closing monitor safely...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Hardware Detection ---
def is_raspberry_pi():
    machine = platform.machine().lower()
    return 'arm' in machine or 'aarch64' in machine


# ==========================================
# 🚨 THE ALERT SYSTEM 🚨
# ==========================================
def trigger_alert(elapsed_seconds):
    """
    This function fires exactly when the patient has been missing for 5 minutes.
    This is where your logs, alarms, and notifications happen!
    """
    # 1. Log the critical error to the system console
    logging.critical(f"🚨 ALERT: Patient absent for {int(elapsed_seconds)} seconds! 🚨")
    
    # 2. Write to a permanent 'Incident Report' file
    with open("INCIDENT_ALERTS.txt", "a") as f:
        f.write(f"[{time.ctime()}] ALARM TRIGGERED: Patient missing for {int(elapsed_seconds/60)} minutes.\n")
    
    # 3. FUTURE EXPANSION: Here is where you will add code to notify the nurses.
    # For example, sending an SMS using Twilio or hitting a hospital API:
    # requests.post("http://nurse-station-api.local/alert", json={"bed": 12, "status": "empty"})


# --- Camera Abstraction ---
class PiCam:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        
    def get_frame(self):
        arr = self.picam2.capture_array()
        return Image.fromarray(arr)

    def close(self):
        self.picam2.stop()

class PygameCam:
    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("OpenCV could not find a webcam!")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return Image.new("RGB", (640, 480), "black")
        frame_rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def close(self):
        self.cap.release()

# --- Main Engine ---
def main():
    global running
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("AI Bedside Monitor (Production)")
    font = ImageFont.load_default()

    try:
        if is_raspberry_pi():
            logging.info("Hardware: ARM Detected. Initializing PiCamera2...")
            camera = PiCam()
        else:
            logging.info("Hardware: x86 Detected. Initializing OpenCV WebCam...")
            camera = PygameCam()

        logging.info("Loading Inference Engine...")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)

        absence_start = 0
        alert_triggered = False

        logging.info("System Online. Active Monitoring Started.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            pil_image = camera.get_frame()
            orig_width, orig_height = pil_image.size
            
            resized_image = pil_image.resize((width, height))
            input_data = np.expand_dims(resized_image, axis=0)
            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = np.uint8(input_data)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]

            draw = ImageDraw.Draw(pil_image)
            person_detected = False

            for i in range(len(scores)):
                if scores[i] > CONFIDENCE_THRESHOLD and int(classes[i]) == PERSON_CLASS_ID:
                    ymin, xmin, ymax, xmax = boxes[i]
                    box_area = (xmax - xmin) * (ymax - ymin)
                    
                    # Ignore tiny detections (like a photo or distant person)
                    if box_area >= MIN_PERSON_AREA:
                        person_detected = True
                        left, right = int(xmin * orig_width), int(xmax * orig_width)
                        top, bottom = int(ymin * orig_height), int(ymax * orig_height)
                        
                        draw.rectangle([left, top, right, bottom], outline="lime", width=4)
                        draw.text((left, top - 15), f"Patient: {int(scores[i]*100)}%", fill="lime", font=font)
                        break 

            # Timer Logic
            if person_detected:
                if absence_start != 0:
                    logging.info("Patient detected. Resetting timer.")
                absence_start = 0 
                alert_triggered = False # Reset the alert throttle
            else:
                if absence_start == 0:
                    absence_start = time.time()
                    logging.warning("Patient lost from view. Starting timer...")
                
                elapsed = time.time() - absence_start
                draw.text((10, 10), f"WARNING: Target Lost ({int(elapsed)}s)", fill="orange", font=font)
                
                if elapsed > PATIENCE_SECONDS:
                    draw.rectangle([0, 0, orig_width, orig_height], outline="red", width=15)
                    draw.text((20, 40), "ALERT: BED EMPTY", fill="red", font=font)
                    
                    # 🔥 FIRE THE ALERT FUNCTION ONCE 🔥
                    if not alert_triggered:
                        trigger_alert(elapsed)
                        alert_triggered = True

            # Display
            mode = pil_image.mode
            size = pil_image.size
            data = pil_image.tobytes()
            py_image = pygame.image.fromstring(data, size, mode)
            screen.blit(py_image, (0, 0))
            pygame.display.flip()

    except Exception as e:
        logging.error(f"Fatal System Error: {e}")
    finally:
        logging.info("Cleaning up hardware resources...")
        try:
            camera.close()
        except:
            pass
        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()
