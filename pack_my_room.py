# pack_my_room.py

import cv2
import tensorflow as tf
import numpy as np
from py3dbp import Packer, Bin, Item
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

# Path to the pre-trained object detection model.
# Download from: http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# Unzip it and place the 'saved_model' directory in your project folder.
MODEL_PATH = './ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'

# Path to the COCO dataset labels file.
# Create a file 'coco.names' and paste the class names from:
# https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
LABEL_PATH = './coco.names'

# Reference object details for dimension estimation.
# We'll use a standard A4 paper (21.0 cm width).
REFERENCE_OBJECT_LABEL = 'book' # We'll pretend a book is our A4 paper.
REFERENCE_OBJECT_REAL_WIDTH_CM = 21.0

class RoomPacker:
    """
    A class to detect objects in an image, estimate their dimensions,
    and find the optimal packing arrangement in a given container.
    """

    def __init__(self, model_path, label_path):
        """
        Initializes the object detector.
        """
        print("Loading object detection model...")
        self.model = tf.saved_model.load(model_path)
        self.classes = self._load_class_names(label_path)
        print("Model loaded successfully.")

    def _load_class_names(self, file_path):
        """Loads class names from a file."""
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def detect_objects(self, image_path):
        """
        Detects objects in the provided image.

        Args:
            image_path (str): The path to the input image.

        Returns:
            A tuple containing:
            - list: A list of detected items, each a dictionary with 'name', 'box', and 'score'.
            - ndarray: The image with bounding boxes drawn on it.
        """
        print(f"Detecting objects in '{image_path}'...")
        image_np = cv2.imread(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        detected_items = []
        im_height, im_width, _ = image_np.shape

        for i in range(detections['num_detections']):
            score = detections['detection_scores'][i]
            if score > 0.5: # Confidence threshold
                class_id = detections['detection_classes'][i]
                class_name = self.classes[class_id]
                box = detections['detection_boxes'][i]

                # Convert box coordinates to pixels
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                              ymin * im_height, ymax * im_height)
                
                detected_items.append({
                    'name': class_name,
                    'box': [int(left), int(top), int(right), int(bottom)],
                    'score': score
                })

                # Draw the box on the image for visualization
                cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                cv2.putText(image_np, class_name, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Found {len(detected_items)} objects.")
        return detected_items, image_np

    def estimate_dimensions(self, detected_items):
        """
        Estimates the 3D dimensions of objects based on a reference object.

        Args:
            detected_items (list): The list of items from detect_objects.

        Returns:
            list: A list of py3dbp.Item objects with estimated dimensions.
        """
        print("Estimating object dimensions...")
        # Find the reference object in the detected items
        reference_item = next((item for item in detected_items if item['name'] == REFERENCE_OBJECT_LABEL), None)

        if not reference_item:
            raise ValueError(f"Reference object '{REFERENCE_OBJECT_LABEL}' not found in image.")

        # Calculate pixel-to-cm ratio
        ref_box = reference_item['box']
        ref_pixel_width = ref_box[2] - ref_box[0]
        pixel_to_cm_ratio = REFERENCE_OBJECT_REAL_WIDTH_CM / ref_pixel_width
        print(f"Pixel-to-cm ratio: {pixel_to_cm_ratio:.2f} (based on '{REFERENCE_OBJECT_LABEL}')")

        packer_items = []
        for i, item in enumerate(detected_items):
            # Skip the reference object itself from packing
            if item == reference_item:
                continue
            
            box = item['box']
            pixel_width = box[2] - box[0]
            pixel_height = box[3] - box[1]

            # Estimate real-world dimensions
            # We assume depth is half the width as a rough guess
            width = round(pixel_width * pixel_to_cm_ratio)
            height = round(pixel_height * pixel_to_cm_ratio)
            depth = round(width * 0.5)

            # py3dbp requires dimensions to be positive
            if width > 0 and height > 0 and depth > 0:
                packer_items.append(
                    Item(f"{item['name']}_{i}", width, height, depth, 1) # name, W, H, D, weight
                )
                print(f"  - Estimated {item['name']}: {width}cm (W) x {height}cm (H) x {depth}cm (D)")
        
        return packer_items

    def pack_items(self, items, bin_width, bin_height, bin_depth):
        """
        Runs the 3D bin packing algorithm.

        Args:
            items (list): A list of py3dbp.Item objects.
            bin_width, bin_height, bin_depth (int): Dimensions of the container.

        Returns:
            A py3dbp.Packer object containing the packing results.
        """
        print("\nStarting packing simulation...")
        packer = Packer()
        
        # Add the container (bin)
        packer.add_bin(Bin('moving_truck', bin_width, bin_height, bin_depth, 1000)) # name, W, H, D, max_weight

        # Add the items
        for item in items:
            packer.add_item(item)

        # Pack!
        packer.pack()
        print("Packing complete.")
        return packer

    def visualize_packing(self, packer):
        """
        Creates a 3D visualization of the packed bin.

        Args:
            packer (py3dbp.Packer): The packer object after packing.
        """
        print("Generating 3D visualization...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.get_cmap('tab20', len(packer.bins[0].items))
        
        # Plot each fitted item
        for i, item in enumerate(packer.bins[0].items):
            pos = item.position
            dim = item.get_dimension()
            
            # Create vertices for a 3D cube
            x, y, z = pos
            dx, dy, dz = dim
            
            # The box is drawn by plotting the 12 edges
            edges = [
                ((x,y,z), (x+dx,y,z)), ((x,y,z), (x,y+dy,z)), ((x,y,z), (x,y,z+dz)),
                ((x+dx,y+dy,z), (x,y+dy,z)), ((x+dx,y+dy,z), (x+dx,y,z)), ((x+dx,y+dy,z), (x+dx,y+dy,z+dz)),
                ((x+dx,y,z+dz), (x,y,z+dz)), ((x+dx,y,z+dz), (x+dx,y,z)), ((x+dx,y,z+dz), (x+dx,y+dy,z+dz)),
                ((x,y+dy,z+dz), (x,y,z+dz)), ((x,y+dy,z+dz), (x,y+dy,z)), ((x,y+dy,z+dz), (x+dx,y+dy,z+dz))
            ]

            for edge in edges:
                ax.plot3D(*zip(*edge), color=colors(i))

            # Add item label
            ax.text(x, y, z, item.name, fontsize=8)

        # Set plot limits to the bin dimensions
        bin_dim = packer.bins[0].get_dimension()
        ax.set_xlim([0, bin_dim[0]])
        ax.set_ylim([0, bin_dim[1]])
        ax.set_zlim([0, bin_dim[2]])
        
        ax.set_xlabel('Width (cm)')
        ax.set_ylabel('Height (cm)')
        ax.set_zlabel('Depth (cm)')

        plt.title('3D Packing Visualization')
        plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # 1. Define user inputs
    IMAGE_FILE = 'room_scene.jpg' # <-- PUT YOUR IMAGE FILENAME HERE
    # Dimensions of your packing space in CM
    CONTAINER_WIDTH_CM = 200  # 2 meters
    CONTAINER_HEIGHT_CM = 200 # 2 meters
    CONTAINER_DEPTH_CM = 400  # 4 meters

    # Create an instance of our packer app
    app = RoomPacker(model_path=MODEL_PATH, label_path=LABEL_PATH)

    # 2. Detect objects in the scene
    detected_objects, output_image = app.detect_objects(image_path=IMAGE_FILE)
    
    # Save the image with detections drawn on it
    cv2.imwrite('detections_output.jpg', output_image)
    print("Saved detection image to 'detections_output.jpg'")

    try:
        # 3. Estimate the dimensions of the detected objects
        items_to_pack = app.estimate_dimensions(detected_objects)

        if not items_to_pack:
            print("\nNo valid items to pack after estimation.")
        else:
            # 4. Run the packing algorithm
            packer_result = app.pack_items(
                items=items_to_pack,
                bin_width=CONTAINER_WIDTH_CM,
                bin_height=CONTAINER_HEIGHT_CM,
                bin_depth=CONTAINER_DEPTH_CM
            )

            # 5. Report and visualize the results
            bin = packer_result.bins[0]
            print(f"\n--- Packing Results ---")
            print(f"Fitted {len(bin.items)} out of {len(items_to_pack)} items.")
            
            for item in bin.items:
                print(f"  - Item: {item.name}, Position: {item.position}, Rotation: {item.rotation_type}")

            if bin.unfitted_items:
                print("\nUnfitted items:")
                for item in bin.unfitted_items:
                    print(f"  - {item.name}")
            
            app.visualize_packing(packer_result)

    except ValueError as e:
        print(f"\nError: {e}")
