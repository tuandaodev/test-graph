import cv2
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.models import resnet50
from torchvision import transforms
import torch.nn as nn

class SceneGraphVisualizer:
    def __init__(self):
        # Initialize YOLO model for object detection
        self.yolo_model = YOLO('yolov8x.pt')
        
        # Initialize ResNet for feature extraction
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor.eval()
        
        # Transform for ResNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Relationship predictor threshold
        self.relation_threshold = 0.5
        
    def detect_objects(self, frame):
        """Detect objects in a frame using YOLO"""
        results = self.yolo_model(frame)
        
        # Extract bounding boxes, labels and confidence scores
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        names = results[0].names
        
        detections = []
        for box, label, conf in zip(boxes, labels, confidences):
            detections.append({
                'box': box,
                'label': names[int(label)],
                'confidence': conf
            })
        
        return detections
    
    def predict_relationships(self, obj1, obj2):
        """Predict relationship between two objects based on spatial relationships"""
        box1 = obj1['box']
        box2 = obj2['box']
        
        # Calculate center points
        center1 = [(box1[0] + box1[2])/2, (box1[1] + box1[3])/2]
        center2 = [(box2[0] + box2[2])/2, (box2[1] + box2[3])/2]
        
        # Calculate spatial relationships
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Define relationships based on spatial positions
        relationships = []
        
        if abs(dx) > abs(dy):
            if dx > 0:
                relationships.append(('right of', 0.8))
            else:
                relationships.append(('left of', 0.8))
        else:
            if dy > 0:
                relationships.append(('below', 0.8))
            else:
                relationships.append(('above', 0.8))
                
        if distance < 100:  # Close objects
            relationships.append(('near', 0.9))
        
        return relationships
    
    def build_scene_graph(self, frame):
        """Build scene graph from frame"""
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (objects)
        for i, det in enumerate(detections):
            G.add_node(i, 
                      label=f"{det['label']}\n{det['confidence']:.2f}",
                      pos=((det['box'][0] + det['box'][2])/2, 
                          (det['box'][1] + det['box'][3])/2))
        
        # Add edges (relationships)
        for i, obj1 in enumerate(detections):
            for j, obj2 in enumerate(detections):
                if i < j:  # Avoid duplicate relationships
                    relationships = self.predict_relationships(obj1, obj2)
                    for rel, conf in relationships:
                        if conf > self.relation_threshold:
                            G.add_edge(i, j, label=rel)
        
        return G, detections
    
    def visualize_scene_graph(self, frame, save_path=None):
        """Visualize scene graph with object detection and relationships"""
        # Build scene graph
        G, detections = self.build_scene_graph(frame)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot original frame with detections
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for det in detections:
            box = det['box']
            ax1.add_patch(plt.Rectangle((box[0], box[1]), 
                                      box[2] - box[0], 
                                      box[3] - box[1], 
                                      fill=False, 
                                      color='red'))
            ax1.text(box[0], box[1], 
                    f"{det['label']} {det['confidence']:.2f}", 
                    color='white', 
                    bbox=dict(facecolor='red', alpha=0.5))
        ax1.set_title('Object Detection')
        
        # Plot scene graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax2, with_labels=True, 
               node_color='lightblue', 
               node_size=2000, 
               font_size=8)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        ax2.set_title('Scene Graph')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def process_video(video_path, output_dir):
    """Process video and create scene graphs for frames"""
    cap = cv2.VideoCapture(video_path)
    visualizer = SceneGraphVisualizer()
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % 10 == 0:  # Process every 10th frame
            output_path = f"{output_dir}/frame_{frame_count}.png"
            visualizer.visualize_scene_graph(frame, output_path)
            
        frame_count += 1
    
    cap.release()

if __name__ == "__main__":
    video_path = "data/video/VIRAT_S_010204_05_000856_000890.mp4"
    output_dir = "scene_graph_output"
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video
    process_video(video_path, output_dir)