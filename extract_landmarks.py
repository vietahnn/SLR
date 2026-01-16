"""
Script to extract landmarks from frames using MediaPipe 0.10.31 tasks API
Extracts: Hand landmarks (both hands), Lips landmarks, and 6 basic Pose points
Downloads required models automatically
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from tqdm import tqdm


class LandmarkExtractor:
    def __init__(self):
        """Initialize MediaPipe task models with automatic download"""
        # Create models directory
        self.models_dir = Path("mediapipe_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Download models if not exist
        self._download_models()
        
        # Initialize Hand Landmarker
        hand_model_path = str(self.models_dir / "hand_landmarker.task")
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hand_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
        
        # Initialize Face Landmarker
        face_model_path = str(self.models_dir / "face_landmarker.task")
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)
        
        # Initialize Pose Landmarker
        pose_model_path = str(self.models_dir / "pose_landmarker_heavy.task")
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=pose_model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        
        # Lips landmarks indices (10 key points)
        # Upper lip (outer): 0, 37, 267
        # Lower lip (outer): 17, 84, 314
        # Mouth corners: 61, 291
        # Mouth opening: 13, 14
        self.lips_indices = [0, 37, 267, 17, 84, 314, 61, 291, 13, 14]
        
        # Pose indices: shoulders, elbows, wrists
        self.pose_indices = [11, 12, 13, 14, 15, 16]
    
    def _download_models(self):
        """Download MediaPipe models if they don't exist"""
        models = {
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        }
        
        for model_name, url in models.items():
            model_path = self.models_dir / model_name
            if not model_path.exists():
                print(f"Downloading {model_name}...")
                try:
                    urllib.request.urlretrieve(url, model_path)
                    print(f"Downloaded {model_name}")
                except Exception as e:
                    print(f"Error downloading {model_name}: {e}")
                    raise
    
    def extract_landmarks_from_frame(self, image_path):
        """
        Extract landmarks from a single frame
        
        Returns:
            dict with keys: 'left_hand', 'right_hand', 'lips', 'pose'
        """
        # Read image with Unicode path support
        try:
            # Method 1: Direct read with absolute path
            abs_path = str(Path(image_path).absolute())
            image = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            
            # Method 2: Use numpy for Unicode path support if method 1 fails
            if image is None:
                with open(image_path, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Cannot read image: {image_path} - Error: {e}")
            return None
        
        if image is None:
            print(f"Cannot read image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        landmarks = {
            'left_hand': None,
            'right_hand': None,
            'lips': None,
            'pose': None
        }
        
        # Extract hand landmarks
        hand_result = self.hand_landmarker.detect(mp_image)
        if hand_result.hand_landmarks and hand_result.handedness:
            for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                hand_type = handedness[0].category_name  # "Left" or "Right"
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
                
                if hand_type == "Left":
                    landmarks['left_hand'] = landmarks_list
                else:
                    landmarks['right_hand'] = landmarks_list
        
        # Extract face/lips landmarks
        face_result = self.face_landmarker.detect(mp_image)
        if face_result.face_landmarks:
            face_lm = face_result.face_landmarks[0]
            lips = []
            for idx in self.lips_indices:
                if idx < len(face_lm):
                    lm = face_lm[idx]
                    lips.append([lm.x, lm.y, lm.z])
            if lips:
                landmarks['lips'] = lips
        
        # Extract pose landmarks
        pose_result = self.pose_landmarker.detect(mp_image)
        if pose_result.pose_landmarks:
            pose_lm = pose_result.pose_landmarks[0]
            pose_points = []
            for idx in self.pose_indices:
                if idx < len(pose_lm):
                    lm = pose_lm[idx]
                    # Add visibility (use presence from world landmarks if available)
                    visibility = 1.0
                    if pose_result.pose_world_landmarks:
                        world_lm = pose_result.pose_world_landmarks[0][idx]
                        visibility = world_lm.visibility if hasattr(world_lm, 'visibility') else 1.0
                    pose_points.append([lm.x, lm.y, lm.z, visibility])
            if len(pose_points) == 6:
                landmarks['pose'] = pose_points
        
        return landmarks
    
    def extract_landmarks_from_video(self, frames_dir):
        """
        Extract landmarks from all frames of a video
        
        Args:
            frames_dir: Directory containing video frames
            
        Returns:
            list of dicts, each dict contains landmarks of one frame
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            print(f"Directory not found: {frames_dir}")
            return None
        
        # Get list of frame files and sort
        frame_files = sorted(frames_dir.glob("*.jpg"), 
                           key=lambda x: int(x.stem.split('_')[-1]))
        
        if not frame_files:
            print(f"No frames found in: {frames_dir}")
            return None
        
        video_landmarks = []
        for frame_file in frame_files:
            landmarks = self.extract_landmarks_from_frame(frame_file)
            if landmarks:
                video_landmarks.append(landmarks)
        
        return video_landmarks
    
    def process_dataset(self, base_dir, output_dir, split='train'):
        """
        Process an entire split (train/test/val) of the dataset
        
        Args:
            base_dir: Base directory containing dataset
            output_dir: Directory to save landmarks
            split: 'train', 'test', or 'val'
        """
        base_dir = Path(base_dir)
        output_dir = Path(output_dir)
        frames_dir = base_dir / split / 'frames'
        
        if not frames_dir.exists():
            print(f"Frames directory not found: {frames_dir}")
            return
        
        # Create output directory (no class subfolders)
        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of classes
        classes = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
        
        print(f"\n{'='*60}")
        print(f"Processing split: {split.upper()}")
        print(f"Number of classes: {len(classes)}")
        print(f"{'='*60}\n")
        
        total_videos = 0
        successful_videos = 0
        
        for class_dir in tqdm(classes, desc=f"Processing {split}"):
            class_name = class_dir.name
            
            # Get list of videos in this class
            video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
            
            for video_dir in video_dirs:
                total_videos += 1
                video_id = video_dir.name
                
                # Extract landmarks
                video_landmarks = self.extract_landmarks_from_video(video_dir)
                
                if video_landmarks:
                    # Flatten landmarks for each frame
                    flattened_frames = []
                    for frame in video_landmarks:
                        frame_keypoints = []
                        
                        # Left hand (21 points x 3 coords = 63 values)
                        if frame['left_hand']:
                            for point in frame['left_hand']:
                                frame_keypoints.extend(point)
                        else:
                            frame_keypoints.extend([0.0] * 63)
                        
                        # Right hand (21 points x 3 coords = 63 values)
                        if frame['right_hand']:
                            for point in frame['right_hand']:
                                frame_keypoints.extend(point)
                        else:
                            frame_keypoints.extend([0.0] * 63)
                        
                        # Lips (10 points x 3 coords = 30 values)
                        if frame['lips']:
                            for point in frame['lips']:
                                frame_keypoints.extend(point)
                        else:
                            frame_keypoints.extend([0.0] * 30)
                        
                        # Pose (6 points x 4 coords = 24 values)
                        if frame['pose']:
                            for point in frame['pose']:
                                frame_keypoints.extend(point)
                        else:
                            frame_keypoints.extend([0.0] * 24)
                        
                        flattened_frames.append(frame_keypoints)
                    
                    # Create DataFrame with flattened keypoints
                    df = pd.DataFrame(flattened_frames)
                    
                    # Save to Parquet file with class_video naming
                    output_file = split_output_dir / f"{class_name}_{video_id}.parquet"
                    df.to_parquet(output_file, index=False, compression='snappy')
                    successful_videos += 1
        
        print(f"\n{'='*60}")
        print(f"Completed split {split.upper()}")
        print(f"Total videos: {total_videos}")
        print(f"Successfully processed: {successful_videos}")
        print(f"Failed: {total_videos - successful_videos}")
        print(f"{'='*60}\n")  
    
    def process_all_splits(self, base_dir, output_dir):
        """
        Process entire dataset (train, test, val)
        """
        print("\n" + "="*60)
        print("STARTING LANDMARK EXTRACTION FOR ENTIRE DATASET")
        print("="*60)
        
        for split in ['train', 'test', 'val']:
            self.process_dataset(base_dir, output_dir, split)
        
        print("\n" + "="*60)
        print("COMPLETED LANDMARK EXTRACTION FOR ENTIRE DATASET")
        print("="*60 + "\n")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()


def main():
    """Main function to run the script"""
    # Paths
    base_dir = r"c:\Users\dovie\OneDrive\Desktop\Việt anh\archive\preprocessing"
    output_dir = r"c:\Users\dovie\OneDrive\Desktop\Việt anh\landmarks"
    
    # Initialize extractor
    print("Initializing landmark extractor...")
    extractor = LandmarkExtractor()
    
    # Process entire dataset
    extractor.process_all_splits(base_dir, output_dir)
    
    print("Landmark extraction completed successfully!")


if __name__ == "__main__":
    main()
