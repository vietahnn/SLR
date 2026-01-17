"""
Combine frames into video - Simple version
"""

import cv2
from pathlib import Path


def combine_frames_to_video(frames_dir, output_path="output.mp4", fps=30):
    """
    Ghép các frames thành video
    
    Args:
        frames_dir: Đường dẫn folder chứa frames
        output_path: Tên file video output
        fps: Số frames mỗi giây
    """
    frames_dir = Path(frames_dir)
    
    # Lấy tất cả file jpg
    frame_files = sorted(frames_dir.glob("*.jpg"))
    
    if not frame_files:
        print(f"Không tìm thấy file jpg trong {frames_dir}")
        return
    
    print(f"Tìm thấy {len(frame_files)} frames")
    
    # Đọc frame đầu tiên để lấy kích thước
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]
    print(f"Kích thước frame: {width}x{height}")
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Ghi từng frame vào video
    print(f"Đang tạo video {output_path}...")
    for i, frame_path in enumerate(frame_files, 1):
        frame = cv2.imread(str(frame_path))
        out.write(frame)
        if i % 10 == 0:
            print(f"  Đã ghi {i}/{len(frame_files)} frames")
    
    out.release()
    print(f"Hoàn thành! Video: {output_path}")
    print(f"Thời lượng: {len(frame_files) / fps:.2f} giây")


if __name__ == "__main__":
    # Folder mặc định
    frames_folder = r"archive\preprocessing\test\frames\accident\00625"
    
    # Tạo video
    combine_frames_to_video(frames_folder, output_path="accident_00625.mp4", fps=5)
