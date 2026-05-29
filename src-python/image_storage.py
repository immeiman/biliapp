"""
image_storage.py

Manage captured image files on disk.
Handles saving, organizing, and file operations for captured images.
"""

import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


class ImageStorage:
    """
    Manage captured images on disk with date-based directory structure.
    Default: data/captures/YYYY-MM-DD/image_HHMMSS_XXXXXX.jpg
    """

    def __init__(self, base_dir: str = "data/captures"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_today_dir(self) -> Path:
        """Get or create today's capture directory."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_dir = self.base_dir / today
        today_dir.mkdir(parents=True, exist_ok=True)
        return today_dir

    def save_image(
        self,
        image_bgr: np.ndarray,
        prefix: str = "image",
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Save image with timestamp-based filename.
        
        Args:
            image_bgr: Image in BGR format (from cv2)
            prefix: Filename prefix (default: 'image')
            timestamp: datetime object (default: now)
        
        Returns:
            (success: bool, filepath: str)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()

            today_dir = self.get_today_dir()

            free_bytes = shutil.disk_usage(str(today_dir)).free
            if free_bytes < 10 * 1024 * 1024:
                return False, "Disk hampir penuh — ruang kosong < 10 MB"

            # Create filename: prefix_HHMMSS_microseconds.jpg
            time_str = timestamp.strftime("%H%M%S")
            micros = timestamp.microsecond // 1000  # ms
            filename = f"{prefix}_{time_str}_{micros:03d}.jpg"
            filepath = today_dir / filename

            # Save image
            success = cv2.imwrite(str(filepath), image_bgr)
            
            if success:
                return True, str(filepath)
            else:
                return False, str(filepath)

        except Exception as e:
            return False, f"Error saving image: {str(e)}"

    def load_image(self, filepath: str) -> Optional[np.ndarray]:
        """Load image from filepath (returns BGR)."""
        try:
            img = cv2.imread(filepath)
            return img
        except Exception:
            return None

    def delete_image(self, filepath: str) -> bool:
        """Delete image file."""
        try:
            Path(filepath).unlink()
            return True
        except Exception:
            return False

    def list_today_images(self) -> list:
        """List all image files captured today."""
        try:
            today_dir = self.get_today_dir()
            files = sorted(today_dir.glob("*.jpg"))
            return [str(f) for f in files]
        except Exception:
            return []

    def get_capture_count(self) -> int:
        """Get total number of captured images."""
        try:
            all_files = list(self.base_dir.rglob("*.jpg"))
            return len(all_files)
        except Exception:
            return 0

    def cleanup_old_images(self, days_to_keep: int = 7) -> int:
        """
        Delete images older than days_to_keep.
        
        Returns:
            Number of images deleted
        """
        from datetime import timedelta
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0

            for jpeg_file in self.base_dir.rglob("*.jpg"):
                file_mtime = datetime.fromtimestamp(jpeg_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    jpeg_file.unlink()
                    deleted_count += 1

            return deleted_count

        except Exception:
            return 0
