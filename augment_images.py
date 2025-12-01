"""
Professional Image Data Augmentation Script
==========================================
Performs comprehensive augmentation on images in a directory.

Usage:
    python augment_images.py --input ./original_images --output ./augmented_images --per-image 5

Features:
    - Geometric: rotation, flipping, scaling, shearing, translation
    - Color: brightness, contrast, saturation, hue
    - Quality: blur, sharpening, noise
    - Advanced: elastic transform, pixel dropout, channel shifts

Examples:
    # Basic usage (creates 5 versions of each image)
    python augment_images.py -i ./original_folder -o ./augmented_folder

    # Create 10 versions of each image
    python augment_images.py -i ./images -o ./augmented -n 10

    # Create 20 versions with reproducible results
    python augment_images.py -i ./data -o ./aug_data -n 20 --seed 42
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple
import random


class ImageAugmentor:
    """Professional image augmentation with multiple transformation techniques."""

    def __init__(self, seed: int = None):
        """Initialize augmentor with optional random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def rotate(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """Rotate image by random or specified angle."""
        if angle is None:
            angle = random.uniform(-180, 180)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def flip(self, image: np.ndarray, mode: int = None) -> np.ndarray:
        """Flip image horizontally, vertically, or both."""
        if mode is None:
            mode = random.choice([0, 1, -1])  # 0=vertical, 1=horizontal, -1=both
        return cv2.flip(image, mode)

    def adjust_brightness(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust image brightness."""
        if factor is None:
            factor = random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust image contrast."""
        if factor is None:
            factor = random.uniform(0.5, 1.5)
        return np.clip(128 + factor * (image.astype(np.float32) - 128), 0, 255).astype(np.uint8)

    def adjust_saturation(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Adjust color saturation."""
        if factor is None:
            factor = random.uniform(0.5, 1.5)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def adjust_hue(self, image: np.ndarray, shift: int = None) -> np.ndarray:
        """Shift hue values."""
        if shift is None:
            shift = random.randint(-30, 30)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def add_gaussian_noise(self, image: np.ndarray, std: float = None) -> np.ndarray:
        """Add Gaussian noise to image."""
        if std is None:
            std = random.uniform(5, 25)
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def add_salt_pepper_noise(self, image: np.ndarray, amount: float = None) -> np.ndarray:
        """Add salt and pepper noise."""
        if amount is None:
            amount = random.uniform(0.01, 0.05)
        noisy = image.copy()
        # Salt
        num_salt = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255
        # Pepper
        num_pepper = int(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0
        return noisy

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """Apply Gaussian blur."""
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image using unsharp masking."""
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def scale(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Scale (zoom) image."""
        if factor is None:
            factor = random.uniform(0.8, 1.2)
        h, w = image.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)
        scaled = cv2.resize(image, (new_w, new_h))

        if factor > 1:  # Crop to original size
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return scaled[start_h:start_h+h, start_w:start_w+w]
        else:  # Pad to original size
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            return cv2.copyMakeBorder(scaled, pad_h, h-new_h-pad_h, 
                                     pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)

    def translate(self, image: np.ndarray, tx: int = None, ty: int = None) -> np.ndarray:
        """Translate (shift) image."""
        h, w = image.shape[:2]
        if tx is None:
            tx = random.randint(-int(w*0.2), int(w*0.2))
        if ty is None:
            ty = random.randint(-int(h*0.2), int(h*0.2))
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def shear(self, image: np.ndarray, shear_factor: float = None) -> np.ndarray:
        """Apply shear transformation."""
        if shear_factor is None:
            shear_factor = random.uniform(-0.3, 0.3)
        h, w = image.shape[:2]
        matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def elastic_transform(self, image: np.ndarray, alpha: float = None, 
                         sigma: float = None) -> np.ndarray:
        """Apply elastic deformation."""
        if alpha is None:
            alpha = random.uniform(30, 50)
        if sigma is None:
            sigma = random.uniform(5, 7)

        h, w = image.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        mapx = np.float32(x + dx)
        mapy = np.float32(y + dy)

        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def channel_shift(self, image: np.ndarray) -> np.ndarray:
        """Randomly shift color channels."""
        shift = random.randint(-20, 20)
        result = image.astype(np.int16)
        channel = random.randint(0, 2)
        result[:, :, channel] = np.clip(result[:, :, channel] + shift, 0, 255)
        return result.astype(np.uint8)

    def pixel_dropout(self, image: np.ndarray, dropout_prob: float = None) -> np.ndarray:
        """Randomly drop (set to 0) pixels."""
        if dropout_prob is None:
            dropout_prob = random.uniform(0.01, 0.05)
        mask = np.random.rand(*image.shape[:2]) > dropout_prob
        result = image.copy()
        result[~mask] = 0
        return result

    def perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply random perspective transformation."""
        h, w = image.shape[:2]
        margin = int(min(h, w) * 0.1)

        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)]
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def augment_random(self, image: np.ndarray, num_operations: int = None) -> np.ndarray:
        """Apply random combination of augmentations."""
        if num_operations is None:
            num_operations = random.randint(3, 6)

        operations = [
            self.rotate, self.flip, self.adjust_brightness, self.adjust_contrast,
            self.adjust_saturation, self.adjust_hue, self.add_gaussian_noise,
            self.gaussian_blur, self.sharpen, self.scale, self.translate,
            self.shear, self.elastic_transform, self.channel_shift,
            self.pixel_dropout, self.perspective_transform, self.add_salt_pepper_noise
        ]

        selected_ops = random.sample(operations, min(num_operations, len(operations)))
        result = image.copy()

        for op in selected_ops:
            result = op(result)

        return result


def augment_dataset(input_dir: str, output_dir: str, augmentations_per_image: int = 5,
                   seed: int = None, verbose: bool = True):
    """
    Augment all images in input directory and save to output directory.

    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        augmentations_per_image: Number of augmented versions per original image
        seed: Random seed for reproducibility
        verbose: Print progress information
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize augmentor
    augmentor = ImageAugmentor(seed=seed)

    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    # Get all image files
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                  if f.suffix.lower() in extensions and f.is_file()]

    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return

    if verbose:
        print(f"🖼️  Found {len(image_files)} images")
        print(f"📊 Generating {augmentations_per_image} augmentations per image")
        print(f"💾 Output directory: {output_dir}")
        print("-" * 60)

    total_generated = 0

    for idx, img_file in enumerate(image_files, 1):
        # Read image
        image = cv2.imread(str(img_file))

        if image is None:
            print(f"⚠️  Could not read: {img_file.name}")
            continue

        # Save original
        stem = img_file.stem
        ext = img_file.suffix
        original_output = output_path / f"{stem}_original{ext}"
        cv2.imwrite(str(original_output), image)

        # Generate augmentations
        for aug_idx in range(augmentations_per_image):
            augmented = augmentor.augment_random(image)
            output_file = output_path / f"{stem}_aug_{aug_idx+1:03d}{ext}"
            cv2.imwrite(str(output_file), augmented)
            total_generated += 1

        if verbose:
            print(f"[{idx}/{len(image_files)}] ✓ {img_file.name} → "
                  f"{augmentations_per_image} augmentations")

    if verbose:
        print("-" * 60)
        print(f"✅ Complete! Generated {total_generated} augmented images")
        print(f"📁 Total images in output: {total_generated + len(image_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Professional Image Data Augmentation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augment_images.py --input ./data --output ./augmented
  python augment_images.py -i ./images -o ./aug_images -n 10
  python augment_images.py -i ./train -o ./train_aug -n 5 --seed 42
        """
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Input directory containing images')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory for augmented images')
    parser.add_argument('-n', '--per-image', type=int, default=5,
                       help='Number of augmentations per image (default: 5)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        augmentations_per_image=args.per_image,
        seed=args.seed,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
