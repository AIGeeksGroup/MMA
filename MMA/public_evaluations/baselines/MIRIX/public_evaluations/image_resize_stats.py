#!/usr/bin/env python3
"""
Script to analyze image resizing statistics for Gemini API
Processes many images and calculates average sizes, compression ratios, etc.
"""

import os
import glob
from PIL import Image
import base64
import io
import statistics
from collections import defaultdict

def analyze_image_resizing(input_dirs=None, output_dir="tmp_images", max_samples=50):
    """
    Analyze image resizing statistics across many files
    
    Args:
        input_dirs: List of directories to search for images
        output_dir: Directory to save sample resized images
        max_samples: Maximum number of images to process
    """
    
    if input_dirs is None:
        input_dirs = [".", "../", "data/", "../assets/"]
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp', '*.gif']
    all_image_files = []
    
    for input_dir in input_dirs:
        if os.path.exists(input_dir):
            print(f"Searching in: {input_dir}")
            for ext in image_extensions:
                all_image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
                all_image_files.extend(glob.glob(os.path.join(input_dir, '**', ext.upper()), recursive=True))
    
    # Remove duplicates
    all_image_files = list(set(all_image_files))
    
    if not all_image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(all_image_files)} unique image files")
    
    # Statistics tracking
    stats = {
        'processed': 0,
        'errors': 0,
        'original_sizes': [],
        'resized_sizes': [],
        'base64_sizes': [],
        'compression_ratios': [],
        'original_dimensions': [],
        'file_extensions': defaultdict(int)
    }
    
    # Process images
    print(f"Processing up to {min(len(all_image_files), max_samples)} images...")
    
    for i, image_path in enumerate(all_image_files[:max_samples]):
        try:
            if i % 10 == 0:
                print(f"Progress: {i+1}/{min(len(all_image_files), max_samples)}")
            
            # Get file extension stats
            ext = os.path.splitext(image_path)[1].lower()
            stats['file_extensions'][ext] += 1
            
            # Open original image
            pil_image = Image.open(image_path)
            original_size = pil_image.size
            stats['original_dimensions'].append(original_size)
            
            # Get original file size
            original_file_size = os.path.getsize(image_path)
            stats['original_sizes'].append(original_file_size)
            
            # Resize to 256x256 (same logic as in agent)
            resized_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
            
            # Convert to RGB and save to buffer (same as agent)
            img_buffer = io.BytesIO()
            resized_image.convert('RGB').save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            
            # Get resized file size
            resized_file_size = len(img_buffer.getvalue())
            stats['resized_sizes'].append(resized_file_size)
            
            # Get base64 size
            base64_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            stats['base64_sizes'].append(len(base64_data))
            
            # Calculate compression ratio
            compression_ratio = resized_file_size / original_file_size
            stats['compression_ratios'].append(compression_ratio)
            
            # Save first few samples for visual inspection
            if i < 10:
                original_name = os.path.basename(image_path)
                name_without_ext = os.path.splitext(original_name)[0]
                sample_path = os.path.join(output_dir, f"sample_{i:02d}_{name_without_ext}.jpg")
                resized_image.convert('RGB').save(sample_path, format='JPEG', quality=85)
            
            stats['processed'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            if stats['errors'] <= 5:  # Only print first few errors
                print(f"Error processing {image_path}: {e}")
            continue
    
    # Calculate and display statistics
    print("\n" + "="*60)
    print("IMAGE RESIZING STATISTICS")
    print("="*60)
    
    print(f"Total images found: {len(all_image_files)}")
    print(f"Images processed successfully: {stats['processed']}")
    print(f"Images with errors: {stats['errors']}")
    
    if stats['processed'] == 0:
        print("No images processed successfully!")
        return
    
    print(f"\nFILE EXTENSIONS:")
    for ext, count in sorted(stats['file_extensions'].items()):
        print(f"  {ext}: {count} files")
    
    print(f"\nORIGINAL DIMENSIONS (width x height):")
    widths = [dim[0] for dim in stats['original_dimensions']]
    heights = [dim[1] for dim in stats['original_dimensions']]
    print(f"  Width  - Min: {min(widths):,}, Max: {max(widths):,}, Avg: {statistics.mean(widths):.0f}")
    print(f"  Height - Min: {min(heights):,}, Max: {max(heights):,}, Avg: {statistics.mean(heights):.0f}")
    
    print(f"\nFILE SIZES (bytes):")
    print(f"  Original files:")
    print(f"    Average: {statistics.mean(stats['original_sizes']):,.0f} bytes ({statistics.mean(stats['original_sizes'])/1024/1024:.2f} MB)")
    print(f"    Median:  {statistics.median(stats['original_sizes']):,.0f} bytes ({statistics.median(stats['original_sizes'])/1024/1024:.2f} MB)")
    print(f"    Min:     {min(stats['original_sizes']):,} bytes")
    print(f"    Max:     {max(stats['original_sizes']):,} bytes")
    
    print(f"  Resized files (256x256, JPEG 85%):")
    print(f"    Average: {statistics.mean(stats['resized_sizes']):,.0f} bytes ({statistics.mean(stats['resized_sizes'])/1024:.2f} KB)")
    print(f"    Median:  {statistics.median(stats['resized_sizes']):,.0f} bytes ({statistics.median(stats['resized_sizes'])/1024:.2f} KB)")
    print(f"    Min:     {min(stats['resized_sizes']):,} bytes")
    print(f"    Max:     {max(stats['resized_sizes']):,} bytes")
    
    print(f"  Base64 encoded sizes:")
    print(f"    Average: {statistics.mean(stats['base64_sizes']):,.0f} characters ({statistics.mean(stats['base64_sizes'])/1024:.2f} KB)")
    print(f"    Median:  {statistics.median(stats['base64_sizes']):,.0f} characters ({statistics.median(stats['base64_sizes'])/1024:.2f} KB)")
    print(f"    Min:     {min(stats['base64_sizes']):,} characters")
    print(f"    Max:     {max(stats['base64_sizes']):,} characters")
    
    print(f"\nCOMPRESSION RATIOS:")
    avg_compression = statistics.mean(stats['compression_ratios'])
    print(f"  Average: {avg_compression:.4f} ({100*avg_compression:.2f}% of original size)")
    print(f"  Median:  {statistics.median(stats['compression_ratios']):.4f}")
    print(f"  Min:     {min(stats['compression_ratios']):.4f}")
    print(f"  Max:     {max(stats['compression_ratios']):.4f}")
    
    # Calculate total data savings
    total_original = sum(stats['original_sizes'])
    total_resized = sum(stats['resized_sizes'])
    total_savings = total_original - total_resized
    savings_percent = (total_savings / total_original) * 100
    
    print(f"\nTOTAL DATA USAGE:")
    print(f"  Original total: {total_original:,} bytes ({total_original/1024/1024:.2f} MB)")
    print(f"  Resized total:  {total_resized:,} bytes ({total_resized/1024/1024:.2f} MB)")
    print(f"  Data savings:   {total_savings:,} bytes ({total_savings/1024/1024:.2f} MB)")
    print(f"  Savings percent: {savings_percent:.2f}%")
    
    # API cost implications
    base64_total = sum(stats['base64_sizes'])
    print(f"\nAPI IMPLICATIONS:")
    print(f"  Total base64 data per batch: {base64_total:,} characters ({base64_total/1024/1024:.2f} MB)")
    print(f"  Average base64 per image: {statistics.mean(stats['base64_sizes']):.0f} characters")
    
    # Estimate API limits
    avg_base64_size = statistics.mean(stats['base64_sizes'])
    max_images_estimate = 4000000 // avg_base64_size  # Rough estimate for 4MB limit
    print(f"  Estimated max images per API call: ~{max_images_estimate} images")
    
    print(f"\nSample resized images saved to: {output_dir}/")
    print(f"First 10 processed images saved as: sample_00_filename.jpg, sample_01_filename.jpg, etc.")

def main():
    """Main function"""
    print("Image Resizing Statistics Calculator")
    print("=" * 60)
    
    # Search multiple directories
    search_dirs = [
        ".",
        "../",
        "data/",
        "../assets/",
        "../frontend/",
        "../../",
    ]
    
    analyze_image_resizing(
        input_dirs=search_dirs,
        output_dir="tmp_images",
        max_samples=100  # Process up to 100 images
    )

if __name__ == "__main__":
    main() 