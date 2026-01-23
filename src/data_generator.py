import os
import random
import math
from PIL import Image, ImageDraw, ImageFont

def get_random_time():
    """Generates a random time (H, M, S)."""
    h = random.randint(0, 11) # 0-11 for analog visual logic, can map to 24h later if needed
    m = random.randint(0, 59)
    s = random.randint(0, 59)
    return h, m, s

def draw_digital_clock(h, m, s, size=(256, 256)):
    """Draws a digital clock image."""
    # Create black background
    img = Image.new('RGB', size, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Format time string
    time_str = f"{h:02d}:{m:02d}:{s:02d}"
    
    # Try to load a font, fallback to default if not found
    try:
        # Using a monospaced font if available usually looks better for digital
        # On Windows, 'arial.ttf' or 'consola.ttf' are common
        font = ImageFont.truetype("arial.ttf", size=int(size[0]/5))
    except IOError:
        font = ImageFont.load_default()

    # Calculate text position to center it (using getbbox for newer Pillow versions)
    left, top, right, bottom = draw.textbbox((0, 0), time_str, font=font)
    text_width = right - left
    text_height = bottom - top
    
    position = ((size[0] - text_width) / 2, (size[1] - text_height) / 2)
    
    # Draw text in bright red/orange like a classic LED clock
    draw.text(position, time_str, fill=(255, 50, 50), font=font)
    
    return img

def draw_analog_clock(h, m, s, size=(256, 256)):
    """Draws an analog clock image."""
    # White background or light gray
    bg_color = (240, 240, 240)
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    center = (size[0] // 2, size[1] // 2)
    radius = (min(size) // 2) - 10
    
    # Draw clock face (circle)
    draw.ellipse((center[0] - radius, center[1] - radius, 
                  center[0] + radius, center[1] + radius), 
                 outline=(0, 0, 0), width=4, fill=(255, 255, 255))
    
    # Draw ticks (optional, makes it look more realistic)
    for i in range(12):
        angle = math.radians(i * 30 - 90)
        start_x = center[0] + (radius - 15) * math.cos(angle)
        start_y = center[1] + (radius - 15) * math.sin(angle)
        end_x = center[0] + radius * math.cos(angle)
        end_y = center[1] + radius * math.sin(angle)
        draw.line((start_x, start_y, end_x, end_y), fill=(0,0,0), width=3)

    # --- Calculate Angles ---
    # Seconds: 6 degrees per second
    sec_angle = math.radians(s * 6 - 90)
    
    # Minutes: 6 degrees per minute + adjustment for seconds
    min_angle = math.radians(m * 6 + (s * 0.1) - 90)
    
    # Hours: 30 degrees per hour + adjustment for minutes
    hour_angle = math.radians((h % 12) * 30 + (m * 0.5) - 90)
    
    # --- Draw Hands ---
    
    # Hour Hand (Short, Thick)
    hour_len = radius * 0.5
    draw.line((center[0], center[1], 
               center[0] + hour_len * math.cos(hour_angle), 
               center[1] + hour_len * math.sin(hour_angle)), 
              fill=(0, 0, 0), width=8)
              
    # Minute Hand (Long, Medium)
    min_len = radius * 0.75
    draw.line((center[0], center[1], 
               center[0] + min_len * math.cos(min_angle), 
               center[1] + min_len * math.sin(min_angle)), 
              fill=(0, 0, 0), width=5)
              
    # Second Hand (Long, Thin, Red)
    sec_len = radius * 0.85
    draw.line((center[0], center[1], 
               center[0] + sec_len * math.cos(sec_angle), 
               center[1] + sec_len * math.sin(sec_angle)), 
              fill=(255, 0, 0), width=2)
              
    # Center Cap
    draw.ellipse((center[0]-5, center[1]-5, center[0]+5, center[1]+5), fill=(0,0,0))
    
    return img

def generate_dataset(count, output_dir):
    """Generates 'count' pairs of images in 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {count} image pairs in {output_dir}...")
    
    for i in range(count):
        h, m, s = get_random_time()
        
        # Draw images
        dig_img = draw_digital_clock(h, m, s)
        ana_img = draw_analog_clock(h, m, s)
        
        # Save images
        # Filename format: clock_{index}_{H}_{M}_{S}_{type}.png
        base_name = f"clock_{i}_{h:02d}_{m:02d}_{s:02d}"
        
        dig_img.save(os.path.join(output_dir, f"{base_name}_digital.png"))
        ana_img.save(os.path.join(output_dir, f"{base_name}_analog.png"))
        
    print("Done.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic clock dataset.")
    parser.add_argument("--count", type=int, default=10, help="Number of image pairs to generate.")
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "train"), help="Directory to save images.")
    
    args = parser.parse_args()
    
    generate_dataset(args.count, args.output_dir)
