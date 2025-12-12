import os
import torch
from PIL import Image
from placement import generate_placement_input, train_placement

def create_visualization_demo():
    print("Generating visualization demo...")
    
    # Use a medium-sized test case
    num_macros = 5
    num_std_cells = 50
    
    # Generate data
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)
    
    # Initialize positions (random spread)
    total_cells = cell_features.shape[0]
    spread_radius = 30.0
    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)
    
    # Run training with visualization enabled
    visual_dir = "demo_visuals"
    train_placement(
        cell_features, 
        pin_features, 
        edge_list, 
        num_epochs=500,  # Shorter run for demo
        save_visuals=True,
        visual_dir=visual_dir,
        log_interval=50
    )
    
    # Create GIF
    images = []
    filenames = sorted([f for f in os.listdir(visual_dir) if f.endswith('.png')])
    
    if not filenames:
        print("No frames generated.")
        return

    print(f"Stitching {len(filenames)} frames into GIF...")
    for filename in filenames:
        images.append(Image.open(os.path.join(visual_dir, filename)))
        
    output_gif = "optimization_process.gif"
    if images:
        images[0].save(
            output_gif, 
            save_all=True, 
            append_images=images[1:], 
            duration=100,  # 100ms per frame
            loop=0
        )
    
    print(f"Visualization saved to {output_gif}")

if __name__ == "__main__":
    create_visualization_demo()
