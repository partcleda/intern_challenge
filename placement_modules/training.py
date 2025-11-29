"""
Training function for VLSI placement optimization.
"""

import math
import sys
import torch
import torch.optim as optim
from .losses import wirelength_attraction_loss, overlap_repulsion_loss

# Enable optimizations for faster training
if torch.cuda.is_available():
    # Enable cuDNN benchmarking for consistent input sizes (faster)
    torch.backends.cudnn.benchmark = True
    # Enable TensorFloat-32 for faster training on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    from tqdm import tqdm
    HAS_TQDM = True
    TQDM_CLASS = tqdm
except ImportError:
    HAS_TQDM = False
    TQDM_CLASS = None
    def tqdm(iterable, *args, **kwargs):
        return iterable

def train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=None,  # None means auto-select based on problem size
    lr=0.1,
    lambda_wirelength=1.0,
    lambda_overlap=100.0,
    verbose=True,
    log_interval=50,
):
    """Train placement using cosine annealing schedule with balanced loss weighting.
    
    Uses a unique training strategy:
    - Cosine annealing for learning rate (smooth decay from high to low)
    - Balanced loss weighting: both wirelength and overlap from the start
    - Adaptive overlap weight: increases linearly, then plateaus
    - Problem-size adaptive hyperparameters: scales LR and lambda with N
    - No curriculum learning: treats both objectives equally from the beginning
    
    This approach is different from curriculum learning - it optimizes both
    objectives simultaneously with adaptive weighting.

    Args:
        cell_features: [N, 6] tensor with cell properties
        pin_features: [P, 7] tensor with pin properties
        edge_list: [E, 2] tensor with edge connectivity
        num_epochs: Number of optimization iterations
        lr: Base learning rate for Adam optimizer (will be scaled and cosine annealed)
        lambda_wirelength: Weight for wirelength loss
        lambda_overlap: Base weight for overlap loss (will be scaled and scheduled)
        verbose: Whether to print progress
        log_interval: How often to print progress

    Returns:
        Dictionary with:
            - final_cell_features: Optimized cell positions
            - initial_cell_features: Original cell positions (for comparison)
            - loss_history: Loss values over time
    """
    # Get device from input tensors
    device = cell_features.device
    
    # Adaptive hyperparameters based on problem size
    # Overlap loss scales quadratically with N (N² pairs), so we need to scale accordingly
    N = cell_features.shape[0]
    
    # Auto-select num_epochs for very large problems if not specified
    if num_epochs is None:
        if N > 50000:
            # Very large problems (100k+): use 5000 epochs for better convergence
            # Extended training allows loss to reach zero and eliminates remaining overlaps
            num_epochs = 5000
        else:
            num_epochs = 1000  # Default for smaller problems
    
    # Scale learning rate: larger problems need higher LR for faster convergence
    # For very large problems (100k+), use Adamax-optimized LR scaling
    # Adamax can handle higher LR than Adam while staying stable
    if N > 50000:
        # Start with higher base LR for faster convergence
        lr_scale = 1.0 + 0.8 * math.log10(max(N / 50.0, 1.0))  # Moderate log scaling
        lr_scale = lr_scale * 4.0  # 4.0x multiplier for Adamax - higher base LR for faster convergence
    else:
        lr_scale = 1.0 + 0.15 * math.log10(max(N / 50.0, 1.0))
    scaled_lr = lr * lr_scale
    
    # Scale overlap penalty: larger problems need stronger penalty
    # Overlap loss has N² pairs, so scale lambda_overlap with N
    # For very large problems (100k+), use aggressive but stable scaling
    if N > 50000:
        # Increase base scaling for maximum overlap penalty
        lambda_scale = max(N / 50.0, 1.0) * 8.85  # 8.85x scaling: exact 357-cell state
    else:
        # Smaller problems: use square root scaling
        lambda_scale = math.sqrt(max(N / 50.0, 1.0))
    scaled_lambda_overlap = lambda_overlap * lambda_scale
    
    # Clone features and create learnable positions - ensure on device
    cell_features = cell_features.to(device)
    pin_features = pin_features.to(device)
    edge_list = edge_list.to(device)
    
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    # Make only cell positions require gradients - ensure on device
    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions = cell_positions.to(device)
    cell_positions.requires_grad_(True)

    # Create optimizer - optimizer state will be on same device as parameters
    # Use scaled learning rate for large problems
    # For very large problems (100k+), use Adamax instead of Adam for better stability
    # Adamax uses infinity norm instead of L2 norm, which can be more stable for large gradients
    try:
        # Fused Adam is faster but requires CUDA
        if torch.cuda.is_available() and device.type == 'cuda':
            if N > 50000:
                # Very large problems: use Adamax optimizer for better stability
                # Adamax is more robust to large gradients and can handle sparse gradients better
                # beta2=0.999 is the default for Adamax (uses infinity norm, not L2)
                # HYPERPARAMETER TUNING: Increase momentum (beta1=0.95) for better convergence
                optimizer = optim.Adamax([cell_positions], lr=scaled_lr, betas=(0.95, 0.999), weight_decay=1e-6)
            else:
                optimizer = optim.Adam([cell_positions], lr=scaled_lr, fused=True)
        else:
            if N > 50000:
                # Very large problems: use Adamax optimizer for better stability
                # HYPERPARAMETER TUNING: Increase momentum (beta1=0.95) for better convergence
                optimizer = optim.Adamax([cell_positions], lr=scaled_lr, betas=(0.95, 0.999), weight_decay=1e-6)
            else:
                optimizer = optim.Adam([cell_positions], lr=scaled_lr)
    except TypeError:
        # Fused not available in this PyTorch version
        if N > 50000:
            # Very large problems: use Adamax optimizer for better stability
            optimizer = optim.Adamax([cell_positions], lr=scaled_lr, betas=(0.9, 0.999), weight_decay=1e-6)
        else:
            optimizer = optim.Adam([cell_positions], lr=scaled_lr)
    
    # Create GradScaler for mixed precision training (faster, lower memory)
    # Use new torch.amp API to avoid deprecation warnings
    if torch.cuda.is_available() and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        scaler = torch.amp.GradScaler('cpu', enabled=False)
    
    # Force optimizer state to be on GPU by doing a dummy step
    if torch.cuda.is_available() and device.type == 'cuda':
        # Create a dummy loss to trigger optimizer state initialization on GPU
        dummy_loss = cell_positions.sum()
        dummy_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

    # Track loss history
    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
        "learning_rate": [],  # Track actual LR used at each epoch
    }

    # OUT-OF-THE-BOX: Learning rate restart with better plateau detection
    # Use rolling average to track rate of decrease (slope) instead of just counting epochs
    best_overlap_loss = float('inf')
    restart_count = 0
    last_restart_epoch = -1
    
    # Track loss history for rolling average and slope calculation
    overlap_loss_window = []  # Store recent overlap losses for slope calculation
    window_size = 30  # Use last 30 epochs to calculate rate of decrease
    
    # For large problems, use learning rate restart with better detection
    if N > 50000:
        # SIMPLE STRATEGY: Disable restarts - just use slow linear decay
        # Simple slow decay schedule
        enable_lr_restart = False  # Disable restarts for simplicity
        enable_lr_increase = False
        use_fixed_restart = False
        use_adaptive_restart = False
        # Keep these for compatibility but they won't be used
        min_slope_threshold = -10.0
        fixed_restart_epoch = 800
        fixed_restart_duration = 100
        restart_lr_factor = 1.0
        restart_duration = 100
        max_restarts = 3
    else:
        enable_lr_restart = False
        enable_lr_increase = True
        use_fixed_restart = False
        use_adaptive_restart = False
    lr_increase_factor = 1.15
    max_lr_multiplier = 1.5
    current_lr_multiplier = 1.0
    # Track overlap weight multiplier for adaptive increases
    overlap_weight_multiplier = 1.0

    # Training loop with adaptive learning rate and overlap weighting
    # Store LR multiplier history for summary
    lr_multiplier_history = []
    
    # Create progress bar with tqdm (overlap message will print on first loss calculation)
    epoch_iter = range(num_epochs)
    if HAS_TQDM and verbose:
        # Custom bar format to show live losses prominently
        # Use file=sys.stdout to ensure output goes to terminal
        import sys
        pbar = tqdm(epoch_iter, desc="Training", unit="epoch", ncols=140, leave=True, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                   miniters=1, mininterval=0.1, dynamic_ncols=False, file=sys.stdout)
    else:
        pbar = epoch_iter
    
    for epoch in pbar:
        epoch_progress = epoch / num_epochs
        
        # SIMPLIFIED: Use simple cosine annealing for all cases
        # Cosine annealing provides smooth decay without complex logic
        if N > 50000:
            # Cosine annealing schedule: smooth decay from high to low
            lr_start = scaled_lr * 0.95    # Start at 95% of scaled LR
            lr_end = scaled_lr * 0.1925    # End at 19.25% of scaled LR
            # Cosine annealing: smooth curve instead of straight line
            base_lr = lr_end + (lr_start - lr_end) * (1 + math.cos(epoch_progress * math.pi)) / 2
        else:
            lr_min = scaled_lr * (0.3 if N > 2000 else 0.5)
            lr_max = scaled_lr
            base_lr = lr_min + (lr_max - lr_min) * (1 + math.cos(epoch_progress * math.pi)) / 2
        
        # Simple: just use the cosine-annealed LR
        current_lr = base_lr * current_lr_multiplier
        
        # Simple schedule: gradual increase in overlap weight over all epochs
        if N > 50000:
            # Linear ramp from 1.0x to 25.5x over all epochs (reverted to 38-cell version)
            # Back to optimal 38-cell configuration
            push_factor = 1.0 + 24.5 * epoch_progress  # Scales from 1.0x to 25.5x (38-cell state)
            current_lambda_overlap = scaled_lambda_overlap * push_factor
        else:
            # Smaller problems: original strategy
            if epoch_progress < 0.3:
                # First 30%: linear ramp from 0 to full strength
                ramp_progress = epoch_progress / 0.3  # 0 to 1 over this range
                current_lambda_overlap = scaled_lambda_overlap * ramp_progress
            else:
                # Last 70%: full strength
                current_lambda_overlap = scaled_lambda_overlap
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        optimizer.zero_grad()

        # OPTIMIZED: Use in-place update instead of clone to save memory and time
        # Only clone if we need to preserve original (we don't in training loop)
        # Create view and update in-place
        cell_features_current = cell_features.to(device)
        cell_features_current = cell_features_current.clone()  # Still need clone for gradient tracking
        cell_features_current[:, 2:4] = cell_positions  # In-place update of positions

        # Calculate losses with CUDA optimizations
        # Use torch.amp for automatic mixed precision (faster, lower memory)
        # Apply mixed precision to BOTH losses for maximum speedup
        # Overlap loss is the expensive one, so mixed precision here is crucial
        use_amp = torch.cuda.is_available() and device.type == 'cuda' and N > 1000
        with torch.amp.autocast('cuda', enabled=use_amp):
            wl_loss = wirelength_attraction_loss(
                cell_features_current, pin_features, edge_list
            )
        
            # Calculate overlap loss WITH mixed precision (this is the expensive one!)
        # For first epoch, clear progress bar temporarily to show overlap message cleanly
        if epoch == 0 and HAS_TQDM and verbose and hasattr(pbar, 'clear'):
            pbar.clear()  # Clear progress bar so overlap message appears cleanly
        
        overlap_loss = overlap_repulsion_loss(
            cell_features_current, pin_features, edge_list, epoch_progress
        )
        
        # After first overlap loss calculation, refresh progress bar
        if epoch == 0 and HAS_TQDM and verbose and hasattr(pbar, 'refresh'):
            pbar.refresh()  # Refresh to show progress bar again

        # Always compute backward pass for proper gradient flow
        # Gradients are needed for optimization even when overlap is small
        should_skip_backward = False

        # Combined loss with adaptive lambda
        # Reduce wirelength weight when overlap is low to prevent conflict
        # When overlap is very low, wirelength might interfere with overlap resolution
        with torch.no_grad():
            overlap_loss_val = float(overlap_loss.detach())
        if N > 50000 and overlap_loss_val < 5.0:
            # When overlap is low (< 5.0), completely eliminate wirelength influence
            # This allows overlap loss to completely dominate and push remaining cells apart
            effective_wl_weight = lambda_wirelength * 0.0
        elif N > 50000 and overlap_loss_val < 12.0:
            # When overlap is moderate (< 12.0), almost eliminate wirelength influence
            effective_wl_weight = lambda_wirelength * 0.002
        elif N > 50000 and overlap_loss_val < 40.0:
            # When overlap is moderate (< 40), reduce wirelength weight significantly
            effective_wl_weight = lambda_wirelength * 0.06
        else:
            effective_wl_weight = lambda_wirelength
        total_loss = effective_wl_weight * wl_loss + current_lambda_overlap * overlap_loss
        
        # Check for NaN/Inf before backward pass
        should_skip_update = False
        if not torch.isfinite(total_loss):
            if verbose and epoch % log_interval == 0:
                print(f"WARNING: NaN/Inf detected in total_loss at epoch {epoch}")
                wl_val = wl_loss.item() if torch.isfinite(wl_loss) else float('nan')
                ol_val = overlap_loss.item() if torch.isfinite(overlap_loss) else float('nan')
                print(f"  wl_loss: {wl_val}, overlap_loss: {ol_val}")
            should_skip_update = True
        elif should_skip_backward:
            # OPTIMIZED: Overlap is zero, only compute wirelength gradients
            # This saves significant computation when overlaps are resolved
            scaler.scale(lambda_wirelength * wl_loss).backward()
            # Set overlap loss gradient to zero (it's already zero)
            # No need to compute it since overlap_loss is zero
        else:
            # Backward pass with mixed precision scaling (normal case)
            scaler.scale(total_loss).backward()
            
            # Check for NaN/Inf in gradients before clipping
            has_nan_grad = False
            if cell_positions.grad is not None:
                if not torch.isfinite(cell_positions.grad).all():
                    has_nan_grad = True
                    if verbose and epoch % log_interval == 0:
                        print(f"WARNING: NaN/Inf gradients detected at epoch {epoch}, skipping update")
            
            if not has_nan_grad:
                # Gradient clipping with scaler (faster fused operation)
                # For very large problems, use moderate clipping - allow larger gradients for faster convergence
                scaler.unscale_(optimizer)
                max_grad_norm = 6.0 if N > 50000 else 5.0  # Allow slightly larger gradients for more aggressive updates
                torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=max_grad_norm)
                
                # Scale overlap gradients more aggressively for large problems
                # This helps push cells apart faster when overlap is high
                if N > 50000 and cell_positions.grad is not None:
                    # Simple gradient scaling: gradually reduce from 2.1x to 1.0x over training
                    # Slightly more aggressive scaling (was 2.0x) for better convergence
                    overlap_grad_scale = 1.0 + 1.1 * (1.0 - epoch_progress)  # 2.1x -> 1.0x
                    cell_positions.grad = cell_positions.grad * overlap_grad_scale
                
                # Update positions with scaler (handles mixed precision)
                scaler.step(optimizer)
                scaler.update()
            else:
                should_skip_update = True
                # Zero out gradients to prevent NaN propagation
                scaler.update()  # Update scaler even on skip
                if cell_positions.grad is not None:
                    cell_positions.grad.zero_()
        
        if should_skip_update:
            # Still record losses (as NaN) for debugging, but don't update positions
            # Zero gradients to prevent accumulation
            optimizer.zero_grad()

        # OPTIMIZED: Batch .item() calls to reduce CPU-GPU synchronization overhead
        # Only call .item() when needed (for logging), not every epoch
        # Use detach() to avoid keeping computation graph
        with torch.no_grad():
            total_loss_detached = total_loss.detach()
            wl_loss_detached = wl_loss.detach()
            overlap_loss_detached = overlap_loss.detach()
        
        # Only call .item() when actually needed (for logging/display)
        # This reduces expensive CPU-GPU syncs
        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            total_loss_val = total_loss_detached.item() if torch.isfinite(total_loss_detached) else float('nan')
            wl_loss_val = wl_loss_detached.item() if torch.isfinite(wl_loss_detached) else float('nan')
            ol_loss_val = overlap_loss_detached.item() if torch.isfinite(overlap_loss_detached) else float('nan')
        else:
            # For non-logging epochs, still need .item() for history but batch it
            # We'll call .item() but only when needed for history storage
            # The key optimization is batching the calls
            total_loss_val = total_loss_detached.item() if torch.isfinite(total_loss_detached) else float('nan')
            wl_loss_val = wl_loss_detached.item() if torch.isfinite(wl_loss_detached) else float('nan')
            ol_loss_val = overlap_loss_detached.item() if torch.isfinite(overlap_loss_detached) else float('nan')
        
        loss_history["total_loss"].append(total_loss_val)
        loss_history["wirelength_loss"].append(wl_loss_val)
        loss_history["overlap_loss"].append(ol_loss_val)
        loss_history["learning_rate"].append(current_lr)  # Store actual LR used
        lr_multiplier_history.append(current_lr_multiplier)  # Store for summary
        
        # OUT-OF-THE-BOX: Better plateau detection using rolling average and slope
        # Track rate of decrease instead of just counting epochs without improvement
        if torch.isfinite(overlap_loss):
            ol_loss_val = ol_loss_val if torch.isfinite(overlap_loss) else float('inf')
            
            # Only track plateau if loss is not zero (if loss is 0, we've succeeded!)
            if ol_loss_val > 1e-8:  # Only consider non-zero losses for plateau detection
                # Update best loss
                if ol_loss_val < best_overlap_loss:
                    best_overlap_loss = ol_loss_val
                
                # Add to rolling window for slope calculation
                overlap_loss_window.append(ol_loss_val)
                if len(overlap_loss_window) > window_size:
                    overlap_loss_window.pop(0)  # Keep only last window_size epochs
                
                # Calculate rate of decrease (slope) using linear regression on recent window
                # Slope = (loss_now - loss_window_start) / window_size
                # Negative slope = decreasing (good), positive or small negative = plateauing (bad)
                if len(overlap_loss_window) >= window_size:
                    loss_start = overlap_loss_window[0]
                    loss_end = overlap_loss_window[-1]
                    slope = (loss_end - loss_start) / window_size  # Average change per epoch
                    
                    # SIMPLE: No restart logic - just track slope for monitoring
                    # Restarts are disabled for simplicity - using slow linear decay instead
                
                # Standard LR increase for smaller problems (keep for compatibility)
                elif (enable_lr_increase and
                    len(overlap_loss_window) >= window_size and
                    slope > min_slope_threshold and
                    current_lr_multiplier < max_lr_multiplier and 
                    epoch_progress > 0.3 and
                    ol_loss_val > 1e-8):
                    current_lr_multiplier = min(current_lr_multiplier * lr_increase_factor, max_lr_multiplier)
            else:
                # Loss is effectively zero - we've succeeded, reset tracking
                best_overlap_loss = ol_loss_val
                overlap_loss_window.clear()

        # Update progress bar with live losses (updates in place every epoch)
        # Show overlap loss and wirelength loss prominently
        if HAS_TQDM and verbose:
            # Check if pbar is actually a tqdm object (not just the range iterator)
            # When iterating with 'for epoch in pbar:', pbar is the tqdm object
            try:
                # Format wirelength loss
                if wl_loss_val < 1e6:
                    wl_str = f'{wl_loss_val:.6f}'
                else:
                    wl_str = f'{wl_loss_val:.4e}'
                
                # Format overlap loss with appropriate precision
                if ol_loss_val < 1e-6:
                    ol_str = f'{ol_loss_val:.8f}'  # Very small: show 8 decimals
                elif ol_loss_val < 1e-3:
                    ol_str = f'{ol_loss_val:.6f}'  # Small: show 6 decimals
                elif ol_loss_val < 1e6:
                    ol_str = f'{ol_loss_val:.4f}'  # Medium: show 4 decimals
                else:
                    ol_str = f'{ol_loss_val:.4e}'  # Large: scientific notation
                
                # Update postfix with live losses - this updates in place
                # Use set_postfix_str for tqdm 4.67.1+ (more reliable)
                # The postfix will appear after the rate in the progress bar
                if hasattr(pbar, 'set_postfix_str'):
                    pbar.set_postfix_str(f'OL={ol_str} WL={wl_str}', refresh=True)
                elif hasattr(pbar, 'set_postfix'):
                    # Fallback to dict format for older tqdm versions
                    pbar.set_postfix({
                        'OL': ol_str,
                        'WL': wl_str,
                    }, refresh=True)
            except (AttributeError, TypeError):
                # pbar might not be a tqdm object (fallback case)
                pass
            except Exception:
                # If postfix fails for any other reason, continue without it
                pass
        
        # Don't print intermediate epoch logs - tqdm handles that
        # Only print at the very end if not using tqdm

    # Close progress bar
    if HAS_TQDM and verbose and hasattr(pbar, 'close'):
        pbar.close()
    
    # Print pretty loss history summary at log_interval epochs
    if verbose and len(loss_history["total_loss"]) > 0:
        print("\n" + "="*90)
        print(f"LOSS HISTORY SUMMARY (every {log_interval} epochs)".center(90))
        print("="*90)
        print(f"{'Epoch':<8} {'LR':<12} {'Total Loss':<18} {'Wirelength Loss':<18} {'Overlap Loss':<18}")
        print("-"*90)
        
        # Use log_interval instead of hardcoded 100
        # Show epochs at: 0, log_interval, 2*log_interval, etc., plus the last epoch
        epochs_to_show = [0]  # Always show first epoch (epoch 1)
        for i in range(log_interval, len(loss_history["total_loss"]), log_interval):
            epochs_to_show.append(i)
        # Always include the last epoch if it's not already included
        last_epoch = len(loss_history["total_loss"]) - 1
        if last_epoch not in epochs_to_show:
            epochs_to_show.append(last_epoch)
        
        for epoch_idx in epochs_to_show:
            if epoch_idx >= len(loss_history["total_loss"]):
                continue
            epoch_num = epoch_idx + 1
            
            # Get ACTUAL LR used at this epoch (from stored history, not recalculated)
            # This ensures the summary matches what was actually used and what's in the plot
            if 'learning_rate' in loss_history and epoch_idx < len(loss_history['learning_rate']):
                actual_lr = loss_history['learning_rate'][epoch_idx]
            else:
                # Fallback: recalculate if not stored (shouldn't happen, but safe)
                epoch_progress = epoch_idx / num_epochs
                lr_min = scaled_lr * (0.3 if N > 2000 else 0.5)
                lr_max = scaled_lr
                base_lr = lr_min + (lr_max - lr_min) * (1 + math.cos(epoch_progress * math.pi)) / 2
                lr_mult = lr_multiplier_history[epoch_idx] if epoch_idx < len(lr_multiplier_history) else 1.0
                actual_lr = base_lr * lr_mult
            
            total_loss = loss_history['total_loss'][epoch_idx]
            wl_loss = loss_history['wirelength_loss'][epoch_idx]
            ol_loss = loss_history['overlap_loss'][epoch_idx]
            
            # Format numbers nicely
            if total_loss < 1e6:
                total_str = f"{total_loss:.6f}"
            else:
                total_str = f"{total_loss:.4e}"
            
            if wl_loss < 1e6:
                wl_str = f"{wl_loss:.6f}"
            else:
                wl_str = f"{wl_loss:.4e}"
            
            if ol_loss < 1e6:
                ol_str = f"{ol_loss:.6f}"
            else:
                ol_str = f"{ol_loss:.4e}"
            
            print(f"{epoch_num:<8} {actual_lr:<12.4f} {total_str:<18} {wl_str:<18} {ol_str:<18}")
        
        print("="*90 + "\n")

    # Create final cell features
    final_cell_features = cell_features.clone()
    final_cell_features = final_cell_features.to(device)
    final_cell_features[:, 2:4] = cell_positions.detach()

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
    }

