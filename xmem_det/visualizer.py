import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

class TemporalDebugger:
    """
    Complete debugging and visualization for ReasonNet temporal memory.
    
    Usage:
        debugger = TemporalDebugger(save_dir='./debug_vis', log_every=5)
        
        for frame_idx, data in enumerate(sequence):
            bev = model.backbone(data['lidar'])
            mfused, dbg = model.bank.compute_mfused(bev)
            
            # Visualize
            debugger.log_frame(model.bank, frame_idx, bev, mfused, dbg)
            
            # Training code...
            model.bank.update_bank(dbg['q_t'], mfused, mp)
        
        # End of sequence
        debugger.save_sequence_summary()
    """
    
    def __init__(self, save_dir='./temporal_debug', log_every=5, enable_detection_compare=False):
        """
        Args:
            save_dir: Where to save visualizations
            log_every: Create detailed visualizations every N frames
            enable_detection_compare: Whether to compare detections (slower)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_every = log_every
        self.enable_detection_compare = enable_detection_compare
        
        # Tracking data
        self.frames = []
        self.st_counts = []
        self.lt_positions = []
        self.total_positions = []
        self.feature_changes = []
        self.attention_entropy = []
        
        # Status flags
        self.first_memory_frame = None
        self.first_lt_frame = None
        
        print(f"✓ TemporalDebugger initialized")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Logging every {log_every} frames")
    
    def log_frame(self, bank, frame_idx, bev, mfused, dbg):
        """
        Main logging function - call this every frame.
        
        Args:
            bank: ReasonNetTemporalBank instance
            frame_idx: Current frame number
            bev: BEV features before temporal [B, C, H, W]
            mfused: BEV features after temporal [B, C, H, W]
            dbg: Debug dict from compute_mfused()
        """
        # Always track metrics (lightweight)
        self._track_metrics(bank, frame_idx, bev, mfused, dbg)
        
        # Detailed visualizations (heavy, only every N frames)
        if frame_idx % self.log_every == 0:
            self._visualize_memory_state(bank, frame_idx)
            self._visualize_temporal_impact(bev, mfused, frame_idx)
            
            if len(bank._st_keys) > 0:
                self._visualize_attention(bank, dbg, frame_idx)
        
        # Special events
        if len(bank._st_keys) > 0 and self.first_memory_frame is None:
            self.first_memory_frame = frame_idx
            print(f"🎉 Frame {frame_idx}: First memory added!")
        
        if len(bank._lt_keys) > 0 and self.first_lt_frame is None:
            self.first_lt_frame = frame_idx
            print(f"🎉 Frame {frame_idx}: Long-term memory created!")
    
    def _track_metrics(self, bank, frame_idx, bev, mfused, dbg):
        """Track numerical metrics (fast, no plotting)"""
        # Memory state
        st_count = len(bank._st_keys)
        lt_pos = sum(bank._lt_fill) if len(bank._lt_keys) > 0 else 0
        total_pos = st_count * 2200 + lt_pos
        
        # Feature change
        diff = (mfused - bev).abs().mean().item()
        bev_norm = bev.abs().mean().item()
        rel_change = (diff / bev_norm * 100) if bev_norm > 0 else 0
        
        # Store
        self.frames.append(frame_idx)
        self.st_counts.append(st_count)
        self.lt_positions.append(lt_pos)
        self.total_positions.append(total_pos)
        self.feature_changes.append(rel_change)
        
        # Print status every 10 frames
        if frame_idx % 10 == 0:
            status = "✓" if rel_change > 5 or frame_idx < 5 else "⚠️"
            print(f"Frame {frame_idx:3d} | Memory: {total_pos:5d} pos | "
                  f"Change: {rel_change:5.1f}% {status}")
    
    def _visualize_memory_state(self, bank, frame_idx):
        """Visualize current memory state"""
        st_count = len(bank._st_keys)
        lt_count = len(bank._lt_keys)
        lt_fills = [bank._lt_fill[i] for i in range(lt_count)] if lt_count > 0 else []
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # === Left: Bar chart of memory sources ===
        positions = []
        labels = []
        colors = []
        
        # ST frames
        for i in range(st_count):
            positions.append(2200)
            labels.append(f'ST_{i}')
            colors.append('royalblue')
        
        # LT buffers
        for i, fill in enumerate(lt_fills):
            positions.append(fill)
            labels.append(f'LT_{i}')
            colors.append('darkorange')
        
        if positions:
            bars = ax1.bar(range(len(positions)), positions, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, pos, label in zip(bars, positions, labels):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 50,
                        f'{label}\n{pos}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax1.axhline(y=2048, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label='LT Capacity (2048)')
            ax1.set_ylim([0, 2500])
        else:
            ax1.text(0.5, 0.5, 'NO MEMORY YET', 
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=16, fontweight='bold', color='red')
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
        
        ax1.set_xlabel('Memory Source', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Positions', fontsize=11, fontweight='bold')
        ax1.set_title('Current Memory Sources', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # === Right: Statistics ===
        total_st = st_count * 2200
        total_lt = sum(lt_fills) if lt_fills else 0
        total = total_st + total_lt
        
        # Text summary
        summary_text = f"""
MEMORY STATISTICS

Short-Term Memory:
  • Frames: {st_count} / {bank.ts}
  • Positions: {total_st:,}

Long-Term Memory:
  • Buffers: {lt_count} / {bank.tl}
  • Positions: {total_lt:,}

Total Queryable: {total:,}
        """
        
        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes,
                fontsize=11, family='monospace', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax2.axis('off')
        
        # Overall title
        fig.suptitle(f'Frame {frame_idx}: Memory State', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / f'memory_state_{frame_idx:04d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _visualize_temporal_impact(self, bev, mfused, frame_idx):
        """Visualize before/after temporal fusion"""
        # Average over channels
        bev_vis = bev[0].mean(dim=0).cpu().numpy()
        mfused_vis = mfused[0].mean(dim=0).cpu().numpy()
        diff = mfused_vis - bev_vis
        
        # Compute statistics
        rel_change = np.abs(diff).mean() / (np.abs(bev_vis).mean() + 1e-8) * 100
        
        # Determine status
        if frame_idx < 3:
            status = "Expected (no memory)"
            status_color = 'blue'
        elif rel_change < 1:
            status = "⚠️ SUSPICIOUS - Very small change!"
            status_color = 'red'
        elif rel_change > 100:
            status = "⚠️ WARNING - Very large change!"
            status_color = 'orange'
        else:
            status = "✓ Normal - Temporal working"
            status_color = 'green'
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Before
        im1 = axes[0].imshow(bev_vis, cmap='viridis')
        axes[0].set_title('Before Temporal\n(Raw BEV)', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # After
        im2 = axes[1].imshow(mfused_vis, cmap='viridis')
        axes[1].set_title('After Temporal\n(With Memory)', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Difference
        vmax = max(abs(diff.min()), abs(diff.max()))
        if vmax > 0:
            im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im3 = axes[2].imshow(diff, cmap='RdBu_r')
        axes[2].set_title('Difference\n(After - Before)', fontsize=11, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Title with statistics
        fig.suptitle(f'Frame {frame_idx} | Change: {rel_change:.1f}% | {status}',
                    fontsize=13, fontweight='bold', color=status_color)
        
        plt.tight_layout()
        save_path = self.save_dir / f'temporal_impact_{frame_idx:04d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _visualize_attention(self, bank, dbg, frame_idx):
        """Visualize attention distribution"""
        mem_keys, mem_vals, mem_kinds = bank._collect_memory()
        
        if len(mem_keys) == 0:
            return
        
        # Get query
        q_t = dbg['q_t']
        b, k, h, w = q_t.shape
        q_flat = q_t.reshape(b, k, h*w).permute(0, 2, 1)  # [B, HW, K]
        
        # Pick center query
        center_idx = (h * w) // 2
        q_center = q_flat[:, center_idx:center_idx+1, :]  # [1, 1, K]
        
        # Compute attention to each source
        attention_weights = []
        source_labels = []
        source_colors = []
        
        for i, (k_flat, kind) in enumerate(zip(mem_keys, mem_kinds)):
            # Distance
            dist = bank._dist_sq_block(q_center, k_flat)  # [1, 1, M]
            
            # Normalize
            S = dist / (dist.sum() + 1e-8)
            avg_attn = S.mean().item() * 100  # Percentage
            
            attention_weights.append(avg_attn)
            source_labels.append(f'{kind.upper()}_{i}')
            source_colors.append('royalblue' if kind == 'st' else 'darkorange')
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(source_labels)), attention_weights, 
                     color=source_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, weight in zip(bars, attention_weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                   f'{weight:.2f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Memory Source', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Attention (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Frame {frame_idx}: Attention Distribution from Center Position',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(source_labels)))
        ax.set_xticklabels(source_labels, fontsize=10, rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='royalblue', alpha=0.7, label='Short-Term'),
            Patch(facecolor='darkorange', alpha=0.7, label='Long-Term')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        save_path = self.save_dir / f'attention_{frame_idx:04d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_sequence_summary(self, seq_name='sequence'):
        """Save summary plots for entire sequence"""
        if len(self.frames) == 0:
            print("⚠️ No data logged yet")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # === Plot 1: Memory Growth ===
        ax1 = axes[0]
        ax1.fill_between(self.frames, 0, np.array(self.st_counts) * 2200,
                        alpha=0.5, color='royalblue', label='Short-Term')
        ax1.fill_between(self.frames, np.array(self.st_counts) * 2200,
                        self.total_positions,
                        alpha=0.5, color='darkorange', label='Long-Term')
        ax1.set_ylabel('Memory Positions', fontsize=11, fontweight='bold')
        ax1.set_title('Memory Growth Over Time', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Mark special events
        if self.first_memory_frame:
            ax1.axvline(self.first_memory_frame, color='green', linestyle='--', 
                       linewidth=2, label=f'First Memory (F{self.first_memory_frame})')
        if self.first_lt_frame:
            ax1.axvline(self.first_lt_frame, color='red', linestyle='--',
                       linewidth=2, label=f'First LT (F{self.first_lt_frame})')
        
        # === Plot 2: Feature Change ===
        ax2 = axes[1]
        ax2.plot(self.frames, self.feature_changes, 'o-', linewidth=2, 
                markersize=4, color='purple')
        ax2.axhline(5, color='green', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Good (>5%)')
        ax2.axhline(1, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='Suspicious (<1%)')
        ax2.set_ylabel('Feature Change (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Temporal Impact on Features', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # === Plot 3: Diagnostic ===
        ax3 = axes[2]
        
        # ST frame count
        ax3_twin = ax3.twinx()
        ax3.plot(self.frames, self.st_counts, 'o-', linewidth=2, 
                color='blue', label='ST Frames', markersize=4)
        ax3.axhline(4, color='blue', linestyle='--', alpha=0.5)
        
        # LT positions
        ax3_twin.plot(self.frames, self.lt_positions, 's-', linewidth=2,
                     color='orange', label='LT Positions', markersize=4)
        
        ax3.set_xlabel('Frame', fontsize=11, fontweight='bold')
        ax3.set_ylabel('ST Frame Count', fontsize=11, fontweight='bold', color='blue')
        ax3_twin.set_ylabel('LT Position Count', fontsize=11, fontweight='bold', color='orange')
        ax3.set_title('Memory Components Over Time', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        save_path = self.save_dir / f'summary_{seq_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved sequence summary: {save_path}")
        
        # Print summary statistics
        self._print_summary_stats()
    
    def _print_summary_stats(self):
        """Print summary statistics"""
        print(f"\n{'='*60}")
        print(f"SEQUENCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames logged: {len(self.frames)}")
        print(f"First memory frame: {self.first_memory_frame}")
        print(f"First LT frame: {self.first_lt_frame}")
        print(f"\nMemory Statistics:")
        print(f"  Max ST frames: {max(self.st_counts)}")
        print(f"  Max LT positions: {max(self.lt_positions)}")
        print(f"  Max total positions: {max(self.total_positions)}")
        print(f"\nTemporal Impact:")
        avg_change = np.mean(self.feature_changes[5:])  # Skip first few
        print(f"  Avg feature change: {avg_change:.2f}%")
        if avg_change > 5:
            print(f"  Status: ✓ Temporal working well!")
        elif avg_change > 1:
            print(f"  Status: ⚠️ Temporal working but weak")
        else:
            print(f"  Status: ❌ Temporal not working!")
        print(f"{'='*60}\n")
    
    def reset(self):
        """Reset for new sequence"""
        self.frames = []
        self.st_counts = []
        self.lt_positions = []
        self.total_positions = []
        self.feature_changes = []
        self.attention_entropy = []
        self.first_memory_frame = None
        self.first_lt_frame = None