import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class TemporalDebugger:
    """
    Visualizer for temporal 3D detection models.
    Tracks how LSTM modifies features and whether temporal reasoning helps detection.
    """
    def __init__(self, save_dir: str, log_every: int = 1, max_batches: int = 1):
        self.save_dir = save_dir
        self.log_every = int(log_every)
        self.max_batches = int(max_batches)
        self.seq_dir: Optional[str] = None
        
        # Track metrics across sequence
        self.metrics = {
            'mean_confidence': [],
            'num_detections': [],
            'feature_diff_mean': [],
            'feature_diff_max': [],
            'lstm_h_mean': [],
            'lstm_h_std': [],
        }
        
        self.enabled = True

    def start_sequence(self, seq_name: str):
        """Start tracking a new sequence"""
        if not self.enabled:
            return
            
        self.seq_dir = os.path.join(self.save_dir, seq_name)
        os.makedirs(self.seq_dir, exist_ok=True)
        
        # Reset metrics
        for k in self.metrics:
            self.metrics[k] = []

    def _to_cpu_2d(self, x: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy for visualization"""
        return x.detach().float().cpu().numpy()

    def _bev_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm across channel dimension"""
        return torch.linalg.vector_norm(x, dim=0)

    def log_timestep(
        self,
        t: int,
        bev_single: torch.Tensor,          # [B, C, H, W] single-frame BEV
        bev_temporal: torch.Tensor,        # [B, C, H, W] temporal-fused BEV
        lstm_hidden: Optional[torch.Tensor] = None,  # [B, C_h, H, W] LSTM hidden
        lstm_cell: Optional[torch.Tensor] = None,    # [B, C_h, H, W] LSTM cell
        pred_boxes: Optional[torch.Tensor] = None,   # [N, 7+] boxes
        pred_scores: Optional[torch.Tensor] = None,  # [N] scores
        pred_labels: Optional[torch.Tensor] = None,  # [N] labels
        batch_idx: int = 0,
    ):
        """Log a single timestep during sequence processing"""
        if not self.enabled or self.seq_dir is None:
            return
        if self.log_every > 1 and (t % self.log_every) != 0:
            return
        if batch_idx >= self.max_batches:
            return

        # Extract single batch
        bev_s = bev_single[batch_idx]      # [C, H, W]
        bev_t = bev_temporal[batch_idx]    # [C, H, W]

        # Compute feature energies
        bev_s_energy = self._bev_energy(bev_s)
        bev_t_energy = self._bev_energy(bev_t)
        
        # CRITICAL: Feature difference (shows LSTM modification)
        diff = (bev_t - bev_s).abs().mean(dim=0)  # [H, W]
        diff_mean = diff.mean().item()
        diff_max = diff.max().item()
        
        self.metrics['feature_diff_mean'].append(diff_mean)
        self.metrics['feature_diff_max'].append(diff_max)

        # Detection statistics
        if pred_scores is not None and len(pred_scores) > 0:
            mean_conf = pred_scores.mean().item()
            num_dets = len(pred_scores)
        else:
            mean_conf = 0.0
            num_dets = 0
        
        self.metrics['mean_confidence'].append(mean_conf)
        self.metrics['num_detections'].append(num_dets)

        # LSTM hidden state statistics
        h_mean = h_std = 0.0
        h_energy = None
        if lstm_hidden is not None:
            h = lstm_hidden[batch_idx]  # [C_h, H, W]
            h_mean = h.mean().item()
            h_std = h.std().item()
            h_energy = self._bev_energy(h)
            
            self.metrics['lstm_h_mean'].append(h_mean)
            self.metrics['lstm_h_std'].append(h_std)

        # Create visualization
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Feature maps
        axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(5)]
        # Row 2: LSTM analysis
        axes_row2 = [fig.add_subplot(gs[1, 0]),
                     fig.add_subplot(gs[1, 1]),
                     fig.add_subplot(gs[1, 2:4]),  # Wide plot
                     fig.add_subplot(gs[1, 4])]

        # === Row 1: Feature Visualizations ===
        
        # Single-frame BEV energy
        im1 = axes_row1[0].imshow(self._to_cpu_2d(bev_s_energy), cmap='viridis')
        axes_row1[0].set_title("Single-Frame BEV", fontsize=10)
        axes_row1[0].axis("off")
        plt.colorbar(im1, ax=axes_row1[0], fraction=0.046)

        # Temporal BEV energy
        im2 = axes_row1[1].imshow(self._to_cpu_2d(bev_t_energy), cmap='viridis')
        axes_row1[1].set_title("Temporal BEV", fontsize=10)
        axes_row1[1].axis("off")
        plt.colorbar(im2, ax=axes_row1[1], fraction=0.046)

        # Feature difference (CRITICAL!)
        im3 = axes_row1[2].imshow(self._to_cpu_2d(diff), cmap='hot')
        axes_row1[2].set_title(f"Diff (μ={diff_mean:.5f})", fontsize=10, fontweight='bold')
        axes_row1[2].axis("off")
        plt.colorbar(im3, ax=axes_row1[2], fraction=0.046)

        # High-difference regions
        threshold = diff.mean() + diff.std()
        diff_mask = (diff > threshold).float()
        im4 = axes_row1[3].imshow(self._to_cpu_2d(diff_mask), cmap='RdYlGn', vmin=0, vmax=1)
        axes_row1[3].set_title(f"High-Diff Regions", fontsize=10)
        axes_row1[3].axis("off")

        # Relative change
        rel_change = diff / (bev_s_energy + 1e-6)
        im5 = axes_row1[4].imshow(self._to_cpu_2d(rel_change), cmap='coolwarm', vmin=0, vmax=0.5)
        axes_row1[4].set_title("Relative Change", fontsize=10)
        axes_row1[4].axis("off")
        plt.colorbar(im5, ax=axes_row1[4], fraction=0.046)

        # === Row 2: LSTM Analysis ===

        # LSTM hidden state energy
        if h_energy is not None:
            im6 = axes_row2[0].imshow(self._to_cpu_2d(h_energy), cmap='plasma')
            axes_row2[0].set_title(f"LSTM Hidden\n(μ={h_mean:.3f})", fontsize=10)
            axes_row2[0].axis("off")
            plt.colorbar(im6, ax=axes_row2[0], fraction=0.046)
        else:
            axes_row2[0].text(0.5, 0.5, "No LSTM\nHidden", ha='center', va='center', fontsize=12)
            axes_row2[0].axis("off")

        # LSTM hidden channels (RGB visualization)
        if lstm_hidden is not None:
            h = lstm_hidden[batch_idx]
            if h.shape[0] >= 3:
                h_rgb = torch.stack([h[0], h[1], h[2]], dim=-1)
                h_rgb = (h_rgb - h_rgb.min()) / (h_rgb.max() - h_rgb.min() + 1e-8)
                axes_row2[1].imshow(self._to_cpu_2d(h_rgb.permute(2, 0, 1)).transpose(1, 2, 0))
            else:
                axes_row2[1].imshow(self._to_cpu_2d(h[0] if h.shape[0] > 0 else h_energy), cmap='viridis')
            axes_row2[1].set_title("LSTM Channels", fontsize=10)
            axes_row2[1].axis("off")
        else:
            axes_row2[1].text(0.5, 0.5, "No LSTM\nHidden", ha='center', va='center', fontsize=12)
            axes_row2[1].axis("off")

        # Timeline plot (CRITICAL FOR DIAGNOSIS!)
        ax_timeline = axes_row2[2]
        if len(self.metrics['mean_confidence']) > 1:
            timesteps = range(len(self.metrics['mean_confidence']))
            
            # Dual y-axis
            ax_conf = ax_timeline
            ax_diff = ax_timeline.twinx()
            
            # Plot confidence
            line1 = ax_conf.plot(timesteps, self.metrics['mean_confidence'], 
                                'o-', color='blue', linewidth=2, markersize=4, 
                                label='Confidence')
            ax_conf.set_xlabel('Timestep', fontsize=9)
            ax_conf.set_ylabel('Confidence', color='blue', fontsize=9)
            ax_conf.tick_params(axis='y', labelcolor='blue')
            ax_conf.grid(True, alpha=0.3)
            
            # Plot feature diff
            line2 = ax_diff.plot(timesteps, self.metrics['feature_diff_mean'], 
                                's-', color='red', linewidth=2, markersize=4,
                                label='Feature Diff')
            ax_diff.set_ylabel('Feature Diff', color='red', fontsize=9)
            ax_diff.tick_params(axis='y', labelcolor='red')
            
            # Trend line
            if len(timesteps) > 2:
                z = np.polyfit(list(timesteps), self.metrics['mean_confidence'], 1)
                p = np.poly1d(z)
                ax_conf.plot(timesteps, p(timesteps), "--", color='cyan', 
                           alpha=0.7, linewidth=1.5)
                
                # Diagnostic text
                trend_str = f"Trend: {z[0]:+.5f}/step"
                if z[0] > 0.005:
                    trend_str += " ✓"
                    color = 'green'
                elif abs(z[0]) < 0.005:
                    trend_str += " ⚠"
                    color = 'orange'
                else:
                    trend_str += " ✗"
                    color = 'red'
                ax_conf.text(0.02, 0.98, trend_str, transform=ax_conf.transAxes,
                           fontsize=9, verticalalignment='top', color=color,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax_timeline.set_title("Temporal Evolution", fontsize=10, fontweight='bold')
        else:
            ax_timeline.text(0.5, 0.5, f"Timestep {t}\nNeed >1 frame", 
                           ha='center', va='center', fontsize=10)
            ax_timeline.axis("off")

        # Summary text
        axes_row2[3].axis("off")
        summary_text = (
            f"Timestep: {t}\n"
            f"{'─'*15}\n"
            f"Dets: {num_dets}\n"
            f"Conf: {mean_conf:.3f}\n"
            f"{'─'*15}\n"
            f"Diff μ: {diff_mean:.5f}\n"
            f"Diff σ: {diff_max:.5f}\n"
        )
        if lstm_hidden is not None:
            summary_text += (
                f"{'─'*15}\n"
                f"LSTM μ: {h_mean:.3f}\n"
                f"LSTM σ: {h_std:.3f}\n"
            )
        
        axes_row2[3].text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                         verticalalignment='center')

        # Overall title
        info_str = f"Frame {t} | Dets={num_dets} | Conf={mean_conf:.3f} | Diff={diff_mean:.5f}"
        fig.suptitle(info_str, fontsize=12, fontweight='bold')

        # Save
        out_png = os.path.join(self.seq_dir, f"t{t:03d}_b{batch_idx}.png")
        fig.savefig(out_png, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Save data
        save_dict = {
            'bev_single_energy': self._to_cpu_2d(bev_s_energy),
            'bev_temporal_energy': self._to_cpu_2d(bev_t_energy),
            'diff': self._to_cpu_2d(diff),
            'diff_mean': diff_mean,
            'diff_max': diff_max,
            'num_detections': num_dets,
            'mean_confidence': mean_conf,
        }
        
        if lstm_hidden is not None:
            save_dict['lstm_hidden_energy'] = self._to_cpu_2d(h_energy)
        
        if pred_boxes is not None and len(pred_boxes) > 0:
            save_dict['pred_boxes'] = pred_boxes.detach().cpu().numpy()
            save_dict['pred_scores'] = pred_scores.detach().cpu().numpy()
            if pred_labels is not None:
                save_dict['pred_labels'] = pred_labels.detach().cpu().numpy()
        
        out_npz = os.path.join(self.seq_dir, f"t{t:03d}_b{batch_idx}.npz")
        np.savez_compressed(out_npz, **save_dict)

    def finish_sequence(self):
        """Generate summary plots for the sequence"""
        if not self.enabled or self.seq_dir is None:
            return
        
        if len(self.metrics['mean_confidence']) < 2:
            self.seq_dir = None
            return

        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        timesteps = range(len(self.metrics['mean_confidence']))

        # Plot 1: Confidence over time
        ax = axes[0, 0]
        ax.plot(timesteps, self.metrics['mean_confidence'], 'o-', linewidth=2, markersize=5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Mean Detection Confidence')
        ax.set_title('Confidence vs Time (Should INCREASE!)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if len(timesteps) > 2:
            z = np.polyfit(list(timesteps), self.metrics['mean_confidence'], 1)
            p = np.poly1d(z)
            ax.plot(timesteps, p(timesteps), "r--", alpha=0.8, linewidth=2,
                   label=f'Trend: {z[0]:+.5f}/step')
            ax.legend(fontsize=10)

        # Plot 2: Number of detections
        ax = axes[0, 1]
        ax.plot(timesteps, self.metrics['num_detections'], 's-', 
               linewidth=2, markersize=5, color='green')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Number of Detections')
        ax.set_title('Detection Count vs Time')
        ax.grid(True, alpha=0.3)

        # Plot 3: Feature difference
        ax = axes[1, 0]
        ax.plot(timesteps, self.metrics['feature_diff_mean'], 'o-', 
               linewidth=2, markersize=5, color='red', label='Mean')
        ax.plot(timesteps, self.metrics['feature_diff_max'], 's-', 
               linewidth=2, markersize=5, color='orange', alpha=0.6, label='Max')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Feature Difference')
        ax.set_title('LSTM Modification Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: LSTM hidden statistics
        ax = axes[1, 1]
        if len(self.metrics['lstm_h_mean']) > 0:
            ax2 = ax.twinx()
            line1 = ax.plot(timesteps, self.metrics['lstm_h_mean'], 'o-', 
                           linewidth=2, markersize=5, color='blue', label='Mean')
            line2 = ax2.plot(timesteps, self.metrics['lstm_h_std'], 's-', 
                            linewidth=2, markersize=5, color='purple', label='Std')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('LSTM Hidden Mean', color='blue')
            ax2.set_ylabel('LSTM Hidden Std', color='purple')
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='purple')
            ax.set_title('LSTM Hidden State Statistics')
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
        else:
            ax.text(0.5, 0.5, 'No LSTM Hidden Data', 
                   ha='center', va='center', fontsize=12)

        fig.tight_layout()
        summary_path = os.path.join(self.seq_dir, "sequence_summary.png")
        fig.savefig(summary_path, dpi=120)
        plt.close(fig)

        # Print diagnostic summary
        print(f"\n{'='*70}")
        print(f"SEQUENCE SUMMARY: {os.path.basename(self.seq_dir)}")
        print(f"{'='*70}")
        print(f"Total timesteps: {len(timesteps)}")
        print(f"Mean confidence: {np.mean(self.metrics['mean_confidence']):.4f}")
        
        if len(timesteps) > 2:
            z = np.polyfit(list(timesteps), self.metrics['mean_confidence'], 1)
            print(f"Confidence trend: {z[0]:+.6f} per timestep")
            
            if z[0] > 0.005:
                print(f"✅ GOOD: Confidence INCREASING → Temporal helping!")
            elif abs(z[0]) < 0.005:
                print(f"⚠️  WARNING: Confidence FLAT → Temporal might be ignored")
            else:
                print(f"❌ BAD: Confidence DECREASING → Problem detected!")
        
        print(f"Mean #detections: {np.mean(self.metrics['num_detections']):.1f}")
        print(f"Mean feature diff: {np.mean(self.metrics['feature_diff_mean']):.6f}")
        
        if len(self.metrics['lstm_h_mean']) > 0:
            print(f"Mean LSTM hidden: {np.mean(self.metrics['lstm_h_mean']):.4f}")
            print(f"Mean LSTM std: {np.mean(self.metrics['lstm_h_std']):.4f}")
        
        print(f"{'='*70}\n")
        
        self.seq_dir = None