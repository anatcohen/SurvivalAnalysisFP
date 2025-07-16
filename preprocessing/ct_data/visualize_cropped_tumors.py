import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider, RadioButtons
import glob

matplotlib.use('TkAgg')


class EnhancedTumorViewer:
    def __init__(self, data_path=None, subject_id=None):
        """
        Initialize the Enhanced Tumor Viewer.

        Args:
            data_path: Path to .npz file containing processed data
        """
        # Load from file
        data = np.load(data_path)
        self.ct_array = data['ct']
        self.mask_array = data['mask']
        self.subject_id = subject_id

        # Load metadata if available
        self.metadata = {}
        if 'tumor_size_mm' in data:
            self.metadata['tumor_size_mm'] = data['tumor_size_mm']
        if 'tumor_center_mm' in data:
            self.metadata['tumor_center_mm'] = data['tumor_center_mm']

        # Denormalize CT values for display (assuming normalized to [-1, 1])
        self.ct_display = self.ct_array

        # Current display settings
        self.current_slice = self.ct_array.shape[0] // 2  # Start at middle
        self.current_axis = 0  # 0: axial, 1: coronal, 2: sagittal
        self.display_mode = 'overlay'  # 'ct', 'mask', 'overlay', 'side_by_side'

        # Setup figure
        self.setup_figure()

        # Initial display
        self.update_display()

    def setup_figure(self):
        """Setup the matplotlib figure with controls."""
        # Create figure with adjusted size
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title(f"{self.subject_id} - Tumor Viewer")

        # Create grid for layout
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 8, 1], width_ratios=[1, 4])

        # Main display axes
        self.ax_main = self.fig.add_subplot(gs[1, 1])

        # Slice slider
        self.ax_slider = self.fig.add_subplot(gs[2, 1])
        self.slice_slider = Slider(
            self.ax_slider, 'Slice',
            0, self.get_max_slice(),
            valinit=self.current_slice,
            valstep=1,
            valfmt='%d'
        )
        self.slice_slider.on_changed(self.on_slider_change)

        # Radio buttons for view mode
        self.ax_radio_view = self.fig.add_subplot(gs[1, 0])
        self.radio_view = RadioButtons(
            self.ax_radio_view,
            ('Overlay', 'CT Only', 'Mask Only', 'Side by Side'),
            active=0
        )
        self.radio_view.on_clicked(self.on_view_change)

        # Radio buttons for axis
        self.ax_radio_axis = self.fig.add_subplot(gs[0, 1])
        self.radio_axis = RadioButtons(
            self.ax_radio_axis,
            ('Axial', 'Coronal', 'Sagittal'),
            active=0,
            activecolor='blue'
        )
        self.radio_axis.on_clicked(self.on_axis_change)

        # Add title with metadata
        title = f"{self.subject_id} - Processed Tumor Region"
        if 'tumor_size_mm' in self.metadata:
            size = self.metadata['tumor_size_mm']
            title += f"\nTumor Size: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm"
        self.fig.suptitle(title, fontsize=14)

        # Instructions
        self.fig.text(0.5, 0.02,
                      'Use arrow keys (↑↓) or slider to navigate | Q to quit',
                      ha='center', fontsize=10)

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.9)

    def get_max_slice(self):
        """Get maximum slice number for current axis."""
        return [self.ct_array.shape[0], self.ct_array.shape[1], self.ct_array.shape[2]][self.current_axis] - 1

    def get_current_slice_data(self):
        """Get current slice data based on axis."""
        if self.current_axis == 0:  # Axial
            ct_slice = self.ct_display[self.current_slice, :, :]
            mask_slice = self.mask_array[self.current_slice, :, :]
        elif self.current_axis == 1:  # Coronal
            ct_slice = self.ct_display[:, self.current_slice, :]
            mask_slice = self.mask_array[:, self.current_slice, :]
        else:  # Sagittal
            ct_slice = self.ct_display[:, :, self.current_slice]
            mask_slice = self.mask_array[:, :, self.current_slice]

        return ct_slice, mask_slice

    def update_display(self):
        """Update the displayed image."""
        self.ax_main.clear()

        ct_slice, mask_slice = self.get_current_slice_data()

        if self.display_mode == 'overlay':
            # Display CT with mask overlay
            self.ax_main.imshow(ct_slice, cmap='gray', aspect='auto')

            # Create colored overlay for mask
            mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
            self.ax_main.imshow(mask_overlay, cmap='Reds', alpha=0.6, aspect='auto')

        elif self.display_mode == 'ct':
            # CT only
            self.ax_main.imshow(ct_slice, cmap='gray', aspect='auto')

        elif self.display_mode == 'mask':
            # Mask only
            self.ax_main.imshow(mask_slice, cmap='binary', aspect='auto')

        elif self.display_mode == 'side_by_side':
            # Clear and create two subplots
            self.ax_main.clear()
            self.ax_main.axis('off')

            # Create two subaxes
            ax1 = self.fig.add_axes([0.35, 0.25, 0.25, 0.5])
            ax2 = self.fig.add_axes([0.65, 0.25, 0.25, 0.5])

            # Left: CT with mask overlay
            ax1.imshow(ct_slice, cmap='gray', aspect='auto')
            mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax1.imshow(mask_overlay, cmap='Reds', alpha=0.5, aspect='auto')
            ax1.set_title('CT + Mask Overlay')
            ax1.axis('off')

            # Right: Masked tumor (CT * mask)
            masked_ct = ct_slice * mask_slice
            ax2.imshow(masked_ct, cmap='gray', aspect='auto')
            ax2.set_title('Masked Tumor')
            ax2.axis('off')

            # Store for cleanup
            self.temp_axes = [ax1, ax2]

        # Update title
        axis_names = ['Axial', 'Coronal', 'Sagittal']
        self.ax_main.set_title(
            f'{axis_names[self.current_axis]} - Slice {self.current_slice + 1}/{self.get_max_slice() + 1}'
        )

        # Add coordinate info
        if hasattr(self, 'temp_axes'):
            for ax in self.temp_axes[:-2]:  # Don't remove the last two we just created
                ax.remove()
            self.temp_axes = self.temp_axes[-2:]

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'up':
            self.current_slice = min(self.current_slice + 1, self.get_max_slice())
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
            self.slice_slider.set_val(self.current_slice)
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key in ['1', '2', '3']:
            # Quick axis switch
            self.current_axis = int(event.key) - 1
            self.current_slice = min(self.current_slice, self.get_max_slice())
            self.slice_slider.valmax = self.get_max_slice()
            self.slice_slider.set_val(self.current_slice)
            self.update_display()

    def on_scroll(self, event):
        """Handle mouse scroll events."""
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.get_max_slice())
        else:
            self.current_slice = max(self.current_slice - 1, 0)
        self.slice_slider.set_val(self.current_slice)

    def on_slider_change(self, val):
        """Handle slider change."""
        self.current_slice = int(val)
        self.update_display()

    def on_view_change(self, label):
        """Handle view mode change."""
        mode_map = {
            'Overlay': 'overlay',
            'CT Only': 'ct',
            'Mask Only': 'mask',
            'Side by Side': 'side_by_side'
        }
        self.display_mode = mode_map[label]

        # Clean up temporary axes if switching from side_by_side
        if hasattr(self, 'temp_axes') and self.display_mode != 'side_by_side':
            for ax in self.temp_axes:
                ax.remove()
            delattr(self, 'temp_axes')

        self.update_display()

    def on_axis_change(self, label):
        """Handle axis change."""
        axis_map = {'Axial': 0, 'Coronal': 1, 'Sagittal': 2}
        self.current_axis = axis_map[label]

        # Update slider range
        self.slice_slider.valmax = self.get_max_slice()
        self.current_slice = min(self.current_slice, self.get_max_slice())
        self.slice_slider.set_val(self.current_slice)

        self.update_display()

    def show(self):
        """Display the viewer."""
        plt.show()


if __name__ == "__main__":
    data_dir = "../../data"
    cropped_data_dir = "../../data/processed_data_physical"

    df = pd.read_csv(os.path.join(data_dir, "CT_RTSTRUCT_locations.csv"))

    patient_ind = 0
    subject_id = df['Subject ID'][patient_ind]

    data_path = os.path.join(cropped_data_dir, subject_id + '_processed.npz')

    viewer = EnhancedTumorViewer(data_path, subject_id)
    viewer.show()