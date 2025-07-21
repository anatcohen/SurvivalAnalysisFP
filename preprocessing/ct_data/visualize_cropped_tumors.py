import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from skimage import measure

from config.paths import DATA_DIR

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
        self.display_mode = 'overlay'  # 'ct', 'mask', 'overlay', 'side_by_side', '3d'

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
        self.gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 8, 1], width_ratios=[1, 4])

        # Main display axes
        self.ax_main = self.fig.add_subplot(self.gs[1, 1])

        # Slice slider
        self.ax_slider = self.fig.add_subplot(self.gs[2, 1])
        self.slice_slider = Slider(
            self.ax_slider, 'Slice',
            0, self.get_max_slice(),
            valinit=self.current_slice,
            valstep=1,
            valfmt='%d'
        )
        self.slice_slider.on_changed(self.on_slider_change)

        # Radio buttons for view mode
        self.ax_radio_view = self.fig.add_subplot(self.gs[1, 0])
        self.radio_view = RadioButtons(
            self.ax_radio_view,
            ('Overlay', 'CT Only', 'Mask Only', 'Side by Side', '3D View'),
            active=0
        )
        self.radio_view.on_clicked(self.on_view_change)

        # Radio buttons for axis
        self.ax_radio_axis = self.fig.add_subplot(self.gs[0, 1])
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
                      'Use arrow keys (↑↓) or slider to navigate | Q to quit | For 3D: drag to rotate',
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

    def create_3d_view(self):
        """Create 3D visualization of the 4 augmented tumors."""
        # Clear main axis
        self.ax_main.clear()
        self.ax_main.axis('off')

        # Hide the radio buttons for axis selection in 3D mode
        self.ax_radio_axis.clear()
        self.ax_radio_axis.axis('off')

        # Clean up existing 3d axes if any
        if hasattr(self, 'ax_3d_list'):
            for ax in self.ax_3d_list:
                ax.remove()

        # Create 4 subplots for the 4 augmented versions with tighter spacing
        self.ax_3d_list = []
        titles = ['Original', 'Augmented 1', 'Augmented 2', 'Augmented 3']

        # Load augmented data
        augmented_dir = os.path.join(DATA_DIR, "CT_tumors_augmented")

        # Define custom positions for tighter layout
        # [left, bottom, width, height]
        subplot_width = 0.35
        subplot_height = 0.35
        h_spacing = -0.05  # Negative spacing for overlap
        v_spacing = 0  # Negative spacing for overlap

        # Center the grid but shift right to avoid left menu
        total_width = 2 * subplot_width + h_spacing
        total_height = 2 * subplot_height + v_spacing
        left_offset = 0.55 - total_width / 2  # Shifted from 0.5 to 0.55
        bottom_offset = 0.5 - total_height / 2 - 0.05  # Slight adjustment for title

        positions = [
            [left_offset, bottom_offset + subplot_height + v_spacing, subplot_width, subplot_height],  # Top left
            [left_offset + subplot_width + h_spacing, bottom_offset + subplot_height + v_spacing, subplot_width,
             subplot_height],  # Top right
            [left_offset, bottom_offset, subplot_width, subplot_height],  # Bottom left
            [left_offset + subplot_width + h_spacing, bottom_offset, subplot_width, subplot_height]  # Bottom right
        ]

        for idx, pos in enumerate(positions):
            # Create subplot with custom position
            ax_3d = self.fig.add_axes(pos, projection='3d')
            self.ax_3d_list.append(ax_3d)

            # Load augmented tumor data
            aug_file = os.path.join(augmented_dir, f"{self.subject_id}_{idx + 1}.npz")
            try:
                aug_data = np.load(aug_file)
                masked_ct = aug_data['masked_ct']

                # Extract mask from masked CT (non-zero values)
                mask = masked_ct > 0

                # Create bounding box vertices
                box_size = 64  # 64^3 cropping box
                vertices = [
                    [0, 0, 0], [box_size, 0, 0], [box_size, box_size, 0], [0, box_size, 0],
                    [0, 0, box_size], [box_size, 0, box_size], [box_size, box_size, box_size], [0, box_size, box_size]
                ]

                # Define edges of the bounding box
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                    [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                    [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical edges
                ]

                # Draw bounding box with pink/red lines
                for edge in edges:
                    points = [vertices[edge[0]], vertices[edge[1]]]
                    ax_3d.plot3D(*zip(*points), color='#FF6B6B', alpha=0.6, linewidth=1.5)

                # Create tumor surface using marching cubes
                try:
                    # Use marching cubes to get surface mesh
                    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=(1, 1, 1))

                    # Plot the surface in green
                    ax_3d.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0],
                                       triangles=faces,
                                       color='#2ECC40',  # Bright green
                                       alpha=0.95,
                                       shade=True,
                                       edgecolor='none')
                except:
                    # Fallback to voxel representation if marching cubes fails
                    tumor_coords = np.where(mask > 0)
                    if len(tumor_coords[0]) > 0:
                        # Downsample for performance
                        max_voxels = 500  # Less voxels per subplot
                        if len(tumor_coords[0]) > max_voxels:
                            indices = np.random.choice(len(tumor_coords[0]), max_voxels, replace=False)
                            ax_3d.scatter(tumor_coords[2][indices],
                                          tumor_coords[1][indices],
                                          tumor_coords[0][indices],
                                          c='#2ECC40', s=2, alpha=0.8)
                        else:
                            ax_3d.scatter(tumor_coords[2], tumor_coords[1], tumor_coords[0],
                                          c='#2ECC40', s=2, alpha=0.8)

                # Remove all axis elements for clean look
                ax_3d.set_axis_off()

                # Set equal aspect ratio and limits
                ax_3d.set_xlim([0, box_size])
                ax_3d.set_ylim([0, box_size])
                ax_3d.set_zlim([0, box_size])

                # Set viewing angle - same for all subplots for better comparison
                ax_3d.view_init(elev=20, azim=45)

                # Add title
                ax_3d.text2D(0.5, 0.95, titles[idx],
                             transform=ax_3d.transAxes,
                             ha='center', va='top', fontsize=12, weight='bold')

                # Set white background
                ax_3d.xaxis.pane.fill = False
                ax_3d.yaxis.pane.fill = False
                ax_3d.zaxis.pane.fill = False

            except Exception as e:
                # If file not found or error, show empty box with error message
                ax_3d.text2D(0.5, 0.5, f"Error loading\n{titles[idx]}",
                             transform=ax_3d.transAxes,
                             ha='center', va='center', fontsize=10, color='red')
                ax_3d.set_axis_off()

        # Hide slice controls and axis radio buttons
        self.ax_slider.set_visible(False)
        self.ax_radio_axis.set_visible(False)

        # Update main title for 3D view
        self.fig.suptitle(f"{self.subject_id} - Augmented Tumor Visualizations", fontsize=14)

    def update_display(self):
        """Update the displayed image."""
        # Clean up 3D elements if switching from 3D view
        if hasattr(self, 'ax_3d_list') and self.display_mode != '3d':
            for ax in self.ax_3d_list:
                ax.remove()
            delattr(self, 'ax_3d_list')
            # Recreate main axis
            self.ax_main = self.fig.add_subplot(self.gs[1, 1])
            # Recreate and show axis radio buttons
            self.ax_radio_axis = self.fig.add_subplot(self.gs[0, 1])
            self.radio_axis = RadioButtons(
                self.ax_radio_axis,
                ('Axial', 'Coronal', 'Sagittal'),
                active=self.current_axis,
                activecolor='blue'
            )
            self.radio_axis.on_clicked(self.on_axis_change)
            # Show slice controls
            self.ax_slider.set_visible(True)
            self.ax_radio_axis.set_visible(True)
            # Restore original title
            title = f"{self.subject_id} - Processed Tumor Region"
            if 'tumor_size_mm' in self.metadata:
                size = self.metadata['tumor_size_mm']
                title += f"\nTumor Size: {size[0]:.1f} × {size[1]:.1f} × {size[2]:.1f} mm"
            self.fig.suptitle(title, fontsize=14)

        if self.display_mode == '3d':
            self.create_3d_view()
            self.fig.canvas.draw_idle()
            return

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
        # Don't process arrow keys in 3D view
        if self.display_mode == '3d':
            if event.key == 'q':
                plt.close(self.fig)
            return

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
        # Don't process scroll events in 3D view
        if self.display_mode == '3d':
            return

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
            'Side by Side': 'side_by_side',
            '3D View': '3d'
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
    data_dir = DATA_DIR
    cropped_data_dir = os.path.join(DATA_DIR, 'processed_data_physical')

    df = pd.read_csv(os.path.join(data_dir, "CT_RTSTRUCT_locations.csv"))

    patient_ind = 4
    subject_id = df['Subject ID'][patient_ind]

    data_path = os.path.join(cropped_data_dir, subject_id + '_processed.npz')

    viewer = EnhancedTumorViewer(data_path, subject_id)
    viewer.show()