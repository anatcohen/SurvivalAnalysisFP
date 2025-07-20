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
        """Create 3D visualization of the tumor."""
        # Clear main axis
        self.ax_main.clear()
        self.ax_main.axis('off')

        # Create 3D subplot
        if hasattr(self, 'ax_3d'):
            self.ax_3d.remove()

        self.ax_3d = self.fig.add_subplot(self.gs[1, 1], projection='3d')

        # Create bounding box vertices
        box_size = 64  # Assuming 64^3 cropping box
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

        # Draw bounding box with pink/red lines like in the reference
        for edge in edges:
            points = [vertices[edge[0]], vertices[edge[1]]]
            self.ax_3d.plot3D(*zip(*points), color='#FF6B6B', alpha=0.6, linewidth=2)

        # Create tumor surface using marching cubes
        try:
            # Use marching cubes to get surface mesh
            verts, faces, _, _ = measure.marching_cubes(self.mask_array, level=0.5, spacing=(1, 1, 1))

            # Plot the surface in green to match reference
            self.ax_3d.plot_trisurf(verts[:, 2], verts[:, 1], verts[:, 0],
                                    triangles=faces,
                                    color='#2ECC40',  # Bright green
                                    alpha=0.95,
                                    shade=True,
                                    edgecolor='none')
        except:
            # Fallback to voxel representation if marching cubes fails
            tumor_coords = np.where(self.mask_array > 0)
            if len(tumor_coords[0]) > 0:
                # Use voxels for visualization
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                # Create cubes for each voxel
                cubes = []
                # Downsample for performance
                max_voxels = 2000
                if len(tumor_coords[0]) > max_voxels:
                    indices = np.random.choice(len(tumor_coords[0]), max_voxels, replace=False)
                    positions = [(tumor_coords[2][i], tumor_coords[1][i], tumor_coords[0][i])
                                 for i in indices]
                else:
                    positions = [(tumor_coords[2][i], tumor_coords[1][i], tumor_coords[0][i])
                                 for i in range(len(tumor_coords[0]))]

                for pos in positions:
                    x, y, z = pos
                    # Create vertices for a small cube
                    r = 0.45  # cube radius
                    cube_verts = [
                        [x - r, y - r, z - r], [x + r, y - r, z - r], [x + r, y + r, z - r], [x - r, y + r, z - r],
                        # bottom
                        [x - r, y - r, z + r], [x + r, y - r, z + r], [x + r, y + r, z + r], [x - r, y + r, z + r]
                        # top
                    ]

                    # Define the 6 faces of the cube
                    cube_faces = [
                        [cube_verts[0], cube_verts[1], cube_verts[2], cube_verts[3]],  # bottom
                        [cube_verts[4], cube_verts[5], cube_verts[6], cube_verts[7]],  # top
                        [cube_verts[0], cube_verts[1], cube_verts[5], cube_verts[4]],  # front
                        [cube_verts[2], cube_verts[3], cube_verts[7], cube_verts[6]],  # back
                        [cube_verts[0], cube_verts[3], cube_verts[7], cube_verts[4]],  # left
                        [cube_verts[1], cube_verts[2], cube_verts[6], cube_verts[5]]  # right
                    ]
                    cubes.extend(cube_faces)

                # Create collection and add to plot
                cube_collection = Poly3DCollection(cubes, facecolors='#2ECC40',
                                                   edgecolors='#27AE60', alpha=0.95, linewidth=0.05)
                self.ax_3d.add_collection3d(cube_collection)

        # Remove all axis elements for clean look
        self.ax_3d.set_axis_off()

        # Set equal aspect ratio and limits
        self.ax_3d.set_xlim([0, box_size])
        self.ax_3d.set_ylim([0, box_size])
        self.ax_3d.set_zlim([0, box_size])

        # Set viewing angle similar to reference
        self.ax_3d.view_init(elev=15, azim=30)

        # Add title
        self.ax_3d.text2D(0.5, 0.95, '3D Tumor Visualization',
                          transform=self.ax_3d.transAxes,
                          ha='center', va='top', fontsize=14)

        # Set white background
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False

        # Hide slice controls
        self.ax_slider.set_visible(False)
        self.ax_radio_axis.set_visible(False)

    def update_display(self):
        """Update the displayed image."""
        # Clean up 3D elements if switching from 3D view
        if hasattr(self, 'ax_3d') and self.display_mode != '3d':
            self.ax_3d.remove()
            delattr(self, 'ax_3d')
            # Recreate main axis
            self.ax_main = self.fig.add_subplot(self.gs[1, 1])
            # Show slice controls
            self.ax_slider.set_visible(True)
            self.ax_radio_axis.set_visible(True)

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

    patient_ind = 13
    subject_id = df['Subject ID'][patient_ind]

    data_path = os.path.join(cropped_data_dir, subject_id + '_processed.npz')

    viewer = EnhancedTumorViewer(data_path, subject_id)
    viewer.show()