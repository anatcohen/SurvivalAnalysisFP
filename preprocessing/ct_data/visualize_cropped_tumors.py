import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

matplotlib.use('TkAgg')


class TumorViewer:
    def __init__(self, downsampled_ct, subject_id):
        """
        Initialize the TumorViewer with downsampled CT data.

        Args:
            downsampled_ct (np.ndarray): Downsampled CT data of shape (32, 64, 64).
            subject_id (str): Subject ID for display purposes.
        """
        self.downsampled_ct = downsampled_ct
        self.subject_id = subject_id
        self.current_slice = 0 # Start at the first slice

        # Set up the figure with patient ID in window title
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title(f"{subject_id} Tumor Viewer")

        # Add main title at the top of the figure
        self.fig.suptitle(f"{subject_id} Downsampled Tumor Region", fontsize=16)

        # Adjust subplot positions to make room for the main title
        self.ax = plt.subplot(111)
        plt.subplots_adjust(top=0.9)  # Make room for the main title

        # Display initial slice
        self.im = self.ax.imshow(self.downsampled_ct[self.current_slice], cmap='gray', aspect='auto', vmin=0, vmax=1)
        self.ax.set_title(f'Slice {self.current_slice + 1}/{self.downsampled_ct.shape[0]}')

        # Add text instructions
        self.fig.text(0.5, 0.02,
                      'Use arrow keys (↑↓) or mouse wheel to navigate slices | Press Q to quit',
                      ha='center', fontsize=10)

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Initialize display
        self.update_display()

    def on_key(self, event):
        """
        Handle key press events for slice navigation.
        """
        if event.key == 'up':
            self.current_slice = min(self.current_slice + 1, self.downsampled_ct.shape[0] - 1)
            self.update_display()
        elif event.key == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
            self.update_display()
        elif event.key == 'q':
            plt.close(self.fig)

    def on_scroll(self, event):
        """
        Handle mouse scroll events for slice navigation.
        """
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.downsampled_ct.shape[0] - 1)
        elif event.button == 'down':
            self.current_slice = max(self.current_slice - 1, 0)
        self.update_display()

    def update_display(self):
        """
        Update the displayed slice.
        """
        self.im.set_data(self.downsampled_ct[self.current_slice])
        self.ax.set_title(f'Slice {self.current_slice + 1}/{self.downsampled_ct.shape[0]}')
        self.fig.canvas.draw_idle()

    def show(self):
        """
        Show the viewer.
        """
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load downsampled data for a specific patient
    data_dir = "../../data/processed_data"
    subject_id = "LUNG1-001"  # Replace with actual subject ID
    downsampled_ct = np.load(os.path.join(data_dir, f"{subject_id}_downsampled.npy"))

    # Create and show the viewer
    viewer = TumorViewer(downsampled_ct, subject_id)
    viewer.show()