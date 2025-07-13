import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk

matplotlib.use('TkAgg')

# Loading CSV with patient IDs and DICOM paths
data_dir = "../../data"
df = pd.read_csv(os.path.join(data_dir, "CT_RTSTRUCT_locations.csv"))

patient_ind = 0
subject_id = df['Subject ID'][patient_ind]

# Construct full paths using os.path.join
RTSTRUCT_loc = os.path.join(data_dir, df['RTSTRUCT Location'][patient_ind], '1-1.dcm')
CT_loc = os.path.join(data_dir, df['CT Location'][patient_ind])

print(f"Subject ID: {subject_id}")
print(f"RTSTRUCT Location: {RTSTRUCT_loc}")
print(f"CT Location: {CT_loc}")

print(f"Loading data for Subject ID: {subject_id}")

# Loading RTSTRUCT
rtstruct = pydicom.dcmread(RTSTRUCT_loc)

# Loop through structures to find GTV-1
roi_number = None
for item in rtstruct.StructureSetROISequence:
    if item.ROIName == "GTV-1":
        roi_number = item.ROINumber
        break

# Extract contours for GTV-1
contour_data = []
for contour in rtstruct.ROIContourSequence:
    if contour.ReferencedROINumber == roi_number:
        for contour_seq in contour.ContourSequence:
            coords = contour_seq.ContourData
            contour_data.append(coords)

# Load CT data
reader = sitk.ImageSeriesReader()
series = reader.GetGDCMSeriesFileNames(CT_loc)
reader.SetFileNames(series)
ct_image = reader.Execute()
ct_array = sitk.GetArrayFromImage(ct_image)

# Get image spacing and origin for proper coordinate transformation
spacing = np.array(ct_image.GetSpacing())
origin = np.array(ct_image.GetOrigin())

print(f"CT Array Shape: {ct_array.shape}")
print(f"Spacing: {spacing}")
print(f"Origin: {origin}")


def find_starting_slice(contour_data, spacing, origin, ct_shape):
    """Find the slice that is one below the first contour of GTV-1"""
    if not contour_data:
        print("No contour data found, using middle slice")
        return ct_shape[0] // 2

    # Get all Z coordinates from contours
    z_coords = []
    for contour in contour_data:
        points = np.array(contour).reshape(-1, 3)
        z_coord = points[0, 2]  # All points in a contour have same Z
        z_coords.append(z_coord)

    # Find the first (minimum) Z coordinate
    first_z = min(z_coords)

    # Convert to slice index
    first_slice_idx = int(round((first_z - origin[2]) / spacing[2]))
    first_slice_idx = max(0, min(first_slice_idx, ct_shape[0] - 1))

    # One slice below (considering that slice indices increase from superior to inferior)
    starting_slice = max(0, first_slice_idx - 1)

    print(f"First contour Z coordinate: {first_z:.3f} mm")
    print(f"First contour slice index: {first_slice_idx}")
    print(f"Starting slice (one below): {starting_slice}")

    return starting_slice


def plot_interactive_viewer():
    class CTViewer:
        def __init__(self, ct_array, contour_data, spacing, origin, subject_id):
            self.ct_array = ct_array
            self.contour_data = contour_data
            self.spacing = spacing
            self.origin = origin
            self.subject_id = subject_id

            # Find smart starting slice
            self.current_slice = find_starting_slice(contour_data, spacing, origin, ct_array.shape)

            # Set up the figure with patient ID in window title
            self.fig = plt.figure(figsize=(14, 8))
            self.fig.canvas.manager.set_window_title(f"{subject_id} GTV-1")

            # Add main title at the top of the figure
            self.fig.suptitle(f"{subject_id} CT scan (GTV-1 contours highlighted)", fontsize=16)

            # Adjust subplot positions to make room for the main title
            self.ax1 = plt.subplot(121)
            self.ax2 = plt.subplot(122, projection='3d')
            plt.subplots_adjust(top=0.9)  # Make room for the main title

            # Display initial slice
            self.im = self.ax1.imshow(ct_array[self.current_slice], cmap='gray', aspect='auto')
            self.ax1.set_title(f'Axial Slice {self.current_slice}/{ct_array.shape[0] - 1}')

            # Process ROI contours and organize by Z-coordinate
            self.contours_by_z = {}
            self.xs, self.ys, self.zs = [], [], []
            self.z_coords = []  # Store all Z coordinates (including duplicates)

            for i, contour in enumerate(contour_data):
                points = np.array(contour).reshape(-1, 3)
                z_coord = points[0, 2]  # All points in a contour have same Z

                # Handle multiple contours at same Z
                if z_coord not in self.contours_by_z:
                    self.contours_by_z[z_coord] = []
                self.contours_by_z[z_coord].append((i, points))

                self.xs.append(points[:, 0])
                self.ys.append(points[:, 1])
                self.zs.append(points[:, 2])
                self.z_coords.append(z_coord)

            # Get unique Z coordinates for analysis
            unique_z_coords = sorted(set(self.z_coords))
            print(f"Total contours: {len(contour_data)}")
            print(f"Unique Z coordinates: {len(unique_z_coords)}")

            # Find duplicate Z coordinates
            z_counts = {}
            for z in self.z_coords:
                z_counts[z] = z_counts.get(z, 0) + 1

            duplicates = {z: count for z, count in z_counts.items() if count > 1}
            if duplicates:
                print(f"\nWarning: Found multiple contours at same Z coordinates:")
                for z, count in duplicates.items():
                    print(f"  Z={z:.3f} mm: {count} contours")

            # Check for contours that might be too close together
            if len(unique_z_coords) > 1:
                z_diffs = np.diff(unique_z_coords)
                min_diff = np.min(z_diffs)
                print(f"\nMinimum Z spacing between contours: {min_diff:.3f} mm")
                print(f"Slice spacing: {self.spacing[2]:.3f} mm")

            # Calculate fixed axis limits for 3D plot with some padding
            if len(self.xs) > 0:
                x_min = min([min(x) for x in self.xs])
                x_max = max([max(x) for x in self.xs])
                y_min = min([min(y) for y in self.ys])
                y_max = max([max(y) for y in self.ys])
                z_min = min([min(z) for z in self.zs])
                z_max = max([max(z) for z in self.zs])

                # Add 10% padding to each dimension
                x_range = x_max - x_min
                y_range = y_max - y_min
                z_range = z_max - z_min

                padding = 0.1
                self.x_limits = [x_min - padding * x_range, x_max + padding * x_range]
                self.y_limits = [y_min - padding * y_range, y_max + padding * y_range]
                self.z_limits = [z_min - padding * z_range, z_max + padding * z_range]
            else:
                # Default limits if no contours
                self.x_limits = [-100, 100]
                self.y_limits = [-100, 100]
                self.z_limits = [origin[2], origin[2] + ct_array.shape[0] * spacing[2]]

            # Plot ROI points on slice
            self.roi_lines = []

            # Create empty line objects that will be updated
            for _ in range(10):  # Pre-create some line objects (adjust if you expect more contours per slice)
                line, = self.ax1.plot([], linestyle='-')
                self.roi_lines.append(line)
                line.set_visible(False)

            # 3D view - plot all contours in gray initially
            self.contour_lines_3d = []
            for x, y, z in zip(self.xs, self.ys, self.zs):
                line, = self.ax2.plot(x, y, z, 'gray', alpha=0.3, linewidth=1.5)
                self.contour_lines_3d.append(line)

            # Placeholder for highlighted contours (now a list)
            self.highlighted_contours = []

            # Add current slice indicator in 3D view
            self.slice_indicator = None

            # Set fixed axis limits
            self.ax2.set_xlim(self.x_limits)
            self.ax2.set_ylim(self.y_limits)
            self.ax2.set_zlim(self.z_limits)

            self.ax2.set_xlabel('X (mm)')
            self.ax2.set_ylabel('Y (mm)')
            self.ax2.set_zlabel('Z (mm)')
            self.ax2.set_title('3D Contour View')

            # Add text instructions - adjusted position for main title
            self.fig.text(0.5, 0.02,
                          'Use arrow keys (↑↓) or mouse wheel to navigate slices | Press Q to quit',
                          ha='center', fontsize=10)

            # Connect events
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

            # Initialize
            self.find_contour_slices()  # Add diagnostic output
            self.update_display()

        def on_key(self, event):
            if event.key == 'up':
                self.current_slice = min(self.current_slice + 1, self.ct_array.shape[0] - 1)
                self.update_display()
            elif event.key == 'down':
                self.current_slice = max(self.current_slice - 1, 0)
                self.update_display()
            elif event.key == 'q':
                plt.close(self.fig)

        def on_scroll(self, event):
            if event.button == 'up':
                self.current_slice = min(self.current_slice + 1, self.ct_array.shape[0] - 1)
            else:
                self.current_slice = max(self.current_slice - 1, 0)
            self.update_display()

        def update_display(self):
            # Update CT slice
            self.im.set_data(self.ct_array[self.current_slice])
            self.ax1.set_title(f'Axial Slice {self.current_slice}/{self.ct_array.shape[0] - 1}')

            # Find ROI points for this slice
            z_coord = self.origin[2] + self.current_slice * self.spacing[2]
            tolerance = self.spacing[2] / 2

            # Remove previous highlighted contours FIRST, before any other operations
            for contour in self.highlighted_contours:
                if contour is not None:
                    contour.remove()
            self.highlighted_contours = []

            # Reset all contours to gray
            for line in self.contour_lines_3d:
                line.set_color('gray')
                line.set_alpha(0.3)
                line.set_linewidth(1.5)

            # Find and highlight matching contours (multiple possible per slice)
            slice_x, slice_y = [], []
            highlighted_indices = []

            for idx, (x, y, z) in enumerate(zip(self.xs, self.ys, self.zs)):
                z_mean = np.mean(z)
                if abs(z_mean - z_coord) < tolerance:
                    highlighted_indices.append(idx)

                    # Convert to pixel coordinates for 2D display
                    px = (x - self.origin[0]) / self.spacing[0]
                    py = (y - self.origin[1]) / self.spacing[1]
                    slice_x.extend(px)
                    slice_y.extend(py)

                    # Add NaN to create separation between contours in scatter plot
                    slice_x.append(np.nan)
                    slice_y.append(np.nan)

            # Highlight all matching contours in 3D view
            for idx in highlighted_indices:
                self.contour_lines_3d[idx].set_color('red')
                self.contour_lines_3d[idx].set_alpha(1.0)
                self.contour_lines_3d[idx].set_linewidth(3)

                # Draw each as a highlighted separate line
                x = self.xs[idx]
                y = self.ys[idx]
                z = self.zs[idx]

                highlighted_line, = self.ax2.plot(
                    x, y, z, 'yellow', alpha=1.0, linewidth=1.5,
                    marker='*', markersize=0.5, markevery=5
                )
                self.highlighted_contours.append(highlighted_line)

            # Update 2D contour display
            # First hide all lines
            for line in self.roi_lines:
                line.set_visible(False)

            # Display each contour on the current slice
            if highlighted_indices:
                for i, idx in enumerate(highlighted_indices):
                    if i < len(self.roi_lines):
                        x = self.xs[idx]
                        y = self.ys[idx]

                        # Convert to pixel coordinates
                        px = (x - self.origin[0]) / self.spacing[0]
                        py = (y - self.origin[1]) / self.spacing[1]

                        # Close the contour by adding first point at the end
                        px = np.append(px, px[0])
                        py = np.append(py, py[0])

                        # Update line data
                        self.roi_lines[i].set_data(px, py)
                        self.roi_lines[i].set_visible(True)

                        self.roi_lines[i].set_color("yellow")
                        self.roi_lines[i].set_linewidth(1.5)

            # Update slice indicator plane in 3D view
            if self.slice_indicator:
                self.slice_indicator.remove()

            # Draw a semi-transparent plane at current slice position
            if len(self.xs) > 0:
                # Use the fixed limits for the plane
                xx, yy = np.meshgrid(
                    np.linspace(self.x_limits[0], self.x_limits[1], 10),
                    np.linspace(self.y_limits[0], self.y_limits[1], 10)
                )
                zz = np.ones_like(xx) * z_coord
                self.slice_indicator = self.ax2.plot_surface(
                    xx, yy, zz, alpha=0.1, color='cyan'
                )

            # Adjust 3D view angle for better visibility
            self.ax2.view_init(elev=20, azim=45)

            # Re-apply the fixed limits (to prevent auto-scaling)
            self.ax2.set_xlim(self.x_limits)
            self.ax2.set_ylim(self.y_limits)
            self.ax2.set_zlim(self.z_limits)

            # Update 3D plot title
            if highlighted_indices:
                if len(highlighted_indices) > 1:
                    # Show the actual contour numbers being displayed
                    contour_numbers = [idx + 1 for idx in highlighted_indices]
                    contour_range = f"{min(contour_numbers)}-{max(contour_numbers)}" if max(contour_numbers) - min(
                        contour_numbers) == len(contour_numbers) - 1 else ",".join(map(str, contour_numbers))
                    self.ax2.set_title(f'3D Contour View ({contour_range}/{len(self.contour_lines_3d)})')
                else:
                    self.ax2.set_title(f'3D Contour View ({highlighted_indices[0] + 1}/{len(self.contour_lines_3d)})')
            else:
                self.ax2.set_title('3D Contour View')

            self.fig.canvas.draw_idle()

        def find_contour_slices(self):
            """Helper method to find which CT slices correspond to each contour"""
            print("\nContour to slice mapping:")
            print("-" * 50)

            unique_z_coords = sorted(set(self.z_coords))
            for i, z_coord in enumerate(unique_z_coords):
                # Find corresponding CT slice
                slice_idx = int(round((z_coord - self.origin[2]) / self.spacing[2]))
                slice_idx = max(0, min(slice_idx, self.ct_array.shape[0] - 1))

                # Check if this slice would actually show this contour
                z_slice = self.origin[2] + slice_idx * self.spacing[2]
                tolerance = self.spacing[2] / 2
                will_display = abs(z_coord - z_slice) < tolerance

                # Count how many contours at this Z
                count = self.z_coords.count(z_coord)
                if count > 1:
                    print(
                        f"Contour {i + 1}: Z={z_coord:.3f} mm → Slice {slice_idx} (Z={z_slice:.3f} mm) - Display: {will_display} [{count} contours here]")
                else:
                    print(
                        f"Contour {i + 1}: Z={z_coord:.3f} mm → Slice {slice_idx} (Z={z_slice:.3f} mm) - Display: {will_display}")

            print("-" * 50)

            # Check for slices that might show multiple contours
            slice_contour_count = {}
            for z_coord in self.z_coords:
                for slice_idx in range(self.ct_array.shape[0]):
                    z_slice = self.origin[2] + slice_idx * self.spacing[2]
                    if abs(z_coord - z_slice) < self.spacing[2] / 2:
                        if slice_idx not in slice_contour_count:
                            slice_contour_count[slice_idx] = []
                        slice_contour_count[slice_idx].append(z_coord)

            # Report slices with multiple contours
            multi_contour_slices = {k: v for k, v in slice_contour_count.items() if len(v) > 1}
            if multi_contour_slices:
                print("\nSlices with multiple contours:")
                for slice_idx, z_coords in multi_contour_slices.items():
                    print(f"  Slice {slice_idx}: {len(z_coords)} contours at Z={z_coords}")

        def show(self):
            plt.show()

    # Create and show the viewer
    viewer = CTViewer(ct_array, contour_data, spacing, origin, subject_id)
    viewer.show()


plot_interactive_viewer()