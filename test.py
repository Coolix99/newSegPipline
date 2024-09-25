import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def compute_divergence(vector_field):
    """
    Compute the divergence of a 3D vector field.

    Parameters:
    - vector_field: numpy array of shape (nx, ny, nz, 3)

    Returns:
    - divergence: numpy array of shape (nx, ny, nz)
    """
    vx = vector_field[..., 0]
    vy = vector_field[..., 1]
    vz = vector_field[..., 2]

    dvx_dx = np.gradient(vx, axis=0)
    dvy_dy = np.gradient(vy, axis=1)
    dvz_dz = np.gradient(vz, axis=2)

    divergence = dvx_dx + dvy_dy + dvz_dz

    return divergence

def compute_curl(vector_field):
    """
    Compute the curl of a 3D vector field.

    Parameters:
    - vector_field: numpy array of shape (nx, ny, nz, 3)

    Returns:
    - curl: numpy array of shape (nx, ny, nz, 3)
    """
    vx = vector_field[..., 0]
    vy = vector_field[..., 1]
    vz = vector_field[..., 2]

    dvz_dy = np.gradient(vz, axis=1)
    dvy_dz = np.gradient(vy, axis=2)
    dvx_dz = np.gradient(vx, axis=2)
    dvz_dx = np.gradient(vz, axis=0)
    dvy_dx = np.gradient(vy, axis=0)
    dvx_dy = np.gradient(vx, axis=1)

    curl_x = dvz_dy - dvy_dz
    curl_y = dvx_dz - dvz_dx
    curl_z = dvy_dx - dvx_dy

    curl = np.stack((curl_x, curl_y, curl_z), axis=-1)

    return curl

def compute_boundary_proxy(vector_field, sigma=1.0):
    """
    Compute a boundary proxy from a 3D vector field and distinguish between centers and boundaries.

    Parameters:
    - vector_field: numpy array of shape (nx, ny, nz, 3)
    - sigma: float, standard deviation for Gaussian smoothing

    Returns:
    - labels: numpy array of shape (nx, ny, nz), with labels:
              0: background, 1: center, 2: boundary
    - divergence_norm: normalized divergence array
    - curl_norm: normalized curl magnitude array
    """
    # Apply Gaussian smoothing to each component of the vector field
    vx_smooth = gaussian_filter(vector_field[..., 0], sigma=sigma)
    vy_smooth = gaussian_filter(vector_field[..., 1], sigma=sigma)
    vz_smooth = gaussian_filter(vector_field[..., 2], sigma=sigma)
    
    # Re-normalize the vectors to unit length to maintain directionality
    magnitude = np.sqrt(vx_smooth**2 + vy_smooth**2 + vz_smooth**2)
    # Avoid division by zero
    magnitude[magnitude == 0] = 1
    vx_smooth /= magnitude
    vy_smooth /= magnitude
    vz_smooth /= magnitude
    
    # Stack the smoothed components back into a vector field
    vector_field_smooth = np.stack((vx_smooth, vy_smooth, vz_smooth), axis=-1)
    
    # Compute divergence and curl
    divergence = compute_divergence(vector_field_smooth)
    curl = compute_curl(vector_field_smooth)
    curl_magnitude = np.linalg.norm(curl, axis=-1)
    
  
    # Normalize divergence and curl magnitude to [0, 1]
    divergence_norm = (divergence - divergence.min()) / (divergence.max() - divergence.min() + 1e-8)
    curl_norm = (curl_magnitude - curl_magnitude.min()) / (curl_magnitude.max() - curl_magnitude.min() + 1e-8)
    
    # Define thresholds (you may need to adjust these based on your data)
    divergence_threshold = 0.5
    curl_threshold = 0.5
    
    # Create masks
    center_mask = (divergence_norm > divergence_threshold) 
    boundary_mask = (divergence_norm < divergence_threshold)
    
    # For visualization, create an array with different labels
    # 0: background, 1: center, 2: boundary
    labels = np.zeros(vector_field.shape[:-1], dtype=np.uint8)
    labels[center_mask] = 1
    labels[boundary_mask] = 2
    
    return labels, divergence_norm, curl_norm

# Example usage:
# Assume you have your vector_field numpy array of shape (nx, ny, nz, 3)
# Here, we'll create a sample vector field for demonstration.

def generate_sample_data(nx, ny, nz):
    """
    Generate a sample 3D vector field with two centers.
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing='ij')
    
    # Define two centers
    center1 = np.array([-0.5, 0, 0])
    center2 = np.array([0.5, 0, 0])
    
    # Compute vectors pointing to center1 and center2
    vec1 = np.stack((x_grid - center1[0], y_grid - center1[1], z_grid - center1[2]), axis=-1)
    vec2 = np.stack((x_grid - center2[0], y_grid - center2[1], z_grid - center2[2]), axis=-1)
    
    # Initialize the vector field
    vector_field = np.zeros((nx, ny, nz, 3))
    
    # Assign vectors based on a condition (e.g., x < 0)
    mask = x_grid < 0
    vector_field[mask] = vec1[mask]
    vector_field[~mask] = vec2[~mask]
    
    # Normalize vectors to unit length
    magnitude = np.linalg.norm(vector_field, axis=-1)
    nonzero = magnitude > 0
    vector_field[nonzero] /= magnitude[nonzero][..., np.newaxis]
    
    return vector_field

def plot_result(nuc_img,segmentation,pred_flows):
    import napari

    viewer = napari.Viewer()
    viewer.add_image(nuc_img, name='3D Nuc')
    viewer.add_labels(segmentation, name='seg')
    z, y, x = np.nonzero(segmentation) 
    origins = np.stack((z, y, x), axis=-1)
    flow_vector_field = pred_flows.transpose(1, 2, 3, 0)
    vectors = flow_vector_field[z, y, x]
    vector_data = np.stack((origins, vectors), axis=1)
    viewer.add_vectors(vector_data, name='3D Flow Field pred', edge_width=0.1, length=1, ndim=3,edge_color='blue')

    napari.run()

def real_example():
    import os
    from config import nuclei_folders_path, applyresult_folder_path
    from IO import get_JSON, getImage, load_compressed_array
    from CPC.std_op import prepareData
    nuclei_folder_list = os.listdir(nuclei_folders_path)
    nuclei_folder_list =['20220611_mAG-zGem_H2a-mcherry_96hpf_LM_B2_analyzed_nuclei']
    for nuclei_folder in nuclei_folder_list:
        print(nuclei_folder)
        nuclei_folder_path = os.path.join(nuclei_folders_path, nuclei_folder)
        MetaData = get_JSON(nuclei_folder_path)
        if not 'nuclei_image_MetaData' in MetaData:
            continue
        
        res_folder_path = os.path.join(applyresult_folder_path, nuclei_folder)
        MetaData_apply = get_JSON(res_folder_path)
        if not 'apply_MetaData' in MetaData_apply:
            continue

        print(MetaData)
        
        nuc_file_name = MetaData['nuclei_image_MetaData']['nuclei image file name']
        nuc_file_path = os.path.join(nuclei_folder_path, nuc_file_name)
        nuc_img = getImage(nuc_file_path)
        scale = np.array(MetaData['nuclei_image_MetaData']['XYZ size in mum']).copy()
        scale[0], scale[2] = scale[2], scale[0]
        nuclei,profile=prepareData(nuc_img,scale)
        print(nuclei.shape)
        print(scale)
        
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(nuclei, name='3D Nuc')
        # napari.run()
        # return

        segmentation = load_compressed_array(os.path.join(res_folder_path, 'segmentation.h5py'))
        pred_flows = load_compressed_array(os.path.join(res_folder_path, 'pred_flows.h5py'))
        print('segmentation.shape:', segmentation.shape)
        print('pred_flows.shape:', pred_flows.shape)

        nuclei=nuclei[426:535,129:614,163:457]
        segmentation=segmentation[426:535,129:614,163:457]
        pred_flows=pred_flows[:,426:535,129:614,163:457]


        #plot_result(nuclei,segmentation,pred_flows)
        
        # Transpose pred_flows to match expected shape
        vector_field = pred_flows.transpose(1, 2, 3, 0)
        print('vector_field.shape:', vector_field.shape)
        
        # Compute the labels
        labels, divergence_norm, curl_norm = compute_boundary_proxy(vector_field, sigma=1.0)
        print(divergence_norm.shape, curl_norm.shape)
        print('labels.shape:', labels.shape)
    
        import napari

        viewer = napari.Viewer()
        viewer.add_image(nuclei, name='3D Nuc')
        viewer.add_image(divergence_norm, name='divergence_norm')
        viewer.add_image(curl_norm, name='curl_norm')
        viewer.add_labels(labels, name='labels')
        z, y, x = np.nonzero(nuclei) 
        origins = np.stack((z, y, x), axis=-1)
        # Use the correctly shaped vector_field
        vectors = vector_field[z, y, x]
        vector_data = np.stack((origins, vectors), axis=1)
        viewer.add_vectors(vector_data, name='3D Flow Field pred', edge_width=0.1, length=1, ndim=3, edge_color='blue')
        napari.run()



if __name__ == "__main__":
    real_example()
    