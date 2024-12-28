import os
import shutil
import atexit
import tempfile
from typing import Literal
import streamlit.components.v1 as components


parent_dir = os.path.dirname(os.path.abspath(__file__))

class STLComponent:
    def __init__(self):
        """Initialize the STLComponent class and set up the environment."""
        self.has_setup = False
        self.temp_folder = None
        self.current_temp_files = []  # List to track created temporary files
        self.setup()  # Automatically call setup upon initialization

    def setup(self):
        """Set up the necessary directories for the Streamlit_stl component."""
        if not self.has_setup:

            ### Create a unique temporary directory for the component
            if self.temp_folder and os.path.exists(self.temp_folder):
                shutil.rmtree(self.temp_folder)
            self.temp_folder = tempfile.mkdtemp(suffix='_st_stl')
            
            ### Copy the current component directory to the temporary folder
            for file in os.listdir(parent_dir):
                src = parent_dir + os.sep + file
                dst = self.temp_folder + os.sep + file
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy(src, dst)

            ### Mark setup as complete to prevent re-initialization
            self.has_setup = True  

    def stl_from_text(self, 
                        text: str,
                        color: str = '#696969', 
                        material: Literal['material', 'flat', 'wireframe'] = 'material',
                        auto_rotate: bool = False,
                        opacity: int = 1,
                        shininess: int = 100,
                        cam_v_angle: int = 60,
                        cam_h_angle: int = -90,
                        cam_distance: int = 0,
                        height: int = 500,
                        max_view_distance: int  =1000, 
                        **kwargs):
        """
        Create a 3D STL viewer component in Streamlit using a text-based STL file.

        Parameters:
        ----------
        text : str
            The text content of the STL file to render.
        color : str, optional
            The hexadecimal color (starting with '#') for the 3D object. Default is '#696969'.
        material : Literal['material', 'flat', 'wireframe'], optional
            The material style of the 3D object. Options are:
            - 'material': Basic physical material.
            - 'flat': Flat shading.
            - 'wireframe': Wireframe view.
            Default is 'material'.
        auto_rotate : bool, optional
            Whether to enable auto-rotation of the 3D object. Default is False.
        opacity : int, optional
            Opacity of the 3D object, ranging from 0 (fully transparent) to 1 (fully opaque). Default is 1.
        shininess : int, optional
            How shiny the specular highlight is, when using the 'material' material style. Default is 100.
        cam_v_angle : int, optional
            Vertical angle (in degrees) for the camera view. Default is 60.
        cam_h_angle : int, optional
            Horizontal angle (in degrees) for the camera view. Default is -90.
        cam_distance : int, optional
            Distance of the camera from the object. If zero, defaults to three times the largest bounding box size. Default is zero.
        height : int, optional
            Height of the 3D viewer component in pixels. Default is 500.
        max_view_distance : int, optional
            Maximum viewing distance for the camera. Default is 1000.
        **kwargs :
            Additional arguments passed to the Streamlit component.

        Returns:
        -------
        bool
            True if the component is successfully created, False otherwise.
        """
        self.setup()  # Ensure the environment is set up
        file_path = []  # The path of the created temporary file
        if material not in ('material', 'flat', 'wireframe'):
            raise ValueError(f'The possible materials are "material", "flat" or "wireframe", got {material} instead')
        if color[0] != '#':
            raise ValueError(f"The color must be a hexadecimal value starting with '#', got {color} instead")
        if text is not None:

            ### Create a temporary file in the temporary stl folder
            try:
                with tempfile.NamedTemporaryFile(dir=self.temp_folder, suffix='.stl', delete=False) as temp_file:
                    if isinstance(text, bytes):
                        temp_file.write(text)
                    elif isinstance(text, str):
                        # Write the text content to the file
                        temp_file.write(text.encode("utf-8"))  
                    else:
                        raise ValueError(f"Invalid text type for the stl file")
                    # Ensure all data is written to disk
                    temp_file.flush()  
                    # Store the relative path
                    file_path = temp_file.name.split(os.sep)[-1]  
                    # Keep track of the file for cleanup
                    self.current_temp_files.append(temp_file.name)  

            except Exception as e:
                print(f"Error processing the stl file: {e}")
                _component_func(files_text='', height=height **kwargs)
                return False

        ### Call the stl component with the list of file paths and their types
        _component_func(file_path=file_path, 
                        color=color, 
                        material=material, 
                        auto_rotate=bool(auto_rotate), 
                        opacity=opacity, 
                        shininess=shininess,
                        cam_v_angle=cam_v_angle,
                        cam_h_angle=cam_h_angle,
                        cam_distance=cam_distance,
                        height=height, 
                        max_view_distance=max_view_distance,
                        **kwargs)
        return True

    def stl_from_file(self, 
                      file_path: str, 
                      color: str = '#696969',
                      material: Literal['material', 'flat', 'wireframe'] = 'material',
                      auto_rotate: bool = False,
                      opacity: int = 1, 
                      shininess: int = 100,
                      cam_v_angle: int = 60,
                      cam_h_angle: int = -90,
                      cam_distance: int = 0,
                      height: int = 500,
                      max_view_distance: int = 1000, 
                      **kwargs):
        """
        Render a 3D STL file in Streamlit using a file path.

        Parameters:
        ----------
        file_path : str
            The path to the STL file to render.
        color : str, optional
            The hexadecimal color (starting with '#') for the 3D object. Default is '#696969'.
        material : Literal['material', 'flat', 'wireframe'], optional
            The material style of the 3D object. Options are:
            - 'material': Basic physical material.
            - 'flat': Flat shading.
            - 'wireframe': Wireframe view.
            Default is 'material'.
        auto_rotate : bool, optional
            Whether to enable auto-rotation of the 3D object. Default is False.
        opacity : int, optional
            Opacity of the 3D object, ranging from 0 (fully transparent) to 1 (fully opaque). Default is 1.
        shininess : int, optional
            How shiny the specular highlight is, when using the 'material' material style. Default is 100.
        cam_v_angle : int, optional
            Vertical angle (in degrees) for the camera view. Default is 60.
        cam_h_angle : int, optional
            Horizontal angle (in degrees) for the camera view. Default is -90.
        cam_distance : int, optional
            Distance of the camera from the object. If zero, defaults to three times the largest bounding box size. Default is zero.
        height : int, optional
            Height of the 3D viewer component in pixels. Default is 500.
        max_view_distance : int, optional
            Maximum viewing distance for the camera. Default is 1000.
        **kwargs :
            Additional arguments passed to the Streamlit component.

        Returns:
        -------
        bool
            True if the component is successfully created, False otherwise.
        """

        file_text = None

        ### Read the file content and add it to the list
        if file_path is not None:
            with open(file_path, "rb") as f:
                file_text = f.read()  
        
        ### Pass the file content to stl_from_text
        return self.stl_from_text(text=file_text, 
                                  color=color, 
                                  material=material, 
                                  auto_rotate=auto_rotate, 
                                  opacity=opacity,
                                  shininess=shininess,
                                  height=height, 
                                  cam_v_angle=cam_v_angle,
                                  cam_h_angle=cam_h_angle,
                                  cam_distance=cam_distance,
                                  max_view_distance=max_view_distance,
                                  **kwargs)

    def cleanup_temp_files(self):
        """Clean up temporary files and directories created during the session."""
        ### Remove the entire temporary directory
        try:
            if os.path.exists(self.temp_folder):
                shutil.rmtree(self.temp_folder)  

        except Exception as e:
            print(f"Error deleting temporary streamlit-stl folder {self.temp_folder}: {e}")
            # If the directory can't be deleted, try to delete each file individually
            for temp_file in self.current_temp_files:
                try: # Remove individual temporary files
                    os.unlink(temp_file)  
                except Exception as e:
                    print(f"Error deleting temp file {temp_file}: {e}")

# Instantiate the STLComponent class to set up the environment and handle resources
stl_component = STLComponent()
# Register the cleanup function to be called automatically when the program exits
atexit.register(stl_component.cleanup_temp_files)


### Declare the functions to be used in the Streamlit script
stl_from_text = stl_component.stl_from_text
stl_from_file = stl_component.stl_from_file

# Declare the Streamlit component and link it to the temporary directory
_component_func = components.declare_component(
    "streamlit_stl",
    path=stl_component.temp_folder,
)