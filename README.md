# Streamlit STL Display Component

A Streamlit component to display STL files.

## Installation

**This component requires access to write files to the temporary directory.**

```
pip install streamlit_stl
```

## Example

![Alt Text](https://github.com/Lucandia/streamlit_stl/blob/main/example.png?raw=true)

Look at the [example](https://st-stl.streamlit.app/) for a streamlit Web App:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://st-stl.streamlit.app/)

The original STL file is from [Printables](https://www.printables.com/it/model/505713-flexifier-flexi-3d-models-generator-print-in-place).

## Usage

### Display from file paths

```python
import streamlit as st
from streamlit_stl import stl_from_file

success = stl_from_file(
    file_path=path_to_conf,          # Path to the STL file
    color='#FF9900',                 # Color of the STL file (hexadecimal value)
    material='material',             # Material of the STL file ('material', 'flat', or 'wireframe')
    auto_rotate=True,                # Enable auto-rotation of the STL model
    opacity=1,                       # Opacity of the STL model (0 to 1)
    shininess=100,                   # How shiny the specular highlight is, when using the 'material' style.
    cam_v_angle=60,                  # Vertical angle (in degrees) of the camera
    cam_h_angle=-90,                 # Horizontal angle (in degrees) of the camera
    cam_distance=None,               # Distance of the camera from the object (defaults to 3x bounding box size)
    height=500,                      # Height of the viewer frame
    max_view_distance=1000,          # Maximum viewing distance for the camera
    key=None                         # Streamlit component key
)
```

### Display from file text

```python
import streamlit as st
from streamlit_stl import stl_from_text

file_input = st.file_uploader("Or upload an STL file", type=["stl"])

if file_input is not None:
    success = stl_from_text(
        text=file_input.getvalue(),  # Content of the STL file as text
        color='#FF9900',             # Color of the STL file (hexadecimal value)
        material='material',         # Material of the STL file ('material', 'flat', or 'wireframe')
        auto_rotate=True,            # Enable auto-rotation of the STL model
        opacity=1,                   # Opacity of the STL model (0 to 1)
        shininess=100,               # How shiny the specular highlight is, when using the 'material' style.
        cam_v_angle=60,              # Vertical angle (in degrees) of the camera
        cam_h_angle=-90,             # Horizontal angle (in degrees) of the camera
        cam_distance=None,           # Distance of the camera from the object (defaults to 3x bounding box size)
        height=500,                  # Height of the viewer frame
        max_view_distance=1000,      # Maximum viewing distance for the camera
        key=None                     # Streamlit component key
    )
```

The functions return a boolean value indicating if the program was able to write and read the files.

The 'material' style is the default style, it uses the [Phong shading](https://threejs.org/docs/api/en/materials/MeshPhongMaterial.html) model from Three.js.

## License

Code is licensed under the GNU General Public License v3.0 ([GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html))

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%20v3-lightgrey.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)
