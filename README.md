# Pixels to Pyrometrics
This repo is a collection of code (python and R) to convert UAS-based infrared imagery of fire spread into quantitative measurements of fire behavior, as seen in [O'Neill et al., 2024](https://doi.org/10.1071/WF24067).

## thermal_image_preprocessing
This section processes, stabilizes, georeferences, and calibrates thermal images collected from Autel Evo radiometric thermal imagery. The scripts handle various preprocessing steps such as sorting, stabilization, radiative power calculation, and georeferencing using Ground Control Points (GCPs).

## Project Structure
```
📂 Project Directory  
 ├── 📂 Input Images/  *(Folder containing raw images)*  
 ├── 📂 Output Images/  *(Processed images are saved here)*  
 ├── 📜 1_preprocess_images.py  *(Sorts and organizes RGB and thermal images, extracts EXIF metadata)*  
 ├── 📜 2_stabilize_images_main.py  *(Stabilizes images using image alignment techniques)*  
 ├── 📜 3_calculate_radiative_power.py  *(Computes radiative power using Stefan-Boltzmann law)*  
 ├── 📜 4_georef_images.py  *(Georeferences images using a reference image and GCPs)*  
 ├── 📜 5_calibrate_autel_evo_thermal_images.py  *(Calibrates thermal images from Autel EVO drone)*  
 └── 📜 requirements.txt  *(List of required Python dependencies)*  
```

### Setup Instructions

#### 1. **Create a Virtual Environment (Optional)**
To keep dependencies isolated, create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

#### 2. **Install Dependencies**
Install all required packages:

```bash
conda create --name thermal_processing
conda activate thermal_processing
```

# R Data Processing and Analysis

This section contains R Markdown files used for processing pyrometrics data, calibrating burn datasets, and generating manuscript figures.

#### **1. Pyrometrics Data Processing (`1_pyrometrics_processing.Rmd`)**
- Processes and analyzes pyrometric data for fire behavior.
- Cleans and structures datasets for further statistical analysis.
- Generates summary statistics and visualizations.

#### **2. Calibration Burn Processing (`2_calibration_burn_processing.Rmd`)**
- Processes calibration burn datasets for validation and accuracy assessment.
- Applies statistical methods to correct measurements.
- Outputs structured datasets for modeling.

#### **3. Manuscript Figures (`3_manuscript_figures.Rmd`)**
- Script that was used to generate figures for manuscript
- Uses processed fire severity and calibration data for visualization.
- Generates reproducible plots for scientific reporting.

## Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request.

## Citation

If you use this code or data in your research, please cite the following publication:

O’Neill, L., Fulé, P. Z., Watts, A., Moran, C., Hopkins, B., Rowell, E., Thode, A., & Afghah, F. (2024).  
**Pixels to pyrometrics: UAS-derived infrared imagery to evaluate and monitor prescribed fire behaviour and effects.**  
*International Journal of Wildland Fire, 33*(11).  
[https://doi.org/10.1071/WF24067](https://doi.org/10.1071/WF24067)


## License
[MIT License](LICENSE)
