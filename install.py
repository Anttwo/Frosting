import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup the environment')
    
    parser.add_argument('--no_nvdiffrast', action='store_true', help='Skip installation of Nvdiffrast')
    args = parser.parse_args()
    
    # Create a new conda environment
    print("[INFO] Creating the conda environment for Frosting...")
    os.system("conda env create -f environment.yml")
    print("[INFO] Conda environment created.")
    
    # Install 3D Gaussian Splatting rasterizer
    print("[INFO] Installing the 3D Gaussian Splatting rasterizer...")
    os.chdir("gaussian_splatting/submodules/diff-gaussian-rasterization/")
    os.system("conda run -n frosting pip install -e .")
    print("[INFO] 3D Gaussian Splatting rasterizer installed.")
    
    # Install simple-knn
    print("[INFO] Installing simple-knn...")
    os.chdir("../simple-knn/")
    os.system("conda run -n frosting pip install -e .")
    print("[INFO] simple-knn installed.")
    os.chdir("../../../")
    
    # Install Nvdiffrast
    if args.no_nvdiffrast:
        print("[INFO] Skipping installation of Nvdiffrast.")
    else:
        print("[INFO] Installing Nvdiffrast...")
        os.system("git clone https://github.com/NVlabs/nvdiffrast")
        os.chdir("nvdiffrast")
        os.system("conda run -n frosting pip install .")
        print("[INFO] Nvdiffrast installed.")
        print("[INFO] Please note that Nvdiffrast will take a few seconds or minutes to build the first time it is used.")
        os.chdir("../")

    print("[INFO] Frosting installation complete.")
