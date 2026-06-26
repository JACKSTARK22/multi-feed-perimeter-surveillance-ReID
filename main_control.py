import os
from reid_live import build_gallery, run_live

# 1. SETUP PARAMETERS
# This class mimics the 'args' that the original script expects from the command line
class Config:
    def __init__(self):
        # Path to your trained model weights
        self.checkpoint = "osnet_x0_75_BEST.pth" 
        
        # Path to the folder containing subfolders of known people
        self.gallery_dir = "known_persons" 
        
        # Where to save/load the simplified feature database
        self.gallery = "gallery.pkl"
        self.output = "gallery.pkl"
        
        # Source: "0" for webcam, or "path/to/video.mp4"
        self.source = r"0"
        
        # Recognition sensitivity (higher = stricter)
        self.threshold = 0.60

def main():
    params = Config()

    # 2. VALIDATION
    if not os.path.exists(params.checkpoint):
        print(f"ERROR: Model file {params.checkpoint} not found!")
        return

    # 3. STEP 1: BUILD THE GALLERY
    # Only needs to be run once, or whenever you add new people to the folder
    print("\n--- [Phase 1: Encoding Known Persons] ---")
    try:
        build_gallery(params)
    except Exception as e:
        print(f"Gallery build failed: {e}")
        return

    # 4. STEP 2: RUN LIVE RE-ID
    print("\n--- [Phase 2: Starting Live Tracking] ---")
    print("Controls: Press 's' for screenshot, 'q' to quit.")
    run_live(params)

if __name__ == "__main__":
    main()