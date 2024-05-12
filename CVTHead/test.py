import subprocess
import glob


subprocess.run(["python3", "face_detect/detect.py"]) 

jpgFilenamesList = glob.glob('results/detection_*.jpg')
# print(jpgFilenamesList)

for i, jpgFileName in enumerate(jpgFilenamesList):
    print(jpgFileName)
    subprocess.run(["python3", "inference.py", 
                        "--src_pth", f"{jpgFileName}",
                        # "--drv_pth", "examples/2.png",
                        # "--drv_pth", f"{jpgFileName}",
                        "--drv_pth", f"{jpgFilenamesList[(i+1) % 2]}",
                        "--out_pth", f"results/output_{i}.png",
                        "--ckpt_pth", "data/cvthead.pt"]
                    )
    # subprocess.run(["python3", "inference.py",
    #                     "--src_pth", f"{jpgFileName}",
    #                     "--out_pth", "results/",
    #                     "--ckpt_pth", "data/cvthead.pt",
    #                     "--flame"
    #                 ])
