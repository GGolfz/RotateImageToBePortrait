import os
import pandas as pd
images_list = sorted(os.listdir("output"))
data = []
for img in images_list:
    data.append([img, img.split("_")[1]])
df = pd.DataFrame(data, columns=["file_path", "class"])
df.to_csv("df.csv", index=False)