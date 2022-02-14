import os
import pandas as pd
images_list = sorted(os.listdir("images"))
data = []
for img in images_list:
    data.append([img.split("_")[0], img.split("_")[1]])
df = pd.DataFrame(data, columns=["file_path", "class"])
df.to_csv("df.csv", index=False)