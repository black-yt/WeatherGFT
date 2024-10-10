Download **`/example_data`** from [Google Drive](https://drive.google.com/drive/folders/1fZlr0LS3aFJAym79ojn3njYiUhtdzKF6?usp=sharing).

The size of the data is [4, 69, 128, 256], corresponding to [B, C, H, W].

The 69 channels are:

| Variable name | Description                    | Pressure layer |
| ------------- | -----------                    | -------------- |
| t2m           | temperature at 2m height       | single         |
| u10           | x-direction wind at 10m height | single         |
| v10           | y-direction wind at 10m height | single         |
| tp            | hourly precipitation           | single         |
| z             | geopotential                   | 13             |
| t             | temperature                    | 13             |
| r             | relative humidity              | 13             |
| u             | x-direction wind               | 13             |
| v             | y-direction wind               | 13             |

13 pressure layers are: `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` hPa.

**`mean_std.json`** is the mean and variance of the variables corresponding to 69 channels, which can be used for normalization and denormalization.

**`input.npy`** and **`target.npy`** downloaded from Google Drive are normalized data.
