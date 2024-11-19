import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class WeatherBench128(Dataset):
    def __init__(self, 
                 start_time: str='2000-01-01 00:00:00', 
                 end_time: str='2000-01-05 23:00:00',
                 include_target: bool=False,
                 lead_time: int=6, 
                 interval: int=6,
                 muti_target_steps: int=1):
        
        self.variables_list = [
        0, 1, 2, 4 ,
        6 ,7 ,8 ,9 ,10,11,12,13,14,15,16,17,18,
        19,20,21,22,23,24,25,26,27,28,29,30,31,
        45,46,47,48,49,50,51,52,53,54,55,56,57,
        58,59,60,61,62,63,64,65,66,67,68,69,70,
        71,72,73,74,75,76,77,78,79,80,81,82,83]
        self.data_folder = "your_weatherbench_data_path"
        self.start_time = start_time
        self.end_time = end_time
        self.include_target = include_target
        self.lead_time = lead_time
        self.interval = interval
        self.muti_target_steps = muti_target_steps
        self.init_time_list()
        self.init_file_list()
        self.get_mean_std()
        self.length = len(self.x_time_ilst)


    def init_time_list(self):
        if self.include_target:
            target_end_time = pd.to_datetime(self.end_time)
            input_end_time = target_end_time - pd.Timedelta(hours=self.muti_target_steps*self.lead_time)
            input_end_time_str = input_end_time.strftime('%Y-%m-%d %H:%M:%S')
            self.x_time_ilst = pd.date_range(self.start_time, input_end_time_str, freq=str(self.interval)+'h')
        else:
            self.x_time_ilst = pd.date_range(self.start_time, self.end_time, freq=str(self.interval)+'h')


    def idx_in_year(self, time_stamp):
        year = time_stamp.year
        first_day = pd.to_datetime(f'{year}-01-01 00:00:00')
        idx = int((time_stamp - first_day).total_seconds() / 3600)
        return idx
    

    def init_file_list(self):
        # your_weatherbench_data_path/1979/1979-0000.npy
        self.x_file_list= [os.path.join(self.data_folder, 
                                        str(time_stamp.year),
                                        str(time_stamp.year)+'-{:04d}'.format(self.idx_in_year(time_stamp))+'.npy')
                                        for time_stamp in self.x_time_ilst]
        

    def get_mean_std(self):
        mean_std = np.load("your_weatherbench_data_path/mean_std.npy")
        # mean_std = np.ones([2, 110]) # Test
        self.the_mean = mean_std[0]
        self.the_std = mean_std[1]
        self.data_mean_tensor = torch.from_numpy(self.the_mean[self.variables_list]).float()
        self.data_std_tensor = torch.from_numpy(self.the_std[self.variables_list]).float()

   
    def normalization(self, sample):
        return (sample[self.variables_list] - self.the_mean[self.variables_list, None, None]) / self.the_std[self.variables_list, None, None]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file_path = self.x_file_list[index]
        sample_x = np.load(file_path)
        # sample_x = np.zeros([110, 128, 256]) # Test
        sample_x = self.normalization(sample_x)
        sample_x = torch.from_numpy(sample_x).float()
        
        y_list = []
        x_time = self.x_time_ilst[index]
        for steps in range(self.muti_target_steps):
            y_time = x_time + pd.Timedelta(hours=(steps+1)*self.lead_time)
            y_file_path = os.path.join(self.data_folder, 
                                       str(y_time.year), 
                                       str(y_time.year)+'-{:04d}'.format(self.idx_in_year(y_time))+'.npy')
            sample_y = np.load(y_file_path)
            # sample_y = np.zeros([110, 128, 256]) # Test
            sample_y = self.normalization(sample_y)
            sample_y = torch.from_numpy(sample_y).float()
            y_list.append(sample_y)

        if self.muti_target_steps > 1:
            sample_y_all = torch.stack(y_list, dim=0)
        else:
            sample_y_all = y_list[0]

        return sample_x, sample_y_all


if __name__ == '__main__':
    train_data = WeatherBench128(start_time='2000-01-01 00:00:00', 
                                end_time='2000-01-10 23:00:00',
                                include_target=False,
                                lead_time=6, 
                                interval=6,
                                muti_target_steps=4)
    # 240h 40input 10batch
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=False, num_workers=8)
    print("batch len:", len(train_loader))

    for batch_idx, (X, Y) in enumerate(train_loader):
        print("batch", batch_idx)
        now_mean = []
        now_std = []
        print("X")
        print(X.shape)
        for i in range(X.shape[1]):
            now_mean.append(round(float(torch.mean(X[0][i])), 2))
            now_std.append(round(float(torch.std(X[0][i])),2))
        print(now_mean)
        print(now_std)
        
        now_mean = []
        now_std = []
        print("Y")
        print(Y.shape)
        for i in range(Y.shape[2]):
            now_mean.append(round(float(torch.mean(Y[0][0][i])),2))
            now_std.append(round(float(torch.std(Y[0][0][i])),2))
        print(now_mean)
        print(now_std)
        
        
        if batch_idx == 1:
            break
