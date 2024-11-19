import torch
import numpy as np

SEDI_t2m_128 = torch.from_numpy(np.load("/mnt/petrelfs/xuwanghan/projects/high_resolution/sedi/Q_SEDI_128/SEDI_t2m.npy"))
SEDI_ws10_128 = torch.from_numpy(np.load("/mnt/petrelfs/xuwanghan/projects/high_resolution/sedi/Q_SEDI_128/SEDI_ws10.npy"))
SEDI_t2m_721 = torch.from_numpy(np.load("/mnt/petrelfs/xuwanghan/projects/high_resolution/sedi/Q_SEDI_721/SEDI_t2m.npy"))
SEDI_ws10_721 = torch.from_numpy(np.load("/mnt/petrelfs/xuwanghan/projects/high_resolution/sedi/Q_SEDI_721/SEDI_ws10.npy"))

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

def weighted_latitude_weighting_factor_torch(j: torch.Tensor, real_num_lat:int, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return real_num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

# @torch.jit.script
def type_weighted_bias_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)


    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))

    result = torch.mean(weight * pred, dim=(-1, -2))

    # result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
    return result

# @torch.jit.script
def type_weighted_bias_torch(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_bias_torch_channels(pred, metric_type=metric_type)
    return torch.mean(result, dim=0)

# @torch.jit.script
def type_weighted_activity_torch_channels(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(weighted_latitude_weighting_factor_torch(lat_t, num_lat, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - torch.mean(weight * pred, dim=(-1, -2), keepdim=True)) ** 2, dim=(-1, -2)))
    return result

def type_weighted_activity_torch(pred: torch.Tensor, metric_type="all") -> torch.Tensor:
    result = type_weighted_activity_torch_channels(pred, metric_type=metric_type)
    return torch.mean(result, dim=0)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat)) / s

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def top_quantiles_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    qs = 50
    qlim = 4
    qcut = 1
    n, c, h, w = pred.size()
    qtile = 1. - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=target.dtype)
    P_tar = torch.quantile(target.view(n,c,h*w), q=qtile, dim=-1)
    qtile = 1. - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device, dtype=pred.dtype)
    P_pred = torch.quantile(pred.view(n,c,h*w), q=qtile, dim=-1)
    return torch.mean(torch.mean((P_pred - P_tar)/P_tar, dim=0), dim=0)
    # return torch.mean((P_pred - P_tar)/P_tar, dim=0).view(c)


class Metrics():
    def __init__(self, dtat_mean=None, data_std=None):
        self.data_mean = dtat_mean
        self.data_std = data_std
        
    def MSE(self, pred, gt):
        sample_mse = torch.mean((pred - gt) ** 2)
        return sample_mse.item()

    def Bias(self, pred, gt):
        data_std = self.data_std.to(gt.device)
        return (type_weighted_bias_torch(pred - gt, metric_type="all") * data_std).tolist()

    def Activity(self, pred, clim_time_mean_daily):
        clim_time_mean_daily = clim_time_mean_daily.to(pred.device)
        data_std = self.data_std.to(pred.device)
        return (type_weighted_activity_torch(pred - clim_time_mean_daily, metric_type="all") * data_std).tolist()

    def WRMSE(self, pred, gt):
        data_std = self.data_std.to(gt.device)
        return (weighted_rmse_torch(pred, gt) * data_std).tolist()

    def WACC(self, pred, gt, clim_time_mean_daily):
        clim_time_mean_daily = clim_time_mean_daily.to(gt.device)
        return (weighted_acc_torch(pred - clim_time_mean_daily, gt - clim_time_mean_daily)).tolist()


    def RQE(self, pred, gt):
        data_mean = self.data_mean.to(gt.device)
        data_std = self.data_std.to(gt.device)
        pred_real = pred * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(1, gt.shape[1], 1, 1)
        gt_real = gt * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(1, gt.shape[1], 1, 1)
        return (top_quantiles_error_torch(pred_real[:,[37,24,0,11,2,66],:,:], gt_real[:,[37,24,0,11,2,66],:,:])).tolist()
        
    def SEDI(self, pred, gt, month):
        t2m_c = 2
        u_c = 0
        v_c = 1

        data_mean = self.data_mean.to(gt.device)
        data_std = self.data_std.to(gt.device)
        pred_real = pred * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(1, gt.shape[1], 1, 1)
        gt_real = gt * data_std.view(1, gt.shape[1], 1, 1) + data_mean.view(1, gt.shape[1], 1, 1)

        if gt.shape[-2] == 721:
            SEDI_t2m_device = SEDI_t2m_721[month-1].to(gt.device)
            SEDI_ws10_device = SEDI_ws10_721[month-1].to(gt.device)
        else:
            SEDI_t2m_device = SEDI_t2m_128[month-1].to(gt.device)
            SEDI_ws10_device = SEDI_ws10_128[month-1].to(gt.device)
        
        sedi_t2m_ws10 = torch.zeros([8]).to(gt.device)

        for i in range(4):
            SEDI_t2m_th = SEDI_t2m_device[i]
            SEDI_ws10_th = SEDI_ws10_device[i]

            gt_t2m_ex = (gt_real[:, t2m_c] > SEDI_t2m_th).float()
            gt_ws10 = (gt_real[:, u_c]**2 + gt_real[:, v_c]**2)**0.5
            gt_ws10_ex = (gt_ws10 > SEDI_ws10_th).float()

            pred_t2m_ex = (pred_real[:, t2m_c] > SEDI_t2m_th).float()
            pred_ws10 = (pred_real[:, u_c]**2 + pred_real[:, v_c]**2)**0.5
            pred_ws10_ex = (pred_ws10 > SEDI_ws10_th).float()

            FP_t2m = (pred_t2m_ex-gt_t2m_ex == 1).float().sum() # pred = 1; tar = 0
            TN_t2m = (pred_t2m_ex+gt_t2m_ex == 0).float().sum() # pred = 0; tar = 0
            TP_t2m = (pred_t2m_ex+gt_t2m_ex == 2).float().sum() # pred = 1; tar = 1
            FN_t2m = (gt_t2m_ex-pred_t2m_ex == 1).float().sum() # pred = 0; tar = 1

            if FP_t2m == 0:
                FP_t2m += 1
            if TN_t2m == 0:
                TN_t2m += 1
            if TP_t2m == 0:
                TP_t2m += 1
            if FN_t2m == 0:
                FN_t2m += 1

            F_t2m = FP_t2m/(FP_t2m+TN_t2m)
            H_t2m = TP_t2m/(TP_t2m+FN_t2m)

            SEDI_t2m = (torch.log(F_t2m)-torch.log(H_t2m)-torch.log(1-F_t2m)+torch.log(1-H_t2m))/ \
                        (torch.log(F_t2m)+torch.log(H_t2m)+torch.log(1-F_t2m)+torch.log(1-H_t2m))
            
            FP_ws10 = (pred_ws10_ex-gt_ws10_ex == 1).float().sum() # pred = 1; tar = 0
            TN_ws10 = (pred_ws10_ex+gt_ws10_ex == 0).float().sum() # pred = 0; tar = 0
            TP_ws10 = (pred_ws10_ex+gt_ws10_ex == 2).float().sum() # pred = 1; tar = 1
            FN_ws10 = (gt_ws10_ex-pred_ws10_ex == 1).float().sum() # pred = 0; tar = 1

            if FP_ws10 == 0:
                FP_ws10 += 1
            if TN_ws10 == 0:
                TN_ws10 += 1
            if TP_ws10 == 0:
                TP_ws10 += 1
            if FN_ws10 == 0:
                FN_ws10 += 1

            F_ws10 = FP_ws10/(FP_ws10+TN_ws10)
            H_ws10 = TP_ws10/(TP_ws10+FN_ws10)

            SEDI_ws10 = (torch.log(F_ws10)-torch.log(H_ws10)-torch.log(1-F_ws10)+torch.log(1-H_ws10))/ \
                        (torch.log(F_ws10)+torch.log(H_ws10)+torch.log(1-F_ws10)+torch.log(1-H_ws10))

            sedi_t2m_ws10[i] = SEDI_t2m
            sedi_t2m_ws10[4+i] = SEDI_ws10

        return sedi_t2m_ws10.tolist()

if __name__ == "__main__":
    pred = torch.randn([2, 69, 128, 256])
    gt = torch.randn([2, 69, 128, 256])
    data_mean = torch.randn([69])
    data_std = torch.randn([69])
    climate = torch.randn([2, 69, 128, 256])
    metrics = Metrics(data_mean, data_std)
    print(metrics.MSE(pred, gt))
    print(metrics.Bias(pred, gt))
    print(metrics.Activity(pred, climate))
    print(metrics.WRMSE(pred, gt))
    print(metrics.WACC(pred, gt, climate))
    print(metrics.RQE(pred, gt))
    print(metrics.SEDI(pred, gt, 12))