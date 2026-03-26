"""1D 波函数演化：光镊加热模拟（Torch GPU 加速）。

势能由实验给定的 I(x,t) 直接构造：V(x,t)=scale*I(x,t)+offset。
使用 Crank–Nicolson 求解 1D 时间依赖薛定谔方程，输出总能量增量对应的
“有效温度”变化。
"""
import math
import os
import numpy as np
from scipy.constants import hbar, k as k_B
import torch
from tqdm import tqdm

from src.PhysicalSimu import BasePhysicalSimulation, DataGroup

class TweezerHeatingSimulation(BasePhysicalSimulation):
    """光镊加热模拟（基于 AtomArrange 模块）。"""

    def __init__(self):
        super().__init__()
        self.name = "原子重排"
        self.description = "光镊加热模拟：输出有效温度随时间变化以及波函数密度热图"
        self.define_parameters()
        self.datafile=None
        self.nx=0
        self.nt=0
        self.inter_func='bilinear'
        self.I_data=None
        self.I_refresh=False
        self.mass=1.44316060e-25

    def define_parameters(self):
        self.params = {
            #'load_I':{'value':self._on_loadI_click,'desc':'加载光强','unit':'','type':'funcbtn'},
            'I_data':{'value':'trace_noise1','desc':'光强数据','unit':'','type':'text'},
            'pixel_size': {'value': 2.3e-7, 'min': 1e-9, 'max': 1e-6, 'step': 1e-9, 'desc': '像素大小', 'unit': 'm', 'type': 'scientific'},
            'nx': {'value': 301, 'min': 10, 'max': 5000, 'step': 1, 'desc': 'x 点数', 'unit': '', 'type': 'scientific'},
            'rec_freq': {'value': 500.0, 'min': 1.0, 'max': 10000.0, 'step': 1.0, 'desc': '记录频率', 'unit': 'Hz', 'type': 'scientific'},
            'nt': {'value': 4000, 'min': 10, 'max': 1e10, 'step': 1, 'desc': 't 点数', 'unit': '', 'type': 'scientific'},
            't_range':{'value': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.1, 'desc': '模拟范围', 'unit': '', 'type': 'scientific'},
            'x0_start': {'value': 1.7e-6, 'min': -1e-3, 'max': 1e-3, 'step': 1e-7, 'desc': '起始位置', 'unit': 'm', 'type': 'scientific'},
            'expos': {'value': 1.0, 'min': 0.0, 'max': 1e4, 'step': 0.1, 'desc': '曝光时间', 'unit': 'ms', 'type': 'scientific'},
            'inter_func': {'value': 'bicubic', 'desc': '插值方法', 'unit': '', 'type': 'combo', 'options': ['nearest', 'bilinear', 'bicubic']},
            'potential_scale': {'value': -1e-29, 'min': -1e20, 'max': 1e20, 'step': 1e-30, 'desc': '势能尺度', 'unit': '', 'type': 'scientific'},
            'potential_offset': {'value': 0.0, 'min': -1e-20, 'max': 1e-20, 'step': 1e-30, 'desc': '势能偏移', 'unit': '', 'type': 'scientific'},
            'harm_n': {'value': 8, 'min': 1e-20, 'max': 1e20, 'step': 1e-7, 'desc': '谐振子能级', 'unit': 'm', 'type': 'scientific'},
            'sigma0': {'value': 2e-7, 'min': 1e-20, 'max': 1e20, 'step': 1e-7, 'desc': '初始宽度', 'unit': 'm', 'type': 'scientific'},
            #'mass': {'value': 1.44316060e-25, 'min': 1e-30, 'max': 1e-20, 'step': 1e-30, 'desc': '粒子质量', 'unit': 'kg', 'type': 'scientific'},
            'absorb_strength': {'value': 1e-30, 'min': 0.0, 'max': 10.0, 'step': 0.1, 'desc': '吸收强度', 'unit': '', 'type': 'scientific'},
            'device': {'value': 'cpu', 'desc': '计算设备', 'unit': '', 'type': 'combo', 'options': ['cpu', 'cuda']},
        }

    def refresh_param(self):
        params = self.get_parameters()
        self.pixel_size = params.get('pixel_size', 1e-8)
        self.rec_freq = params.get('rec_freq', 500.0)
        self.expos = params.get('expos', 1.0)
        self.x0_start = params.get('x0_start', -2e-6)
        self.potential_scale = params.get('potential_scale', -1e-28)
        self.potential_offset = params.get('potential_offset', 0.0)
        self.sigma0 = params.get('sigma0', 1e-6)
        self.harm_n=int(params.get('harm_n', 5))
        #self.mass = params.get('mass', 1.44316060e-25)
        self.absorb_strength = params.get('absorb_strength', 1.0)
        self.device = params.get('device', 'cpu')

        data_file = params.get('I_data')
        nx = int(params.get('nx', 501))
        nt = int(params.get('nt', 801))
        t_range = params.get('t_range', 0.5)
        inter_func=params.get('inter_func','bilinear')
        if data_file!=self.datafile or nx!=self.nx or nt!=self.nt or inter_func!=self.inter_func or t_range!=self.t_range:
            self.datafile=data_file
            self.nx=nx
            self.nt=nt
            self.inter_func=inter_func
            self.t_range=t_range
            self.I_refresh=True
            self.I_data=self._load_I_data(data_file)

    def get_derived_parameters(self):
        return {
            #'T_final_K': {'value': getattr(self, 'T_final_K', 0.0), 'desc': '最终温度', 'unit': 'K'},
        }

    def _load_I_data(self, filename):
        file_path = os.path.join('load', f'{filename}.npy')
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'光强数据文件不存在: {file_path}')

        raw = np.load(file_path)
        if raw.ndim != 2:
            raise ValueError('光强数据必须为二维数组，shape=(nt,nx)')

        I_tensor = torch.as_tensor(raw[:int(self.t_range * raw.shape[0]), :], dtype=torch.float64)
        self.nt_raw, self.nx_raw = I_tensor.shape
        target_nt = int(self.nt)
        target_nx = int(self.nx)

        if self.nt_raw >= target_nt and self.nx_raw >= target_nx:
            I_res = I_tensor
            
        else:
            I4 = I_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,nt_raw,nx_raw)
            I4_res = torch.nn.functional.interpolate(
                I4,size=(target_nt, target_nx),mode=self.inter_func,align_corners=True,)
            I_res = I4_res[0, 0]

        return I_res

    def _harmonic_eigenstate(self, n, x, x0, m, V0, w0):
        """谐振子本征态生成函数"""
        omega = torch.sqrt(torch.abs(8 * V0 / (m * w0**2)))
        a = torch.sqrt(hbar / (m * omega))
        xi = (x - x0) / a
        
        # 递归计算厄米多项式
        H_prev2 = torch.ones_like(xi)  # H_0
        H_prev1 = 2 * xi               # H_1
        
        if n == 0:
            H = H_prev2
        elif n == 1:
            H = H_prev1
        else:
            for k in range(2, n+1):
                H = 2 * xi * H_prev1 - 2 * (k-1) * H_prev2
                H_prev2, H_prev1 = H_prev1, H
        
        # 波函数
        N = 1 / torch.sqrt(2**n * torch.tensor(math.factorial(n)) * torch.pi) / torch.sqrt(a)
        psi = N * H * torch.exp(-xi**2 / 2)
        
        return psi

    def _cn_step_torch(self, psi: torch.Tensor, V: torch.Tensor, dt: float, dx: float, mass: float) -> torch.Tensor:
        """Crank–Nicolson 步进（带吸收边界）"""
        coeff = hbar**2 / (2.0 * mass * dx**2)
        alpha = 1j * dt / (2.0 * hbar)

        diag = 2.0 * coeff + V
        A_diag = 1.0 + alpha * diag
        B_diag = 1.0 - alpha * diag
        A_off = -alpha * coeff
        B_off = +alpha * coeff

        rhs = B_diag[1:-1] * psi[1:-1] + B_off * (psi[2:] + psi[:-2])
        n = rhs.shape[0]

        dtype = psi.dtype
        A = torch.zeros((n, n), dtype=dtype, device=psi.device)
        A.diagonal().copy_(A_diag[1:-1])
        off_diag = torch.full((n - 1,), A_off, dtype=dtype, device=psi.device)
        A += torch.diag(off_diag, diagonal=1)
        A += torch.diag(off_diag, diagonal=-1)

        psi_interior = torch.linalg.solve(A, rhs)

        psi_next = torch.zeros_like(psi)
        psi_next[1:-1] = psi_interior
        return psi_next

    def _simu_scattering(self, psi, x_t, V_now_real, dt, dx, delta_temp_per_step):
        # 傅里叶变换到动量空间
        psi_k = torch.fft.fft(psi, norm='ortho')
        k_grid = 2 * torch.pi * torch.fft.fftfreq(len(x_t), d=dx)
        
        # 加热效应：动量分布的高斯展宽
        # 对应温度增加 ΔT 导致动量方差增加 Δσ_p² = m k_B ΔT
        sigma_p_increase = torch.sqrt(self.mass * k_B * delta_temp_per_step)
        
        # 当前动量分布的高斯滤波
        # 注意：这是近似，实际加热是扩散过程
        current_sigma = torch.std(k_grid * torch.abs(psi_k))
        new_sigma = torch.sqrt(current_sigma**2 + sigma_p_increase**2)
        
        # 重新生成展宽的动量分布
        gaussian_filter = torch.exp(-0.5 * (k_grid / new_sigma)**2)
        psi_k_filtered = psi_k * gaussian_filter
        
        # 变换回坐标空间
        psi = torch.fft.ifft(psi_k_filtered, norm='ortho')
        
        # 重新归一化
        psi = psi / torch.sqrt(torch.sum(torch.abs(psi)**2) * dx)
        return psi

    def simulate_tweezer_heating(
        self,x,t,V_xt,a,*,mass: float = 1.44316060e-25) -> dict:
        """模拟 1D 光镊加热（带吸收边界）"""
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        x_t = torch.as_tensor(x, dtype=torch.float64, device=device)
        t_t = torch.as_tensor(t, dtype=torch.float64, device=device)

        V_xt = torch.as_tensor(V_xt, dtype=torch.float64, device=device)
        if V_xt.shape == (len(x_t), len(t_t)):
            V_xt = V_xt.T
        if V_xt.shape != (len(t_t), len(x_t)):
            raise ValueError("V_xt must have shape (len(t), len(x)) or (len(x), len(t)).")

        dx = float(x_t[1] - x_t[0])
        dx_tensor = torch.tensor(dx, dtype=x_t.dtype, device=x_t.device)
        if not torch.allclose(torch.diff(x_t), dx_tensor, rtol=1e-9, atol=0.0):
            raise ValueError("x must be an evenly spaced grid.")

        # V_xt = potential_scale * I_xt + potential_offset

        # 修改为截尾高斯波包
        # p0=2e-26
        # psi = torch.exp(-(x_t - a) ** 2 / (4 * self.sigma0**2)- 1j * p0 * x_t / hbar).to(torch.complex128)
        # # mask = torch.abs(x_t - a) <= (2.0*self.sigma0)
        # # psi = psi * mask.to(torch.complex128)
        # psi /= torch.sqrt(torch.sum(torch.abs(psi) ** 2) * dx)

        psi=self._harmonic_eigenstate(self.harm_n, x_t, a, self.mass, V_xt[0].max(), self.sigma0).to(torch.complex128)
        psi /= torch.sqrt(torch.sum(torch.abs(psi) ** 2) * dx)
        state=[psi.cpu()]

        nt = t_t.shape[0]
        norms = torch.zeros(nt, dtype=torch.float64, device=device)

        n = len(x_t)
        absorb_width = n//16
        absorb_strength = self.absorb_strength

        pbar = tqdm(total=nt, desc="处理进度")#, disable=True)
        for idx in range(nt):
            pbar.update(1)
            V_now_real = V_xt[idx]
            norms[idx] = torch.sum(torch.abs(psi) ** 2)*dx
            
            if idx == nt - 1:
                break
            dt = float(t_t[idx + 1] - t_t[idx])
            
            # 添加吸收边界
            V_complex = V_now_real.to(torch.complex128)
            # 创建左边界吸收系数
            left_indices = torch.arange(absorb_width, device=x_t.device)
            #left_factors = ((absorb_width - left_indices) / absorb_width) ** 2
            left_factors = torch.cos(torch.pi * left_indices / (2*absorb_width)) ** 2
            V_complex[:absorb_width] -= 1j * absorb_strength * left_factors
            # 创建右边界吸收系数
            right_indices = torch.arange(absorb_width, device=x_t.device)
            #right_factors = (right_indices / absorb_width) ** 2
            right_factors = torch.cos(torch.pi * right_indices / (2*absorb_width)) ** 2
            V_complex[n-absorb_width:] -= 1j * absorb_strength * right_factors
            
            psi = self._cn_step_torch(psi, V_complex, dt, dx, mass)
            state.append(psi.cpu())

            if idx % 100 == 0:
                stats = pbar.format_dict
                self.progress.emit(stats)

        out = {
            "t": t_t,"x": x_t,"norm": norms,"psi": state,}
        
        return out

    def _get_obs(self, psi_list, dx):
        """
        简洁向量化版本
        """
        # 堆叠波函数
        psi = torch.stack(psi_list)  # (nt, nx)
        
        # 一阶导数
        dpsi = torch.zeros_like(psi)
        dpsi[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2*dx)
        # 边界
        dpsi[:, 0] = (psi[:, 1] - psi[:, 0]) / dx
        dpsi[:, -1] = (psi[:, -1] - psi[:, -2]) / dx
        
        # 二阶导数
        d2psi = torch.zeros_like(psi)
        d2psi[:, 1:-1] = (psi[:, 2:] - 2*psi[:, 1:-1] + psi[:, :-2]) / dx**2
        # 边界
        d2psi[:, 0] = (psi[:, 1] - 2*psi[:, 0]) / dx**2
        d2psi[:, -1] = (psi[:, -2] - 2*psi[:, -1]) / dx**2
        
        # 动能
        E_kin = -hbar**2/(2*self.mass) * torch.sum(torch.conj(psi) * d2psi, dim=1) * dx
        E_kin = E_kin.real
        
        # 动量
        p_avg = -1j * hbar * torch.sum(torch.conj(psi) * dpsi, dim=1) * dx
        p_avg = p_avg.real
        
        return E_kin, p_avg

    def run_simulation(self, **kwargs):
        self.refresh_param()

        nt,nx=self.I_data.shape
        self.x_list = torch.linspace(0.0, self.pixel_size*self.nx_raw, nx, dtype=torch.float64)
        self.t_list = torch.linspace(0.0, self.nt_raw*self.t_range/self.rec_freq, nt, dtype=torch.float64)
        V_data=self.potential_scale * self.I_data/self.expos + self.potential_offset

        out = self.simulate_tweezer_heating(
            self.x_list, self.t_list, V_data, a=float(self.x0_start),mass=self.mass)

        t_np = out['t'].cpu().numpy()
        popular=out['norm'].cpu().numpy()

        # #取每时刻光强最大值为井深，作图判断是否逃逸
        # I_max = self.I_data.max(dim=1).values.cpu().numpy()
        # #将I转换到1-2之间
        # V_max = (I_max - I_max.min()) / (I_max.max() - I_max.min()) * (2 - 1) + 1

        V_max = V_data.min(dim=1).values.cpu().numpy()

        # 1) 温度 vs 时间
        dg_p = DataGroup(0,"popular",t_np,popular,plot_type="线图")
        dg_v = DataGroup(0,"potential",t_np,V_max,plot_type="线图")

        # 2) 波函数密度热图
        psi_list = out.get('psi')
        psi_tensor = torch.stack([s.to(torch.complex128) for s in psi_list])
        density = (torch.abs(psi_tensor) ** 2).cpu().numpy().T
        x_np = out['x'].cpu().numpy()
        extent = [t_np[0], t_np[-1],x_np[0] * 1e6, x_np[-1] * 1e6]
        dg_heat = DataGroup(0,"波函数",extent,density,plot_type="热图",)

        E_kin, p_avg = self._get_obs(psi_list, dx=float(self.x_list[1] - self.x_list[0]))
        dg_E_kin = DataGroup(0,"E_kin",t_np,E_kin,plot_type="线图")
        dg_p_avg = DataGroup(0,"momentum",t_np,p_avg,plot_type="线图")

        np.save('psi_end.npy',psi_list[-1].cpu().numpy())

        # 更新派生参数显示
        #self.T_final_K = out.get('T_final_K', 0.0)
        self.update_derived_params()

        if self.I_refresh:
            self.I_refresh = False
            dg_I = DataGroup(0,"光强",extent,self.I_data.cpu().numpy().T,plot_type="热图")
            return [[dg_I],[dg_p],[dg_v, dg_E_kin], [dg_heat]]

        return [[dg_p], [dg_v, dg_E_kin], [dg_heat]]
