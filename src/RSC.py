from tqdm import tqdm
import numpy as np
from functools import partial
import multiprocessing as mp

from arc import Rubidium87
import qutip as qt
from qutip import parallel_map
from sympy.physics.wigner import wigner_6j, wigner_3j
#物理常数定义
from scipy.constants import hbar, e, c, pi,epsilon_0,k
a0 = 5.29177e-11    # 波尔半径 (m)
u = 1.660539e-27    # 相对原子质量 (kg)

from src.PhysicalSimu import BasePhysicalSimulation, DataGroup

def simulate_single(params, LD_r, LD_op, I_vib, a, I_light, H_self, rho0_eff, t_list, c_ops_div, MP, options,delta):
    """单个参数点的模拟函数"""
    op, rb = params
    H_eff = 0.5 * qt.Qobj([
        [delta, rb, op],
        [rb, -delta, 0],
        [op, 0, 0]
    ])
    
    H_drive_op2 = qt.Qobj([
        [0, -LD_r * rb, -LD_op * op],
        [LD_r * rb, 0, 0],
        [LD_op * op, 0, 0]
    ])
    
    H_all = qt.tensor(H_eff, I_vib) + 1j * 0.5 * qt.tensor(H_drive_op2, a + a.dag()) + qt.tensor(I_light, H_self)
    
    result = qt.mesolve(H_all, rho0_eff, t_list, c_ops_div, [MP], options=options)
    return result.expect[0][-1] - result.expect[0][0]

class RSCsimu(BasePhysicalSimulation):
    def __init__(self):
        super().__init__()
        self.name = "RSC模拟"
        self.description = "Reman Sideband cooling"

        self.simu_fun={
            'Rabi oscillation':self.simu_threelevel_div,
            'Raman spectrum':self.simu_Raman_spectrum,
            'twophoton trans':self.simu_twophoton,
            'Guass pulse':self.simu_guass_pulse,
            'Cooling heatmap':self.simu_cooling_heatmap
        }

        self.define_parameters()
        #初始化能级
        self.atom87 = None
        self.qnum_g = [5, 0, 0.5, 1, 1]
        self.qnum_f = [5, 0, 0.5, 2, 2]
        self.qnum_ep = [5, 1, 0.5, 2 ,2]
        self.qnum_e = [5, 1, 1.5, 2, 2]

        self.options = {
            #'method': 'adams',
            'method': 'bdf',
            'nsteps': 80000,      # 增加最大步数
            'atol': 1e-6,         # 绝对误差容限
            'rtol': 1e-6,         # 相对误差容限
            'store_states': False # 不存储所有状态以节省内存
        }

        self.omg_eff=8888
        self.T_pi=8888
        self.omg_op=8888
        self.n_bar=8888
        self.LD_r=8888
        self.LD_op=8888
        self.final_n=8888

    def define_parameters(self):
        #初始化实验参数
        self.params = {
            'simu_type':{'value': 'Rabi oscillation', 'desc': '模拟模式', 'unit': '',
                          'type': 'combo', 'options': self.simu_fun.keys()},
            'time_steps': {'value': 1000, 'min': 1, 'max': 80000, 'step': 1, 
                         'desc': 'time steps', 'unit': 'a.u.'},
            't_end_factor': {'value': 20, 'min': 0.0, 'max': 1e20, 'step': 0.1, 
                         'desc': 't_end factor', 'unit': 'a.u.'},
            'E_level': {'value': 10, 'min': 1, 'max': 500, 'step': 1, 
                         'desc': 'E level', 'unit': 'a.u.'},
            # 'Delta_raman': {'value': 2*pi*5.3*1e9, 'min': 0.1, 'max': 1e20, 'step': 0.1, 
            #              'desc': 'Delta raman', 'unit': 'Hz'},
            'delta_2photon': {'value': 40000, 'min': -1e7, 'max': 1e7, 'step': 0.1, 
                         'desc': 'delta 2photon', 'unit': 'Hz'},
            'PRB1': {'value': 0.001, 'min': 0.0, 'max': 10.0, 'step': 0.1, #2.01e-4
                         'desc': 'P RB1', 'unit': 'J/s'},
            'PRB2': {'value': 0.0015, 'min': 0.0, 'max': 10.0, 'step': 0.1, #3.07e-4
                         'desc': 'P RB2', 'unit': 'J/s'},
            # 'w0_rb': {'value': 0.745e-3, 'min': 0.0, 'max': 1.0, 'step': 0.1, 
            #              'desc': 'w0 rb', 'unit': 'm'},
            # 'freq_op': {'value': 384.234e12, 'min': 0.0, 'max': 1e20, 'step': 0.1, 
            #              'desc': 'freq op', 'unit': 'Hz'},
            'P_op': {'value': 3e-7, 'min': 0.0, 'max': 10.0, 'step': 0.1, 
                         'desc': 'P op', 'unit': 'J/s'},
            # 'w0_op': {'value': 0.764e-3, 'min': 0.0, 'max': 1.0, 'step': 0.1, 
            #              'desc': 'w0 op', 'unit': 'm'},
            'delta_k': {'value': 1.12e7, 'min': 0.0, 'max': 1e20, 'step': 0.1, 
                         'desc': 'delta k', 'unit': 'm-1'},
            'Tempture': {'value': 1e-5, 'min': 0.0, 'max': 10.0, 'step': 0.1, 
                         'desc': 'Tempture', 'unit': 'K'},
            'omg_trap': {'value': 8.35e5, 'min': 0.1, 'max': 1e20, 'step': 0.1, 
                         'desc': 'omg trap', 'unit': 'Hz'},
            # 'type': {'value': 1, 'min': 1, 'max': 10, 'step': 1, 
            #              'desc': 'type', 'unit': 'a.u.'},
            'show_n': {'value': -1, 'min': -10, 'max': 100, 'step': 1, 
                         'desc': 'show_n', 'unit': 'a.u.'},
        }

    def get_derived_parameters(self):
        derived_parameters={
            'omg_eff':{'value':self.omg_eff/2/pi, 'desc': '拉比频率', 'unit': '2piHz'},
            'Tpi':{'value':self.T_pi*1e6, 'desc': 'pi周期', 'unit': 'us'},
            'omg_op':{'value':self.omg_op/2/pi, 'desc': 'op拉比频率', 'unit': '2piHz'},
            'n_bar':{'value':self.n_bar, 'desc': '声子数', 'unit': 'a.u.'},
            'LD_r':{'value':self.LD_r, 'desc': 'LD_r', 'unit': 'a.u.'},
            'LD_op':{'value':self.LD_op, 'desc': 'LD_op', 'unit': 'a.u.'},
            #'final_n':{'value':self.final_n, 'desc': '最终声子数', 'unit': 'a.u.'},
        }
        return derived_parameters

    def refresh_param(self):
        params = self.get_parameters()
        self.simutype=params.get('simu_type', None)
        self.Delta_raman = 2*pi*5.3*1e9#params.get('Delta_raman', 1.0)
        self.delta_2photon = params.get('delta_2photon', 1.0)
        self.PRB1 = params.get('PRB1', 1.0)
        self.PRB2 = params.get('PRB2', 1.0)
        self.w0_rb = 0.745e-3#params.get('w0_rb', 1.0)
        self.freq_op=384.234e12#params.get('freq_op', 1.0)
        self.P_op=params.get('P_op', 1.0)
        self.w0_op=0.764e-3#params.get('w0_op', 1.0)
        self.delta_k=params.get('delta_k', 1.0)
        self.Tempture=params.get('Tempture', 1.0)
        self.omg_trap=params.get('omg_trap', 1.0)

        dRB1 = self.get_hfs_dipole_matrix_element(self.qnum_g,self.qnum_ep ,1)
        dRB2 = self.get_hfs_dipole_matrix_element(self.qnum_f,self.qnum_ep ,0)
        d_ge = self.get_hfs_dipole_matrix_element(self.qnum_g,self.qnum_e, 1)
        self.Omega1 = self.get_rabi_frequency(dRB1 * e * a0, self.PRB1, self.w0_rb)
        self.Omega2 = self.get_rabi_frequency(dRB2 * e * a0, self.PRB2, self.w0_rb)
        self.omg_op = self.get_rabi_frequency(d_ge*e*a0,self.P_op,self.w0_op)

        self.omg_eff = (self.Omega1 * self.Omega2) / (2 * self.Delta_raman)
        self.delta_eff=0#((self.Omega1)**2-(self.Omega2)**2)/4/self.Delta_raman
        self.T_pi = pi / self.omg_eff
        #模拟参数
        self.time_steps=int(params.get('time_steps', 1.0))
        self.t_end_factor=params.get('t_end_factor', 1.0)
        self.t_list = np.linspace(0, self.t_end_factor*self.T_pi, self.time_steps)
        self.N_max=int(params.get('E_level', 1.0))
        # self.simutype=int(params.get('type', 1.0))
        self.show_n=int(params.get('show_n', 1.0))

    def run_simulation(self, **kwargs):
        self.refresh_param()
        
        #x,y,plot_type=self.simu_fun[self.simutype]()
        #换为本地变量访问，防止线程崩溃
        show_func = self.simu_fun.copy()  # 复制字典
        show_key = str(self.simutype)  # 复制为字符串
        if show_key in show_func:
            func = show_func[show_key]
            x, y, plot_type = func()

        self.update_derived_params()
        label = f"{self.name} ({self.simutype})"
        dg = DataGroup(1, label, x, y, color=(0.0, 0.4, 0.8, 1.0), plot_type=plot_type)
        return [[dg]]

    def get_rabi_frequency(self,dipole_matrix_element, P, w0):
        # Rabi frequency Omega = |d·E|/hbar
        E_field = np.sqrt(2 * P / (np.pi * w0**2 * epsilon_0 * c))  # 电场强度 (V/m)
        Omega = abs(dipole_matrix_element * E_field) / hbar
        return Omega

    #由于arc库只提供了到j的结构常数，因此需要利用wigner定理计算F超精细常数的情况
    def get_hfs_dipole_matrix_element(self,qnum1,qnum2,q):
        n1,l1,j1,F1,mF1=qnum1
        n2,l2,j2,F2,mF2=qnum2
        #q for polarization: -1 (sigma-), 0 (pi), +1 (sigma+)
        # d_j = atom87.getDipoleMatrixElement(n1,l1,j1,n2,l2,j2)
        # 为避免 sqlite 连接在不同线程间共享，需在调用线程中创建 Rubidium87 实例
        atom = Rubidium87()
        d_j = atom.getReducedMatrixElementJ(n1,l1,j1,n2,l2,j2)

        #calculate the wigner 6-j symbol
        I = atom.I
        coeff_6j = wigner_6j(j1, F1, I, F2, j2, 1)
        #calculate the wigner 3-j symbol
        coeff_3j = wigner_3j(F2, 1, F1, -mF2, q, mF1)
        #Total dipole matrix element
        d_F = (-1)**(F2 + j1 + 1 + I) * np.sqrt((2*F1 + 1)*(2*F2 + 1)) * coeff_6j * coeff_3j * d_j
        return float(d_F)
    
    def get_hyperfine_TransitionRate(self,qnum1,qnum2):
        n1,l1,j1,F1,mF1=qnum1
        n2,l2,j2,F2,mF2=qnum2
        atom = Rubidium87()
        A_J=atom.getTransitionRate(n1,l1,j1,n2,l2,j2)
        # 检查选择定则
        if abs(F1 - F2) > 1 or abs(mF1 - mF2) > 1:
            return 0.0
        # 计算6j符号
        sixj_sq = float(wigner_6j(j1, F1, atom.I, F2, j2, 1))**2
        # 计算3j符号
        delta_m = mF1 - mF2
        threej_sq = float(wigner_3j(F1, 1, F2, -mF1, delta_m, mF2))**2
        # 计算A系数
        A_F = A_J * (2*j2 + 1) * (2*F1 + 1) * sixj_sq * threej_sq
        
        return A_F

    def simu_twophoton(self):
        freq_RB1=377e9+100e6+6652.56e6
        freq_RB2=377e9-100e6

        k1=freq_RB1/2/pi/c
        k2=freq_RB2/2/pi/c
        meight=87*u
        uho=np.sqrt(hbar/2/meight/self.omg_trap)
        LD_1=uho*k1
        LD_2=uho*k2

        #从F=2态开始演化
        psi0 = qt.Qobj([[0], [1], [0]])  
        # 振动空间算符
        a = qt.destroy(self.N_max+1)  # 湮灭算符
        I_vib = qt.qeye(self.N_max+1)
        I_light=qt.qeye(3)

        # 初始态：内态在|2>，振动在热态
        self.n_bar = 1 / (np.exp(hbar * self.omg_trap / (k * self.Tempture)) - 1)# 平均声子数
        
        rho_vib_thermal = qt.thermal_dm(self.N_max+1, self.n_bar)  # 热态密度矩阵
        psi_n = qt.basis(self.N_max+1, 0)  # 波函数
        #rho_vib_thermal = qt.ket2dm(psi_n)      # 转化为密度矩阵
        rho_int_eff = qt.ket2dm(psi0)  # 转化为密度矩阵
        rho0_eff = qt.tensor(rho_int_eff, rho_vib_thermal)  # 内态纯态 ⊗ 振动热态

        #创建投影测量算符
        Pe = qt.Qobj([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
        P2 = qt.Qobj([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
        P1 = qt.Qobj([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]])
        if self.show_n==1:
            MP=qt.tensor(P1,I_vib)
        elif self.show_n==2:
            MP=qt.tensor(P2,I_vib)
        elif self.show_n==3:
            MP=qt.tensor(Pe,I_vib)

        H = qt.Qobj([[self.Delta_raman,self.Omega2/2,self.Omega1/2],
                     [self.Omega2/2,-self.delta_2photon,0],
                     [self.Omega1/2,0,0]])
        H_drive = qt.Qobj([
                [0, -LD_2*self.Omega2,-LD_1*self.Omega1],
                [LD_2*self.Omega2, 0, 0],
                [LD_1*self.Omega1, 0, 0]])
        
        gamma_op1 = self.get_hyperfine_TransitionRate(self.qnum_ep,self.qnum_f)
        c_ops1 = np.sqrt(gamma_op1) * qt.Qobj([[0, 0, 0],
                                            [1, 0, 0],
                                            [0, 0, 0]])
        gamma_op2 = self.get_hyperfine_TransitionRate(self.qnum_ep,self.qnum_g)
        c_ops2 = np.sqrt(gamma_op2) * qt.Qobj([[0, 0, 0],
                                            [0, 0, 0],
                                            [1, 0, 0]])
        c_ops_div=qt.tensor(c_ops1+c_ops2,I_vib)
        
        H_self=self.omg_trap*(a.dag()*a+0.5)
        H_all=qt.tensor(H, I_vib)+1j*0.5*qt.tensor(H_drive,a+a.dag())+qt.tensor(I_light,H_self)
        result = qt.mesolve(H_all, rho0_eff, self.t_list, c_ops_div, [MP], options=self.options,progress_bar='tqdm')
        x=self.t_list * 1e6
        #output=[result.expect[0],result.expect[1],result.expect[2]]
        return x,result.expect[0],"线图"

    def simu_threelevel_div(self):
        k_op=self.freq_op/2/pi/c
        meight=87*u
        uho=np.sqrt(hbar/2/meight/self.omg_trap)
        self.LD_r = uho*self.delta_k  # Lamb-Dicke 参数
        self.LD_op = uho*k_op  # Lamb-Dicke 参数

        # 振动空间算符
        a = qt.destroy(self.N_max+1)  # 湮灭算符
        I_vib = qt.qeye(self.N_max+1)
        I_light=qt.qeye(3)

        #从F=2态开始演化
        psi0 = qt.Qobj([[0], [1], [0]])  
        # 初始态：内态在|2>，振动在热态
        self.n_bar = 1 / (np.exp(hbar * self.omg_trap / (k * self.Tempture)) - 1)# 平均声子数
        
        rho_vib_thermal = qt.thermal_dm(self.N_max+1, self.n_bar)  # 热态密度矩阵
        # psi_n = qt.basis(self.N_max+1, 2)  # 波函数
        # rho_vib_thermal = qt.ket2dm(psi_n)      # 转化为密度矩阵
        rho_int_eff = qt.ket2dm(psi0)  # 转化为密度矩阵
        rho0_eff = qt.tensor(rho_int_eff, rho_vib_thermal)  # 内态纯态 ⊗ 振动热态

        #创建投影测量算符
        P2 = qt.Qobj([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
        if self.show_n==-1:
            MP=qt.tensor(P2,I_vib)
        elif self.show_n==-2:
            # 定义声子数算符
            n_operator = a.dag() * a
            MP = qt.tensor(I_light, n_operator)  # 整个空间的声子数算符
        else:
            MP=qt.tensor(P2,qt.ket2dm(qt.basis(self.N_max+1, self.show_n)))
        # 定义声子数算符
        n_operator = a.dag() * a
        nMP = qt.tensor(I_light, n_operator)  # 整个空间的声子数算符

        H_eff = 0.5*qt.Qobj([
                [self.delta_2photon+self.delta_eff, self.omg_eff,self.omg_op],
                [self.omg_eff, -self.delta_2photon-self.delta_eff,0],
                [self.omg_op, 0, 0]
            ])
        H_drive_op2 = qt.Qobj([
                [0, -self.LD_r*self.omg_eff,-self.LD_op*self.omg_op],
                [self.LD_r*self.omg_eff, 0, 0],
                [self.LD_op*self.omg_op, 0, 0]
            ])
        
        H_self=self.omg_trap*(a.dag()*a+0.5)
        H_all=qt.tensor(H_eff, I_vib)+1j*0.5*qt.tensor(H_drive_op2,a+a.dag())+qt.tensor(I_light,H_self)

        gamma_op1 = self.get_hyperfine_TransitionRate(self.qnum_e,self.qnum_f)
        #print('1',gamma_op1)
        gamma_op2 = self.get_hyperfine_TransitionRate(self.qnum_e,self.qnum_g)
        #print('2',gamma_op2)
        c_ops1 = np.sqrt(gamma_op1) * qt.Qobj([[0, 0, 0],
                                            [0, 0, 1],
                                            [0, 0, 0]])
        c_ops2 = np.sqrt(gamma_op2) * qt.Qobj([[0, 0, 1],
                                            [0, 0, 0],
                                            [0, 0, 0]])
        c_ops_div=qt.tensor(c_ops1+c_ops2,I_vib)
        #c_ops_div=qt.tensor(c_ops2,I_vib)

        result = qt.mesolve(H_all, rho0_eff, self.t_list, c_ops_div, [MP], options=self.options, progress_bar='tqdm')
        x=self.t_list * 1e6
        y=np.array(result.expect[0])
        return x,y,"线图"

    def simu_Raman_spectrum(self):
        delta_list = self.delta_2photon*np.linspace(-1,1,self.time_steps)
        output=[]
        for delta_2photon in tqdm(delta_list):
            self.delta_2photon=delta_2photon
            self.show_n=-1
            _,y,_=self.simu_threelevel_div()
            output.append(y[-1])
        return delta_list,output,"线图"
    
    def simu_guass_pulse(self):
        k_op=self.freq_op/2/pi/c
        meight=87*u
        uho=np.sqrt(hbar/2/meight/self.omg_trap)
        self.LD_r = uho*self.delta_k  # Lamb-Dicke 参数
        self.LD_op = uho*k_op  # Lamb-Dicke 参数
        self.omg_op=0

        # 振动空间算符
        a = qt.destroy(self.N_max+1)  # 湮灭算符
        I_vib = qt.qeye(self.N_max+1)
        I_light=qt.qeye(3)

        #从F=2态开始演化
        psi0 = qt.Qobj([[0], [1], [0]])  
        # 初始态：内态在|2>，振动在热态
        self.n_bar = 1 / (np.exp(hbar * self.omg_trap / (k * self.Tempture)) - 1)# 平均声子数
        rho_vib_thermal = qt.thermal_dm(self.N_max+1, self.n_bar)  # 热态密度矩阵
        # psi_n = qt.basis(self.N_max+1, 2)  # 波函数
        # rho_vib_thermal = qt.ket2dm(psi_n)      # 转化为密度矩阵
        rho_int_eff = qt.ket2dm(psi0)  # 转化为密度矩阵
        rho0_eff = qt.tensor(rho_int_eff, rho_vib_thermal)  # 内态纯态 ⊗ 振动热态

        def pulse_func(t, args):
            t0 = args.get('t0', 0.0)  # 脉冲中心时间
            sigma = args.get('sigma', 1.0)  # 脉冲宽度（标准差）
            A = args.get('A', 1.0)  # 脉冲幅度，默认值为1.0
            
            return A * np.exp(-(t - t0)**2 / (2 * sigma**2))
        
        args = {'t0': 0.5*pi/self.omg_eff,'sigma': self.t_end_factor*pi/self.omg_eff,'A':self.P_op}
            
        H_self = self.omg_trap * (a.dag() * a + 0.5)
        H_free_op = qt.Qobj([
                [0, 0, -self.LD_op * self.omg_op],
                [0, 0, 0],
                [self.LD_op * self.omg_op, 0, 0]])
        H_free_eff = 0.5 * qt.Qobj([
                [0, 0, self.omg_op],
                [0, 0, 0],
                [self.omg_op, 0, 0]])
        H_drive_op = qt.Qobj([
                [0, -self.LD_r * self.omg_eff, 0],
                [self.LD_r * self.omg_eff, 0, 0],
                [0, 0, 0]])
        
        t_list2= np.linspace(0, pi/self.omg_eff, 100)

        gamma_op1 = self.get_hyperfine_TransitionRate(self.qnum_e,self.qnum_f)
        gamma_op2 = self.get_hyperfine_TransitionRate(self.qnum_e,self.qnum_g)
        c_ops1 = np.sqrt(gamma_op1) * qt.Qobj([[0, 0, 0],
                                            [0, 0, 1],
                                            [0, 0, 0]])
        c_ops2 = np.sqrt(gamma_op2) * qt.Qobj([[0, 0, 1],
                                            [0, 0, 0],
                                            [0, 0, 0]])
        c_ops_div=qt.tensor(c_ops1+c_ops2,I_vib)

        P2 = qt.Qobj([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
        MP=qt.tensor(P2,I_vib)

        delta_list = self.delta_2photon*np.linspace(-1,1,self.time_steps)
        output=[]
        for delta_2photon in tqdm(delta_list):
            H_drive_eff = 0.5 * qt.Qobj([
                    [delta_2photon, self.omg_eff, 0],
                    [self.omg_eff, -delta_2photon, 0],
                    [0, 0, 0]])
            H_free = qt.tensor(H_free_eff, I_vib) + 1j * 0.5 * qt.tensor(H_free_op, a + a.dag()) + qt.tensor(I_light, H_self)
            H_drive = qt.tensor(H_drive_eff, I_vib) + 1j * 0.5 * qt.tensor(H_drive_op, a + a.dag())
            H_total = [H_free, [H_drive, pulse_func]]

            result = qt.mesolve(H_total, rho0_eff, t_list2, [c_ops_div], MP, args=args, options=self.options)#, progress_bar='tqdm')
            see=np.array(result)
            print(see)
            output.append(result.expect[0][-1])
        x=(self.t_list/self.t_list[-1]-0.5)*2*self.delta_2photon
        return x,output,"线图"

    def simu_cooling_heatmap(self):
        # 参数范围设置
        op_start, op_end = 1e4*2*pi, 80e4*2*pi
        rb_start, rb_end = 1e3*2*pi, 10e4*2*pi
        n_points = self.time_steps

        op_Rabi_list = np.linspace(op_start, op_end, n_points)
        rb_Rabi_list = np.linspace(rb_start, rb_end, n_points)

        k_op=self.freq_op/2/pi/c
        meight=87*u
        uho=np.sqrt(hbar/2/meight/self.omg_trap)
        self.LD_r = uho*self.delta_k  # Lamb-Dicke 参数
        self.LD_op = uho*k_op  # Lamb-Dicke 参数
        
        a = qt.destroy(self.N_max + 1)
        n_operator = a.dag() * a
        I_vib = qt.qeye(self.N_max + 1)
        I_light = qt.qeye(3)

        psi0 = qt.Qobj([[0], [1], [0]])
        self.n_bar = 1 / (np.exp(hbar * self.omg_trap / (k * self.Tempture)) - 1)# 平均声子数
        rho_vib_thermal = qt.thermal_dm(self.N_max + 1, self.n_bar)
        rho_int_eff = qt.ket2dm(psi0)
        rho0_eff = qt.tensor(rho_int_eff, rho_vib_thermal)

        MP = qt.tensor(I_light, n_operator)
        H_self = self.omg_trap * (a.dag() * a + 0.5)

        gamma_op1 = self.get_hyperfine_TransitionRate(self.qnum_e,self.qnum_f)
        gamma_op2 = self.get_hyperfine_TransitionRate(self.qnum_e,self.qnum_g)
        c_ops1 = np.sqrt(gamma_op1) * qt.Qobj([[0, 0, 0],[0, 0, 1],[0, 0, 0]])
        c_ops2 = np.sqrt(gamma_op2) * qt.Qobj([[0, 0, 1],[0, 0, 0],[0, 0, 0]])
        c_ops_div=qt.tensor(c_ops1+c_ops2,I_vib)

        t_list = np.linspace(0, 4000e-6, 200)
        options = qt.Options(method='bdf', nsteps=80000, atol=1e-6, rtol=1e-6, store_states=False)

        param_list = [(op, rb) for op in op_Rabi_list for rb in rb_Rabi_list]
        num_cpus = max(1, mp.cpu_count() - 8)

        func = partial(simulate_single,
            LD_r=self.LD_r, LD_op=self.LD_op, I_vib=I_vib, a=a, I_light=I_light,
            H_self=H_self, rho0_eff=rho0_eff, t_list=t_list, c_ops_div=c_ops_div,
            MP=MP, options=options, delta=self.delta_2photon)

        results = parallel_map(func, param_list, map_kw=dict(num_cpus=num_cpus), progress_bar=True)
        y = np.array(results).reshape((n_points, n_points))

        return [rb_start, rb_end, op_start, op_end],y,'热图'

    # def RSC_fit(self):
    #     output=self.omg_trap*np.exp(-self.t_list/self.Tempture)*np.cos(self.omg_eff*self.t_list+pi)+self.delta_k
    #     return output    
