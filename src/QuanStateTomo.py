import numpy as np
import torch
from typing import List, Dict, Optional,Tuple
from abc import ABC, abstractmethod
#warnings.filterwarnings('ignore')

# 导入Qiskit相关模块
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity,random_clifford
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, thermal_relaxation_error, ReadoutError, reset_error
import scipy.optimize as opt
import itertools

from src.PhysicalSimu import BasePhysicalSimulation, DataGroup

class QuantumStateTomography(ABC):
    """量子态层析基类，定义统一接口"""
    
    def __init__(self, n_qubits: int, shots_per_measurement: int = 1000):
        """
        初始化量子态层析器
        
        参数:
            n_qubits: 量子比特数
            shots_per_measurement: 每次测量的采样次数
        """
        self.n_qubits = n_qubits
        self.shots_per_measurement = shots_per_measurement
        self.state_prep_circuit = None
        self.true_state = None
        self.reconstructed_state = None
        
    def set_state_preparation(self, circuit: QuantumCircuit) -> DensityMatrix:
        """设置量子态制备电路"""
        self.state_prep_circuit = circuit
        # 计算理想状态
        statevector = Statevector.from_instruction(circuit)
        self.true_state = DensityMatrix(statevector)
        return self.true_state
        
    @abstractmethod
    def tomography_state(self, noise_model: Optional[NoiseModel] = None) -> DensityMatrix:
        """执行态层析，返回重建的密度矩阵"""
        pass

    def predict_observable(self, observable: np.ndarray, noise_model: Optional[NoiseModel] = None) -> float:
        '''预测可观测量'''
        if self.reconstructed_state is None:
            self.tomography_state(noise_model)

        observable_mean = np.trace(observable @ self.reconstructed_state.data)
        self.chunk_estimates = [np.real(observable_mean)]
        return observable_mean

class LinearInversionTomography(QuantumStateTomography):
    """线性反转态层析 - 最标准的方法"""
    
    def __init__(self, n_qubits: int, shots_per_measurement: int = 1000):
        super().__init__(n_qubits, shots_per_measurement)
        
    def _generate_pauli_basis(self) -> List[List[str]]:
        """生成所有Pauli测量基组合"""
        pauli_ops = ['I', 'X', 'Y', 'Z']
        # 生成所有可能的组合
        basis_combinations = list(itertools.product(pauli_ops, repeat=self.n_qubits))
        return [list(basis) for basis in basis_combinations]
    
    def _add_measurement_basis(self, circuit: QuantumCircuit, basis: List[str]) -> QuantumCircuit:
        """为电路添加测量基变换"""
        measured_circuit = circuit.copy()
        
        for qubit_idx, pauli in enumerate(basis):
            if pauli == 'X':
                measured_circuit.h(qubit_idx)
            elif pauli == 'Y':
                measured_circuit.sdg(qubit_idx)
                measured_circuit.h(qubit_idx)
            elif pauli == 'I':
                pass  # 'I'基不需要额外操作，与'Z'相同
            # 'Z'基不需要额外操作
        
        # 添加测量
        measured_circuit.measure_all()
        return measured_circuit
    
    def _get_expectation_value(self, measurement_results: Dict, pauli: List[str]) -> float:
        """计算Pauli算符的期望值"""
        expectation = 0
        total_shots = sum(measurement_results.values())
        
        for bitstring, count in measurement_results.items():
            # 计算这个结果对应的本征值
            eigenvalue = 1
            for qubit_idx, (bit, pauli_op) in enumerate(zip(bitstring[::-1], pauli)):
                if pauli_op == 'X':
                    # 对于X基，|0>对应+1，|1>对应-1
                    eigenvalue *= 1 if bit == '0' else -1
                elif pauli_op == 'Y':
                    # 对于Y基，|0>对应+1，|1>对应-1
                    eigenvalue *= 1 if bit == '0' else -1
                elif pauli_op == 'Z':
                    # 对于Z基，|0>对应+1，|1>对应-1
                    eigenvalue *= 1 if bit == '0' else -1
                elif pauli_op == 'I':
                    # 对于I，|0>和|1>都对应+1
                    eigenvalue *= 1
            
            expectation += eigenvalue * count / total_shots
            
        return expectation
    
    def _linear_inversion(self, expectation_values: Dict, n_qubits: int) -> np.ndarray:
        """执行线性反转重建密度矩阵"""
        dim = 2 ** n_qubits
        rho_reconstructed = np.zeros((dim, dim), dtype=complex)
        
        # Pauli基的矩阵表示
        pauli_matrices = {
            'I': np.array([[1, 0], [0, 1]]),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        # 遍历所有Pauli基
        for pauli_str, expectation in expectation_values.items():
            # 构造对应的Pauli矩阵
            pauli_list = list(pauli_str)

            # Qiskit 的张量乘积顺序为 |q_{n-1} ... q_0>,
            # 这里需要将 pauli_list 反向以保证第0号量子比特对应最低阶因子
            pauli_tensor = pauli_matrices[pauli_list[-1]]
            for pauli_op in reversed(pauli_list[:-1]):
                pauli_tensor = np.kron(pauli_matrices[pauli_op], pauli_tensor)

            # 累加
            rho_reconstructed += expectation * pauli_tensor / dim
        
        # 归一化以确保迹为1
        rho_reconstructed = rho_reconstructed / np.trace(rho_reconstructed)
        
        # 投影到密度矩阵空间（确保正定）
        eigvals, eigvecs = np.linalg.eigh(rho_reconstructed)
        eigvals[eigvals < 0] = 0
        eigvals = eigvals / np.sum(eigvals) if np.sum(eigvals) > 0 else eigvals
        rho_reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        return DensityMatrix(rho_reconstructed)
    
    def tomography_state(self, noise_model: Optional[NoiseModel] = None) -> DensityMatrix:
        """执行完整的线性反转态层析"""
        if self.state_prep_circuit is None:
            raise ValueError("请先设置态制备电路")
        
        # 获取所有Pauli基测量设置
        pauli_basis = self._generate_pauli_basis()
        expectation_values = {}
        # 创建支持并行的模拟器
        # NOTE: 如果指定了噪声模型，应使用 density_matrix 方法来让噪声生效；
        #       statevector 方法会忽略噪声模型。
        simulator = AerSimulator(
            noise_model=noise_model,
            method='density_matrix' if noise_model is not None else 'statevector',
            device='CPU',
            max_parallel_threads=0,
            max_parallel_experiments=0)
        # 批量处理所有测量基电路
        circuits = []
        basis_list = []
        # 创建所有测量电路
        for basis in pauli_basis:
            # 创建带测量基变换的电路
            measured_circuit = self._add_measurement_basis(self.state_prep_circuit, basis)
            circuits.append(measured_circuit)
            basis_list.append(basis)
        # 批量编译所有电路
        compiled_circuits = transpile(circuits, simulator, optimization_level=2)
        # 批量执行所有电路（并行处理）
        job = simulator.run(compiled_circuits, shots=self.shots_per_measurement)
        result = job.result()
        # 处理所有结果
        for i, basis in enumerate(basis_list):
            # 获取第i个电路的测量结果
            counts = result.get_counts(i)
            # 计算期望值
            expectation = self._get_expectation_value(counts, basis)
            # 存储结果
            basis_str = ''.join(basis)
            expectation_values[basis_str] = expectation
        
        # 执行线性反转
        self.reconstructed_state = self._linear_inversion(expectation_values, self.n_qubits)
        
        return self.reconstructed_state

class MaximumLikelihoodTomography(LinearInversionTomography):
    """最大似然估计态层析（扩展自线性反转）"""
    
    def __init__(self, n_qubits: int, shots_per_measurement: int = 1000, max_iter: int = 10):
        super().__init__(n_qubits, shots_per_measurement)
        self.max_iter = max_iter
    
    def _positive_semidefinite_projector(self, rho: np.ndarray) -> np.ndarray:
        """投影到半正定矩阵空间"""
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals[eigvals < 0] = 0
        return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    
    def _trace_one_projector(self, rho: np.ndarray) -> np.ndarray:
        """投影到迹为1的空间"""
        return rho / np.trace(rho)
    
    def _likelihood_function(self, rho_flat: np.ndarray, measurement_circuits: List[QuantumCircuit], 
                           counts_data: List[Dict], simulator: AerSimulator) -> float:
        """似然函数（负对数似然）"""
        dim = int(np.sqrt(len(rho_flat)))
        rho = rho_flat.reshape((dim, dim))
        
        # 确保是密度矩阵
        rho = rho.conj().T @ rho
        rho = rho / np.trace(rho)
        
        likelihood = 0
        for circuit, counts in zip(measurement_circuits, counts_data):
            # 计算理论概率
            circuit_copy = circuit.copy()
            circuit_copy.save_density_matrix()
            
            # 创建包含状态准备和测量的完整电路
            full_circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
            if self.state_prep_circuit:
                full_circuit.compose(self.state_prep_circuit, inplace=True)
            full_circuit.compose(circuit_copy, inplace=True)
            
            # 计算理论测量概率
            for outcome, observed_count in counts.items():
                # 这里简化处理，实际需要计算每个结果的理论概率
                # 在完整实现中，这需要更复杂的计算
                theoretical_prob = 1.0 / (2 ** self.n_qubits)  # 简化假设均匀分布
                if theoretical_prob > 0:
                    likelihood -= observed_count * np.log(theoretical_prob)
        
        return likelihood
    
    def tomography_state_opt(self, noise_model: Optional[NoiseModel] = None) -> DensityMatrix:
        """执行最大似然态层析"""
        if self.state_prep_circuit is None:
            raise ValueError("请先设置态制备电路")
        
        # 首先用线性反转得到初始猜测
        rho_linear = super().tomography_state(noise_model)
        
        # 收集所有测量电路和结果
        pauli_basis = self._generate_pauli_basis()
        measurement_circuits = []
        counts_data = []
        
        # 创建模拟器
        # NOTE: 如果指定噪声模型，则应使用 density_matrix 模拟以让噪声生效
        simulator = AerSimulator(
            noise_model=noise_model,
            method='density_matrix' if noise_model is not None else 'statevector')
        
        for basis in pauli_basis:
            measured_circuit = self._add_measurement_basis(self.state_prep_circuit, basis)
            compiled_circuit = transpile(measured_circuit, simulator)
            
            # 执行测量
            job = simulator.run(compiled_circuit, shots=self.shots_per_measurement)
            result = job.result()
            counts = result.get_counts(compiled_circuit)
            
            counts_data.append(counts)
            measurement_circuits.append(measured_circuit)
        
        # 优化负对数似然函数
        dim = 2 ** self.n_qubits
        
        # 使用Cholesky分解确保正定性
        def rho_from_params(params):
            """从实参数重构密度矩阵"""
            L = np.zeros((dim, dim), dtype=complex)
            idx = 0
            for i in range(dim):
                for j in range(i+1):
                    if i == j:
                        L[i, j] = params[idx]
                        idx += 1
                    else:
                        L[i, j] = params[idx] + 1j * params[idx+1]
                        idx += 2
            rho_temp = L @ L.conj().T
            return rho_temp / np.trace(rho_temp)
        
        # 初始参数：从线性反转结果开始
        eigvals, eigvecs = np.linalg.eigh(rho_linear.data)
        eigvals = np.maximum(eigvals, 1e-10)  # 确保正定性
        initial_rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        initial_rho = initial_rho / np.trace(initial_rho)
        
        # 从密度矩阵提取Cholesky参数
        L_init = np.linalg.cholesky(initial_rho)
        initial_params = []
        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    initial_params.append(np.real(L_init[i, j]))
                else:
                    initial_params.append(np.real(L_init[i, j]))
                    initial_params.append(np.imag(L_init[i, j]))
        
        # 定义目标函数
        def objective(params):
            rho = rho_from_params(params)
            
            # 计算负对数似然
            neg_log_likelihood = 0
            for circuit, counts in zip(measurement_circuits, counts_data):
                # 这里简化计算，实际需要更复杂的概率计算
                for outcome, count in counts.items():
                    # 简化假设：均匀分布
                    prob = 1.0 / (2 ** self.n_qubits)
                    if prob > 0:
                        neg_log_likelihood -= count * np.log(prob)
            
            return neg_log_likelihood
        
        # 优化
        result = opt.minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'ftol': 1e-8, 'disp': False}
        )
        
        # 从优化参数重建密度矩阵
        rho_optimal = rho_from_params(result.x)
        
        return DensityMatrix(rho_optimal)
    
    def tomography_state(self, noise_model: Optional[NoiseModel] = None) -> DensityMatrix:
        """执行最大似然态层析"""
        if self.state_prep_circuit is None:
            raise ValueError("请先设置态制备电路")
        
        # 获取所有Pauli基测量设置
        pauli_basis = self._generate_pauli_basis()
        expectation_values = {}
        # 创建支持并行的模拟器
        # NOTE: 如果指定了噪声模型，应使用 density_matrix 方法来让噪声生效；
        #       statevector 方法会忽略噪声模型。
        simulator = AerSimulator(
            noise_model=noise_model,
            method='density_matrix' if noise_model is not None else 'statevector',
            device='CPU',
            max_parallel_threads=0,
            max_parallel_experiments=0)
        # 批量处理所有测量基电路
        circuits = []
        basis_list = []
        # 创建所有测量电路
        for basis in pauli_basis:
            # 创建带测量基变换的电路
            measured_circuit = self._add_measurement_basis(self.state_prep_circuit, basis)
            circuits.append(measured_circuit)
            basis_list.append(basis)
        # 批量编译所有电路
        compiled_circuits = transpile(circuits, simulator, optimization_level=2)
        # 批量执行所有电路（并行处理）
        job = simulator.run(compiled_circuits, shots=self.shots_per_measurement)
        result = job.result()
        # 处理所有结果
        for i, basis in enumerate(basis_list):
            # 获取第i个电路的测量结果
            counts = result.get_counts(i)
            # 计算期望值
            expectation = self._get_expectation_value(counts, basis)
            # 存储结果
            basis_str = ''.join(basis)
            expectation_values[basis_str] = expectation
        # 包含I的期望值
        expectation_values['I' * self.n_qubits] = 1.0
        # 执行线性反转
        rho_linear = self._linear_inversion(expectation_values, self.n_qubits)
        # 初始参数：从线性反转结果开始
        # 将 rho_linear.data 转换为 GPU 张量
        rho_linear_tensor = torch.tensor(rho_linear.data, dtype=torch.complex64, device='cuda')

        # 初始参数：从线性反转结果开始
        eigvals, eigvecs = torch.linalg.eigh(rho_linear_tensor)
        eigvals = torch.tensor(torch.maximum(eigvals, torch.tensor(1e-6, device='cuda')),dtype=torch.complex64)
        initial_rho = eigvecs @ torch.diag(eigvals) @ eigvecs.conj().T
        initial_rho = initial_rho / torch.trace(initial_rho)

        dim = 2 ** self.n_qubits
        
        # 添加小的正则化项以确保数值稳定性
        epsilon = 1e-8
        initial_rho = initial_rho + epsilon * torch.eye(dim, dtype=torch.complex64, device='cuda')

        # 从密度矩阵提取 Cholesky 参数
        L_init = torch.linalg.cholesky(initial_rho)
        initial_params = []
        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    initial_params.append(L_init[i, j].real.item())
                else:
                    initial_params.append(L_init[i, j].real.item())
                    initial_params.append(L_init[i, j].imag.item())

        # 将初始参数转换为可训练的 GPU 张量
        params = torch.tensor(initial_params, dtype=torch.float32, 
                            device='cuda', requires_grad=True)

        # Pauli 基的矩阵表示（GPU 张量）
        pauli_matrices = {
            'I': torch.tensor([[1, 0], [0, 1]], dtype=torch.complex64, device='cuda'),
            'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device='cuda'),
            'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device='cuda'),
            'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device='cuda')}

        # 预计算所有 Pauli 算符的张量积
        pauli_tensors = {}
        for pauli_str in expectation_values.keys():
            pauli_list = list(pauli_str)
            pauli_tensor = pauli_matrices[pauli_list[0]]
            for pauli_op in pauli_list[1:]:
                pauli_tensor = torch.kron(pauli_tensor, pauli_matrices[pauli_op])
            pauli_tensors[pauli_str] = pauli_tensor

        # 使用 Cholesky 分解确保正定性
        def rho_from_params(params_tensor):
            """从实参数重构密度矩阵"""
            L = torch.zeros((dim, dim), dtype=torch.complex64, device='cuda')
            idx = 0
            for i in range(dim):
                for j in range(i+1):
                    if i == j:
                        L[i, j] = params_tensor[idx]
                        idx += 1
                    else:
                        L[i, j] = params_tensor[idx] + 1j * params_tensor[idx+1]
                        idx += 2
            rho_temp = L @ L.conj().T
            return rho_temp / torch.trace(rho_temp)

        def objective(params_tensor):
            """目标函数"""
            rho = rho_from_params(params_tensor)
            L = torch.tensor(0.0, device='cuda')
            for pauli_str, expectation in expectation_values.items():
                pauli_tensor = pauli_tensors[pauli_str]
                alpha = self.shots_per_measurement * expectation
                # 计算迹
                tr = torch.trace(rho @ pauli_tensor)
                # 数值稳定性处理
                tr_real = tr.real.clamp(1e-10, 1-1e-10)
                tr = tr_real + 1j * tr.imag
                # 计算损失
                Lp = (pauli_tensor * (alpha / tr / (1 - tr) + 
                                    self.shots_per_measurement / (tr - 1))).abs().square()
                L += Lp.sum()
            return L

        # 使用 Torch 的 L-BFGS 优化器
        optimizer = torch.optim.LBFGS(
            [params],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-8,
            tolerance_change=1e-8,
            history_size=100,
            line_search_fn='strong_wolfe')

        # 优化循环
        best_loss = float('inf')
        best_params = params.clone()
        prev_loss = None

        def closure():
            optimizer.zero_grad()
            loss = objective(params)
            loss.backward()
            return loss

        for epoch in range(self.max_iter):
            loss = optimizer.step(closure)
            # 保存最佳参数
            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params = params.clone()
            # 检查收敛
            if epoch > 0 and prev_loss is not None and abs(prev_loss - loss.item()) < 1e-8:
                break
            prev_loss = loss.item()

        # 恢复最佳参数
        with torch.no_grad():
            params.data = best_params.data
        # 重构最终密度矩阵
        with torch.no_grad():
            final_rho = rho_from_params(params)
        # 转换为 numpy 输出
        result_rho = final_rho.cpu().numpy()
        self.reconstructed_state=DensityMatrix(result_rho)
        
        return self.reconstructed_state

class ClassicalShadowEstimation(QuantumStateTomography):
    """量子阴影估计方法 - 基于随机测量的高效态层析"""
    
    def __init__(self, n_qubits: int, 
                 measurement_type: str = "local_pauli",
                 num_shadow: int = 1000,
                 num_estimate_chunks: int = 10):
        """
        初始化量子阴影估计器
        
        参数:
            n_qubits: 量子比特数
            shots_per_measurement: 每次测量的采样次数
            measurement_type: 测量类型，可选 "global_clifford" 或 "local_pauli"
            num_snapshots: 经典阴影的数量
            num_estimate_chunks: 中位数均值估计的组数
        """
        super().__init__(n_qubits, shots_per_measurement=1)
        self.measurement_type = measurement_type
        self.num_shadow = num_shadow
        self.measure_shots = 1
        self.num_estimate_chunks = num_estimate_chunks
        self.shadow = None
        self.unitary_ensembles = None

    def _apply_random_unitary(self, circuit: QuantumCircuit) -> Tuple[QuantumCircuit, np.ndarray]:
        """
        应用随机幺正变换到电路
        
        返回:
            circuit_with_random_unitary: 应用随机幺正后的电路
            random_unitary: 应用的随机幺正矩阵
        """
        n_qubits = self.n_qubits
        
        if self.measurement_type == "global_clifford":
            # 全局Clifford随机幺正
            clifford = random_clifford(n_qubits)
            random_unitary = clifford.to_matrix()
            
            new_circuit = circuit.copy()
            new_circuit.append(clifford.to_instruction(), range(n_qubits))
            
        elif self.measurement_type == "local_pauli":
            # 局域Pauli基测量
            random_unitary = np.eye(2**n_qubits, dtype=complex)
            new_circuit = circuit.copy()
            pauli_bases = []
            
            for qubit in range(n_qubits):
                pauli_choice = np.random.choice(['X', 'Y', 'Z'])
                pauli_bases.append(pauli_choice)
                if pauli_choice == 'X':
                    new_circuit.h(qubit)
                elif pauli_choice == 'Y':
                    new_circuit.sdg(qubit)
                    new_circuit.h(qubit)
            
            # 构造对应的幺正矩阵（张量积）
            pauli_matrices = {
                'I': np.array([[1, 0], [0, 1]], dtype=complex),
                'X': np.array([[0, 1], [1, 0]], dtype=complex),
                'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
                'Z': np.array([[1, 0], [0, -1]], dtype=complex)
            }
            
            # 构建整体幺正矩阵
            random_unitary = pauli_matrices[pauli_bases[-1]]
            for i in range(n_qubits-2, -1, -1):
                random_unitary = np.kron(pauli_matrices[pauli_bases[i]], random_unitary)
                
        else:
            raise ValueError(f"不支持的测量类型: {self.measurement_type}")
            
        return new_circuit, random_unitary
    
    def _inverse_channel(self, U: np.ndarray, mea_results: str) -> np.ndarray:
        """
        应用逆量子通道 M^{-1} 到测量结果
        
        参数:
            U: 应用的随机幺正矩阵
            measurement_result: 测量结果的比特串
            
        返回:
            rho_hat: 重建的经典阴影
        """
        n_qubits = self.n_qubits
        dim = 2 ** n_qubits
        allshots=self.measure_shots
        # 构造计算基投影算符 |b⟩⟨b|
        projector=np.zeros([dim,dim], dtype=complex)
        for mea_str, count in mea_results.items():
            b_idx = int(mea_str, 2)
            ket_b = np.zeros(dim, dtype=complex)
            ket_b[b_idx] = 1
            weight=count/allshots
            projector += weight*np.outer(ket_b, ket_b.conj())
        projector/=np.trace(projector)
        # 计算 U†|b⟩⟨b|U
        U_dagger = U.conj().T
        rotated_projector = U_dagger @ projector @ U
        # 应用逆通道 M^{-1}
        rho_hat = (dim + 1) * rotated_projector - np.eye(dim)

        return rho_hat
    
    def _create_shadow(self, noise_model: Optional[NoiseModel] = None) -> List[np.ndarray]:
        """
        创建经典阴影
        
        返回:
            shadow: 经典阴影列表，每个元素是一个密度矩阵
        """
        if self.state_prep_circuit is None:
            raise ValueError("请先设置态制备电路")
            
        shadow = []
        unitary_ensembles = []
        #创建支持并行计算的模拟器
        simulator = AerSimulator(
            noise_model=noise_model,
            method='density_matrix' if noise_model is not None else 'statevector',
            device='CPU',
            # 以下是一些可选的多线程配置参数
            max_parallel_threads=0,       # 0表示使用所有可用CPU线程
            max_parallel_experiments=0,   # 0表示自动决定并行实验数
            max_parallel_shots=0,         # 0表示自动决定并行shot数
            blocking_enable=True,         # 启用块化（对大型电路有优化）
            blocking_qubits=16,           # 块化量子比特数
        )

        #多个电路一起提交执行
        shadow = []
        unitary_ensembles = []
        # 提前生成所有电路
        circuits = []
        unitaries = []
        for _ in range(self.num_shadow):
            # 1. 应用随机幺正
            circuit_with_U, random_U = self._apply_random_unitary(self.state_prep_circuit)
            # 2. 在计算基下测量
            measured_circuit = circuit_with_U.copy()
            measured_circuit.measure_all()
            
            circuits.append(measured_circuit)
            unitaries.append(random_U)
        #画电路
        #circuits[10].draw(output='mpl', filename='quantum_circuit.png')
        # 3. 一次性编译所有电路
        compiled_circuits = transpile(circuits, simulator, optimization_level=3)

        # 4. 一次性运行所有电路（并行处理）
        job = simulator.run(compiled_circuits, shots=self.measure_shots)

        # 5. 获取所有结果
        results = job.result()

        # 6. 处理结果
        for i in range(self.num_shadow):
            counts = results.get_counts(i)  # 获取第i个电路的结果
            # 应用逆通道得到经典阴影
            rho_hat = self._inverse_channel(unitaries[i], counts)
            rho_hat = rho_hat / np.trace(rho_hat)
            
            shadow.append(rho_hat)
            unitary_ensembles.append(unitaries[i])

        self.unitary_ensembles = unitary_ensembles
        return shadow
    
    # def _compute_observable_median_of_means(self, observable: np.ndarray) -> float:
    #     """
    #     使用中位数均值方法计算单个观测量的期望值
        
    #     参数:
    #         observable: 可观测量矩阵
            
    #     返回:
    #         观测量的期望值估计
    #     """
    #     if self.shadow is None:
    #         raise ValueError("请先创建经典阴影（执行态层析）")
        
    #     n_shadow = len(self.shadow)
    #     chunk_size = n_shadow // self.num_estimate_chunks
        
    #     if chunk_size == 0:
    #         raise ValueError(f"阴影数量{ n_shadow}太少，无法分成{ self.num_estimate_chunks}组")
        
    #     chunk_estimates = []
        
    #     # 将阴影分成K组
    #     for k in range(self.num_estimate_chunks):
    #         start_idx = k * chunk_size
    #         end_idx = (k + 1) * chunk_size if k < self.num_estimate_chunks - 1 else n_shadow
            
    #         # 计算该组的均值估计
    #         chunk_sum = 0
    #         for i in range(start_idx, end_idx):
    #             # 计算 tr(O * rho_hat_i)
    #             tr_value = np.trace(observable @ self.shadow[i])
    #             chunk_sum += tr_value
            
    #         chunk_mean = chunk_sum / (end_idx - start_idx)
    #         chunk_estimates.append(np.real(chunk_mean))
        
    #     # 取中位数
    #     median_estimate = np.median(chunk_estimates)
    #     return median_estimate
    
    def tomography_state(self, noise_model: Optional[NoiseModel] = None) -> DensityMatrix:
        """
        执行量子阴影态层析
        
        注意: 与传统的完全态层析不同，量子阴影返回的是平均阴影
        """
        if self.shadow is None:
            # 创建经典阴影
            self.shadow = self._create_shadow(noise_model)
        
        # 计算平均阴影
        dim = 2 ** self.n_qubits
        rho_avg = np.zeros((dim, dim), dtype=complex)
        
        for rho_hat in self.shadow:
            rho_avg += rho_hat
        
        rho_avg = rho_avg / len(self.shadow)
        
        # 投影到合法的密度矩阵空间
        eigvals, eigvecs = np.linalg.eigh(rho_avg)
        eigvals[eigvals < 0] = 0
        if np.sum(eigvals) > 0:
            eigvals = eigvals / np.sum(eigvals)
        else:
            eigvals = np.ones_like(eigvals) / dim
        
        rho_reconstructed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
        
        #临时算保真度
        fides=[]
        for shadow in self.shadow:
            fide=np.trace(shadow@self.true_state.data)
            fides.append(fide)
        self.fidelity=np.mean(fides)

        return DensityMatrix(rho_reconstructed)
    
    def predict_observable(self, observable: np.ndarray, noise_model: Optional[NoiseModel] = None) -> float:
        """
        使用中位数-均值分组估计来预测可观测量（合并自原基类）。
        """
        if self.shadow is None:
            self.shadow = self._create_shadow(noise_model)
        n_shadow = len(self.shadow)
        chunk_size = n_shadow // self.num_estimate_chunks

        if chunk_size == 0:
            raise ValueError(f"阴影数量{n_shadow}太少，无法分成{self.num_estimate_chunks}组")

        chunk_estimates = []
        for k in range(self.num_estimate_chunks):
            start_idx = k * chunk_size
            end_idx = (k + 1) * chunk_size if k < self.num_estimate_chunks - 1 else n_shadow

            chunk_shadow = np.zeros_like(self.shadow[0], dtype=complex)
            for i in range(start_idx, end_idx):
                chunk_shadow += self.shadow[i]
            chunk_shadow /= (end_idx - start_idx)

            tr_value = np.trace(observable @ chunk_shadow)
            chunk_estimates.append(np.real(tr_value))
        self.chunk_estimates = chunk_estimates
        median_estimate = np.median(chunk_estimates)

        return median_estimate

class OptimizedCSE(ClassicalShadowEstimation):
    """基于优化的IC-POVM的量子阴影估计方法，严格遵循论文1-CNOT SIC-POVM实现"""
    
    def __init__(self, n_qubits: int, 
                 num_snapshots: int = 1000,
                 num_estimate_chunks: int = 10):
        """
        初始化优化的POVM阴影估计器
        
        参数:
            n_qubits: 量子比特数
            num_snapshots: 经典阴影的数量
            num_estimate_chunks: 中位数均值估计的组数
        """
        super().__init__(n_qubits, measurement_type="povm", num_shadow=num_snapshots, num_estimate_chunks=num_estimate_chunks)
        self.num_snapshots = num_snapshots
        self.shadow = None
        self.sic_shadows = []  # 存储4个单qubit SIC阴影（对应论文4个POVM算子）
        self._define_sic_states()
    
    def _define_sic_states(self,cou=1):
        if cou==0:
            # 计算系数 a 和 b，使得布洛赫矢量为 (1/√3, 1/√3, 1/√3) 基准态 |ψ0⟩ = a|0⟩ + e^{iπ/4} b|1⟩
            a = np.sqrt((1 + 1 / np.sqrt(3)) / 2)
            b = np.sqrt((1 - 1 / np.sqrt(3)) / 2)
            psi_star = np.array([a, np.exp(1j * np.pi / 4) * b], dtype=complex)
        elif cou==1:
            # 创建单量子比特电路
            qc = QuantumCircuit(1)
            qc.rz(-np.pi/6, 0)         # Rz(-π/6)
            qc.rx(np.arccos(1/np.sqrt(3)), 0)  # Rx(arccos(1/√3))
            qc.rz(np.pi/4, 0)          # Rz(π/4)
            psi_star = Statevector(qc).data.conj() #获取末尾态矢
            self.fiducial_circuit=qc
        
        # Weyl位移算子生成4个SIC-POVM态
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # 4个SIC-POVM态
        phi0 = psi_star  # (k=0,l=0)
        phi1 = X @ psi_star  # (k=1,l=0) → X|ψ*>
        phi2 = Z @ psi_star  # (k=0,l=1) → Z|ψ*>
        phi3 = X@Z @ psi_star  # (k=1,l=1) → -Y|ψ*>
        
        self.sic_povm = [phi0, phi1, phi2, phi3]
        for sic_state in self.sic_povm:
            Pi_b = 0.5 * np.outer(sic_state, sic_state.conj())
            sic_shadow = 3 * Pi_b - np.trace(Pi_b) * np.eye(2)
            self.sic_shadows.append(sic_shadow)

        # 定义常量
        c = np.sqrt((1 + 1/np.sqrt(3)) / 2)  # cos(θ/2)
        s = np.sqrt((1 - 1/np.sqrt(3)) / 2)  # sin(θ/2)

        alpha = c * np.exp(-1j * np.pi/24)
        beta = -1j * s * np.exp(-1j * 5*np.pi/24)  # 等价于 s * exp(-1j*17π/24)
        gamma = -1j * s * np.exp(1j * 5*np.pi/24)   # 等价于 s * exp(-1j*7π/24)
        delta = c * np.exp(1j * np.pi/24)

        # 构建酉矩阵U
        self.Usic0 = (1/np.sqrt(2)) * np.array([
            [alpha, gamma, beta, delta],
            [gamma, alpha, delta, beta],
            [alpha, -gamma, beta, -delta],
            [-gamma, alpha, -delta, beta]], dtype=complex)

    def _check_sic_povm(vectors):
        """检验四个二分量态矢是否构成单比特SIC-POVM"""
        d = np.size(vectors[0])  # 希尔伯特空间维度
        d2 = d * d  # SIC-POVM元素个数

        projectors=[]
        for psi in vectors:
            projector=np.outer(psi, psi.conj())/d
            projectors.append(projector)

        if len(vectors) != d2:
            print(f"错误：需要{d2}个态矢，但输入了{len(vectors)}个")
            return False
        
        # 1. 检验归一化
        for i, psi in enumerate(vectors):
            psi = np.array(psi).flatten()
            norm = np.linalg.norm(psi)
            if abs(norm - 1.0) > 1e-10:
                print(f"态矢{i}未归一化：模长为{norm}")
                return False
        
        # 2. 检验完备性
        identity_check = np.zeros((d, d), dtype=complex)
        for projector in projectors:
            identity_check += projector  # E_i = (1/d)|ψ_i⟩⟨ψ_i|
        identity_error = np.linalg.norm(identity_check - np.eye(d))
        
        # 计算秩
        M = np.zeros((d2, d2), dtype=complex)
        for projector in projectors:
            M += np.outer(projector.flatten(), projector.flatten().conj())
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        
        # 3. 检验对称性
        target_fidelity = 1 / (d2 * (d + 1))  # 1/(4 * 3)=1/12
        max_fidelity_error = 0
        for i in range(d2):
            for j in range(i+1, d2):
                fidelity = abs(np.vdot(projectors[i].flatten(), projectors[j].flatten()))
                error = abs(fidelity - target_fidelity)
                max_fidelity_error = max(max_fidelity_error, error)
        
        # 判断标准
        tol = 1e-10
        is_complete = identity_error < tol
        is_symmetric = max_fidelity_error < tol
        is_IC = (rank_M==d2)
        
        if is_complete and is_symmetric and is_IC:
            print("✅ 这组态矢构成SIC-POVM")
            return True
        else:
            print("❌ 这组态矢不构成SIC-POVM")
            return False

    def _SIC_POVM_unitary(sic_states):
        """
        构造维度膨胀酉矩阵U，将单比特SIC-POVM映射到两比特计算基测量
        
        根据论文Eq.(5)：V = ∑_i |i⟩⟨φ_i| ⊗ √(1/2)
        其中⟨φ_i|是行向量（态矢量的共轭转置）
        """
        # 构造V矩阵：V = ∑_i |i⟩⟨φ_i| ⊗ √(1/2)
        # V是4×2矩阵，V[i,j] = √(1/2) * ⟨i|V|j⟩ = √(1/2) * ⟨φ_i|j⟩
        V = np.zeros((4, 2), dtype=complex)
        
        # 两个单比特计算基
        basis_0 = np.array([1, 0], dtype=complex)
        basis_1 = np.array([0, 1], dtype=complex)
        
        for i in range(4):
            phi_i = sic_states[i]
            # ⟨φ_i|0⟩
            V[i, 0] = np.sqrt(1/2) * np.dot(phi_i.conj(), basis_0)
            # ⟨φ_i|1⟩
            V[i, 1] = np.sqrt(1/2) * np.dot(phi_i.conj(), basis_1)
        
        # 使用QR分解构造正交补空间
        # 先生成随机矩阵，然后对[V, random_matrix]进行QR分解
        random_matrix = np.random.randn(4, 2) + 1j * np.random.randn(4, 2)
        
        # 确保random_matrix与V的列空间正交
        for i in range(2):
            # 减去在V的第i列上的投影
            proj = np.outer(V[:, i], V[:, i].conj()) @ random_matrix
            random_matrix = random_matrix - proj
        
        # 归一化
        for i in range(2):
            norm = np.linalg.norm(random_matrix[:, i])
            if norm > 1e-10:
                random_matrix[:, i] = random_matrix[:, i] / norm
        
        # 构造完整矩阵：U = [V, random_matrix]
        U = np.hstack([V, random_matrix])
        
        # 对U进行QR分解，确保严格酉性
        Q, R = np.linalg.qr(U)
        U = Q
        
        return U

    def _opti_circuit(U_SIC_target: np.ndarray, U_SIC_2: np.ndarray) -> tuple[QuantumCircuit, int]:
        """
        实现论文附录A的Algorithm 1（实用电路参数确定）。
        
        参数:
            U_SIC_target (np.ndarray): 目标酉矩阵 $U_{SIC}$，形状为 (4, 4)。
            U_SIC_2 (np.ndarray): 参考酉矩阵 $U_{SIC-2}(c=1, \Theta_2)$，形状为 (4, 4)。
            
        返回:
            U_S_circuit (QuantumCircuit): 实现 $U_S$ 的量子电路（由Rz, Ry, Rz构成）。
            c (int): 离散参数 c (0 或 1)。
        """
        # 论文Eq.(6)定义: V = [|φ₁> |φ₂> |φ₃> |φ₄>]^†
        V1 = U_SIC_target[:, :2] 
        V2 = U_SIC_2[:, :2] 

        V1_norm = V1 / np.linalg.norm(V1)
        V2_norm = V2 / np.linalg.norm(V2)

        # 构造酉矩阵 U^(i) = |φ_j^(i)><0| + |φ_j^(i)⊥><1|
        def construct_U_perp(phi_j: np.ndarray) -> np.ndarray:
            # 构造正交向量
            phi_perp = np.array([-np.conj(phi_j[1]), np.conj(phi_j[0])], dtype=complex)
            phi_perp = phi_perp / np.linalg.norm(phi_perp) # 归一化
            
            U = np.outer(phi_j, np.array([1, 0], dtype=complex)) + np.outer(phi_perp, np.array([0, 1], dtype=complex))
            return U

        # 以第一个向量为例构造 U^(1) 和 U^(2)
        U1 = construct_U_perp(V1_norm[0, :])
        U2 = construct_U_perp(V2_norm[0, :])

        # phi₂^(i) = a * e^(i*alpha1) |0> + sqrt(1-a²) * e^(i*alpha2) |1>
        phi2_1_transformed = U1 @ V1_norm[1, :]
        phi2_2_transformed = U2 @ V2_norm[1, :]

        # 提取 alpha1, alpha2 (相位) 和 a (模)
        def extract_angles_and_amplitude(phi: np.ndarray) -> tuple[float, float, float]:
            a = np.abs(phi[0]) # 假设 |0> 分量是实数
            alpha1 = np.angle(phi[0])
            alpha2 = np.angle(phi[1])
            return a, alpha1, alpha2

        a1, alpha11, alpha12 = extract_angles_and_amplitude(phi2_1_transformed)
        a2, alpha21, alpha22 = extract_angles_and_amplitude(phi2_2_transformed)

        # U_S = U^(2) * U_r * (U^(1))^†
        # U_r = |0><0| + e^(i(alpha12 - alpha22 - alpha21 + alpha11)) |1><1|
        phase_Ur = np.exp(1j * (alpha12 - alpha22 - alpha21 + alpha11))
        Ur = np.array([[1, 0], [0, phase_Ur]], dtype=complex)
        U_S = U2 @ Ur @ U1.conj().T

        # V_down^(i) = [ |φ₃> |φ₄> ]^†
        V1_down = V1_norm[2:,:] # 第3、4列 (index 2, 3)
        V2_down = V2_norm[2:,:]
        
        # U_pr = V_down^(1) * (V_down^(2) * U_S)^(-1)
        V2_down_US = V2_down @ U_S
        # 求伪逆以确保稳定性
        V2_down_US_inv = np.linalg.pinv(V2_down_US) 
        U_pr = V1_down @ V2_down_US_inv

        # 判断 U_pr 是否为对角矩阵
        def is_diagonal(mat: np.ndarray, tol: float = 1e-10) -> bool:
            mask = ~np.eye(mat.shape[0], dtype=bool)
            return np.all(np.abs(mat[mask]) < tol)

        if is_diagonal(U_pr):
            c = 1
        else:
            c = 0

        U_S, R = np.linalg.qr(U_S)
        return U_S, c
    
    def _build_SIC_POVM_circuit(self, circuit):
        """
        使用量子电路进行SIC-POVM测量（维度膨胀框架），测量部分直接用标准门名（cx、h），噪声直接作用于标准门。
        """
        total_qubits = self.n_qubits + self.n_qubits  # 系统比特 + 辅助比特（1:1对应）
        povm_circuit = QuantumCircuit(total_qubits, 2 * self.n_qubits)
        # 1. 先加载态制备电路
        povm_circuit.compose(circuit, qubits=range(self.n_qubits), inplace=True)
        for i in range(self.n_qubits):
            sys_idx = i
            aux_idx = self.n_qubits + i
            # 2. 辅助比特制备参考态
            povm_circuit.compose(self.fiducial_circuit, qubits=aux_idx, inplace=True)
            # 3. SIC-POVM测量部分
            povm_circuit.cx(aux_idx, sys_idx)
            povm_circuit.h(aux_idx)
            # 4. 测量
            povm_circuit.measure(sys_idx, 2 * i)
            povm_circuit.measure(aux_idx, 2 * i + 1)
        return povm_circuit
    
    def _measure_to_shadow(self, bitstring: str) -> np.ndarray:
        """从测量结果计算阴影（论文Eq.(25)张量积组合）"""
        # 解析比特串：每个qubit对应2个经典位，还原4个测量结果（0-3）
        #bitstring = bitstring[::-1]  # Qiskit比特序修正（低位在前）
        b_list = []
        for i in range(self.n_qubits):
            # 提取第i个qubit的2个经典位（2i和2i+1）
            if 2*i + 1 >= len(bitstring):
                two_bits = '00'  # 补零处理
            else:
                two_bits = bitstring[2*i] + bitstring[2*i+1]
            b = int(two_bits, 2)  # 0→00,1→01,2→10,3→11（对应4个SIC阴影）
            b_list.append(b)
        
        # 论文Eq.(25)：多qubit阴影=单qubit阴影的张量积
        shadow = self.sic_shadows[b_list[0]]
        for b in b_list[1:]:
            shadow = np.kron(shadow, self.sic_shadows[b])
        
        return shadow
    
    def _create_shadow(self, noise_model: Optional[NoiseModel] = None) -> List[np.ndarray]:
        """创建经典阴影（添加论文噪声模型支持）"""
        if self.state_prep_circuit is None:
            raise ValueError("请先调用set_state_prep_circuit设置态制备电路")

        simulator = AerSimulator(
            noise_model=noise_model,
            method='density_matrix' if noise_model is not None else 'statevector',
            device='CPU',
            max_parallel_threads=0
        )
        
        # 编译电路（论文要求优化级别3）
        povm_circuit = self._build_SIC_POVM_circuit(self.state_prep_circuit)
        compiled_circuit = transpile(povm_circuit, simulator, optimization_level=3)
        
        # 执行测量（memory=True获取每个shot的原始比特串）
        job = simulator.run(compiled_circuit, shots=self.num_snapshots, memory=True)
        results = job.result()
        memory = results.get_memory(0)
        
        # 生成阴影列表
        shadow = []
        for bitstring in memory:
            rho_hat = self._measure_to_shadow(bitstring)
            rho_hat = rho_hat / np.trace(rho_hat)
            shadow.append(rho_hat)
        
        return shadow

class Simu_Tomo(BasePhysicalSimulation):
        
    def __init__(self):
        super().__init__()
        self.name = "量子态扫描"
        self.description = "Quantum State Tomography"

        self.tomo_func_list=['linear','mle','cse_pauli','cse_clifford','cse_povm']
        self.show_func={
            'shadow':self.see_shadow,
            'shots-fidelity':self.shots_fidelity,
            'shots-RMSE':self.shots_RMSE,
        }
        self.state_list=['bell','ghz','random','w_state','random_entangled']
        self.obs_list=['one Z','all Z','random pauli','random_local_hamiltonian','ising_model','clustered_entanglement','shadow_friendly']

        self.define_parameters()

        self.state_fidelity=8888
        self.real_obs=8888
        self.estimate_obs=8888
        self.relat_error=8888

    def define_parameters(self):
        #初始化实验参数
        self.params = {
            'tomo_func':{'value': 'linear', 'desc': '扫描方法', 'unit': '', 
                         'type': 'combo', 'options': self.tomo_func_list},
            'show':{'value': 'shadow', 'desc': '展示', 'unit': '', 
                         'type': 'combo', 'options': self.show_func.keys()},
            'nqubit': {'value': 4, 'min': 3, 'max': 80, 'step': 1, 
                       'desc': '量子比特数', 'unit': 'a.u.', 'type': 'text'},
            'state': {'value': self.state_list[0],'desc': '量子态', 'unit': '', 
                    'type': 'combo', 'options': self.state_list},
            'obs': {'value': self.obs_list[0],'desc': '可观测量', 'unit': '', 
                    'type': 'combo', 'options': self.obs_list},
            'num shots': {'value': 20, 'min': 0, 'max': 1e20, 'step': 1, 
                         'desc': 'num shots', 'unit': 'a.u.'},
            'num shadow': {'value': 1000, 'min': 0, 'max': 1e20, 'step': 1, 
                         'desc': 'num shadow', 'unit': 'a.u.'},
            'num chunk': {'value': 10, 'min': 0, 'max': 1e20, 'step': 1,
                         'desc': 'num chunk', 'unit': 'a.u.'},
            'num plot': {'value': 10, 'min': 0, 'max': 1e20, 'step': 1,
                         'desc': '画图点数', 'unit': 'a.u.'},
            'x increase':{'value': 'linear', 'desc': '增长方式', 'unit': '', 
                         'type': 'combo', 'options': ['linear','exp']},
            'noise':{'value':
'''
noise = NoiseModel()

p1=0.001  # 单qubit门的depolarizing概率
p2=0.01   # 双qubit门的depolarizing概率
t1=50e-6  # T1时间
t2=70e-6   # T2时间
gate_time=300e-9  # 门操作时间

single_dep = depolarizing_error(p1, 1)
two_dep = depolarizing_error(p2, 2)

th_single = thermal_relaxation_error(t1, t2, gate_time)
th_two = thermal_relaxation_error(t1, t2, gate_time * 3)

single_error = single_dep.compose(th_single)
two_error = two_dep.compose(th_two)

for op in ['h', 'x', 'sx', 'rx', 'ry', 'rz']:
    noise.add_all_qubit_quantum_error(single_error, [op])
for op in ['cx', 'cz', 'swap']:
    noise.add_all_qubit_quantum_error(two_error, [op])
''',
'desc': '噪声', 'type': 'multext'},
        }

    def refresh_param(self):
        params = self.get_parameters()
        self.tomo_func=params.get('tomo_func')
        self.show=params.get('show')
        self.nqubit=int(params.get('nqubit',4))
        self.state=params.get('state')
        self.obs=params.get('obs')
        self.num_shots=int(params.get('num shots',20))
        self.num_shadow=int(params.get('num shadow',1000))
        self.num_chunk=int(params.get('num chunk',10))
        self.num_plot=int(params.get('num plot',10))
        self.x_increase=params.get('x increase')
        self.noise_json_str=params.get('noise')

    def get_derived_parameters(self):
        derived_parameters={
            'state_fidelity':{'value':self.state_fidelity, 'desc': '态保真度', 'unit': 'a.u.'},
            'real_obs':{'value':self.real_obs, 'desc': '真实测值', 'unit': 'a.u.'},
            'estimate_obs':{'value':self.estimate_obs, 'desc': '估计测值', 'unit': 'a.u.'},
            'relative_error':{'value':self.relat_error, 'desc': '相对误差', 'unit': '%'},
        }
        
        return derived_parameters

    def create_test_circuit(self, n_qubits: int, circuit_type: str = 'random_entangled') -> QuantumCircuit:
        """
        支持多种电路类型：bell、ghz、random、w_state、random_entangled
        """
        qr = QuantumCircuit(n_qubits)
        if circuit_type == 'bell' and n_qubits >= 2:
            qr.h(0)
            qr.cx(0, 1)
        elif circuit_type == 'ghz':
            qr.h(0)
            for i in range(1, n_qubits):
                qr.cx(0, i)
        elif circuit_type == 'random':
            for qubit in range(n_qubits):
                qr.ry(np.random.random() * 2 * np.pi, qubit)
                qr.rz(np.random.random() * 2 * np.pi, qubit)
        elif circuit_type == 'w_state' and n_qubits >= 3:
            qr.ry(2 * np.arccos(1 / np.sqrt(n_qubits)), 0)
            qr.x(0)
            qr.cx(0, 1)
            qr.x(0)
            for i in range(2, n_qubits):
                qr.cx(0, i)
        elif circuit_type == 'random_entangled':
            # 多比特纠缠且多基分布的随机电路
            for q in range(n_qubits):
                if np.random.rand() < 0.5:
                    qr.h(q)
                if np.random.rand() < 0.5:
                    qr.s(q)
                if np.random.rand() < 0.5:
                    qr.rx(np.random.uniform(0, 2*np.pi), q)
                if np.random.rand() < 0.5:
                    qr.ry(np.random.uniform(0, 2*np.pi), q)
                if np.random.rand() < 0.5:
                    qr.rz(np.random.uniform(0, 2*np.pi), q)
            entangle_pairs = [(i, j) for i in range(n_qubits) for j in range(i+1, n_qubits)]
            np.random.shuffle(entangle_pairs)
            for (i, j) in entangle_pairs[:max(1, n_qubits//2)]:
                gate = np.random.choice(['cx', 'cz', 'swap'])
                if gate == 'cx':
                    qr.cx(i, j)
                elif gate == 'cz':
                    qr.cz(i, j)
                elif gate == 'swap':
                    qr.swap(i, j)
            if np.random.rand() < 0.5:
                qr.global_phase = np.random.uniform(0, 2*np.pi)
        else:
            for qubit in range(n_qubits):
                qr.h(qubit)
        return qr

    @staticmethod
    def create_noise_model(noise_json_str: str) -> Optional[NoiseModel]:
        """解析 noise 字符串为 Qiskit NoiseModel。

        兼容两种输入方式：
        1) JSON 字符串（使用 NoiseModel.from_dict）
        2) Python 代码字符串（通过 exec 执行，需在代码中定义名为 `noise` 或 `noise_model` 的 NoiseModel 变量）

        运行 Python 代码时，仅向执行环境暴露有限的对象（如 NoiseModel、depolarizing_error 等），
        以避免直接污染全局命名空间。
        """
        if noise_json_str is None:
            return None

        noise_json_str = str(noise_json_str).strip()
        if noise_json_str == "":
            return None

        # 1) 尝试 JSON 解析（向后兼容）
        import json
        try:
            noise_dict = json.loads(noise_json_str)
            return NoiseModel.from_dict(noise_dict)
        except Exception:
            pass

        # 2) 尝试将 noise_json_str 当作 Python 代码执行
        #    需要在代码中定义 noise 或 noise_model 变量
        try:
            exec_env = {
                'NoiseModel': NoiseModel,
                'depolarizing_error': depolarizing_error,
                'thermal_relaxation_error': thermal_relaxation_error,
                'pauli_error': pauli_error,
                'ReadoutError': ReadoutError,
                'reset_error': reset_error,
                'np': np,
            }
            # 执行用户代码
            exec(noise_json_str, exec_env, exec_env)

            # 优先寻找常用变量名
            if isinstance(exec_env.get('noise'), NoiseModel):
                return exec_env['noise']
            if isinstance(exec_env.get('noise_model'), NoiseModel):
                return exec_env['noise_model']

            # 若执行环境里只有一个 NoiseModel 实例，则返回它
            for v in exec_env.values():
                if isinstance(v, NoiseModel):
                    return v

            print("未找到 NoiseModel 实例（变量名 'noise' 或 'noise_model'），请在代码中定义它。")
            return None
        except Exception as e:
            print(f"噪声字符串处理失败: {e}")
            return None

    def create_obs(self, obstype, n_qubits, **kwargs):
        pauli = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        if obstype == 'one Z':
            ops = ['I'] * n_qubits
            ops[np.random.randint(0, n_qubits)] = 'Z'
            observable = pauli[ops[0]]
            for op in ops[1:]:
                observable = np.kron(observable, pauli[op])
        elif obstype == 'all Z':
            observable = pauli['Z']
            for _ in range(1, n_qubits):
                observable = np.kron(observable, pauli['Z'])
        elif obstype == 'random pauli':
            ops = np.random.choice(['X', 'Y', 'Z'], size=n_qubits)
            observable = pauli[ops[0]]
            for op in ops[1:]:
                observable = np.kron(observable, pauli[op])
        
        elif obstype == 'random_local_hamiltonian':
            density = kwargs.get('density', 0.5)
            strength_range = kwargs.get('strength_range', (-1.0, 1.0))
            k_local = kwargs.get('k_local', 2)
            observable = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
            # 1-local
            for i in range(n_qubits):
                if np.random.random() < density:
                    op = np.random.choice(['X','Y','Z'])
                    term = [op if j==i else 'I' for j in range(n_qubits)]
                    mat = pauli[term[0]]
                    for t in term[1:]:
                        mat = np.kron(mat, pauli[t])
                    observable += np.random.uniform(*strength_range) * mat
            # 2-local
            if k_local>=2 and n_qubits>1:
                for i in range(n_qubits-1):
                    if np.random.random()<density:
                        ops = [np.random.choice(['I','X','Y','Z']) for _ in range(2)]
                        if ops[0]=='I' and ops[1]=='I': continue
                        term = [ops[0] if j==i else (ops[1] if j==i+1 else 'I') for j in range(n_qubits)]
                        mat = pauli[term[0]]
                        for t in term[1:]:
                            mat = np.kron(mat, pauli[t])
                        observable += np.random.uniform(*strength_range) * mat
            # 3-local
            if k_local>=3 and n_qubits>2:
                for i in range(n_qubits-2):
                    if np.random.random()<density:
                        ops = [np.random.choice(['I','X','Y','Z']) for _ in range(3)]
                        if all(o=='I' for o in ops): continue
                        term = [ops[0] if j==i else (ops[1] if j==i+1 else (ops[2] if j==i+2 else 'I')) for j in range(n_qubits)]
                        mat = pauli[term[0]]
                        for t in term[1:]:
                            mat = np.kron(mat, pauli[t])
                        observable += np.random.uniform(*strength_range) * mat
            observable = (observable + observable.conj().T) / 2
        
        elif obstype == 'ising_model':
            has_transverse_field = kwargs.get('has_transverse_field', True)
            has_longitudinal_field = kwargs.get('has_longitudinal_field', True)
            interaction_strength = kwargs.get('interaction_strength', 1.0)
            observable = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
            # 纵向场
            if has_longitudinal_field:
                for i in range(n_qubits):
                    term = ['Z' if j==i else 'I' for j in range(n_qubits)]
                    mat = pauli[term[0]]
                    for t in term[1:]:
                        mat = np.kron(mat, pauli[t])
                    observable += np.random.normal(0, 1) * mat
            # 横向场
            if has_transverse_field:
                for i in range(n_qubits):
                    term = ['X' if j==i else 'I' for j in range(n_qubits)]
                    mat = pauli[term[0]]
                    for t in term[1:]:
                        mat = np.kron(mat, pauli[t])
                    observable += np.random.normal(0, 0.5) * mat
            # 最近邻相互作用
            for i in range(n_qubits-1):
                term = ['Z' if (j==i or j==i+1) else 'I' for j in range(n_qubits)]
                mat = pauli[term[0]]
                for t in term[1:]:
                    mat = np.kron(mat, pauli[t])
                observable += np.random.normal(0, interaction_strength) * mat
            observable = (observable + observable.conj().T) / 2
        
        elif obstype == 'clustered_entanglement':
            n_terms = kwargs.get('n_terms', min(10, 2**n_qubits))
            max_interaction_range = kwargs.get('max_interaction_range', min(3, n_qubits))
            observable = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
            for _ in range(n_terms):
                k = np.random.randint(2, max_interaction_range+1)
                qubits = np.random.choice(n_qubits, size=min(k, n_qubits), replace=False)
                paulis = np.random.choice(['X','Y','Z'], size=len(qubits))
                if all(p=='I' for p in paulis): continue
                op_list = ['I']*n_qubits
                for idx, q in enumerate(qubits):
                    op_list[q] = paulis[idx]
                mat = pauli[op_list[0]]
                for t in op_list[1:]:
                    mat = np.kron(mat, pauli[t])
                observable += np.random.normal(0, 1)/np.sqrt(len(qubits)) * mat
            if np.random.random()<0.3:
                global_pauli = np.random.choice(['X','Y','Z'])
                mat = pauli[global_pauli]
                for _ in range(1, n_qubits):
                    mat = np.kron(mat, pauli[global_pauli])
                observable += np.random.normal(0, 0.5) * mat
            observable = (observable + observable.conj().T) / 2
        
        elif obstype == 'shadow_friendly':
            density = kwargs.get('density', 0.7)
            correlation_strength = kwargs.get('correlation_strength', 0.5)
            terms = []
            for i in range(n_qubits):
                if np.random.random()<density:
                    pauli_op = np.random.choice(['X','Y','Z'])
                    terms.append(([i],[pauli_op],np.random.normal(0,1)))
            for i in range(n_qubits):
                for j in range(i+1,n_qubits):
                    if np.random.random()<density*correlation_strength:
                        interaction_type = np.random.choice(['ZZ','XX','YY','XZ','ZX','XY','YX','YZ','ZY'])
                        pauli1,pauli2 = interaction_type[0],interaction_type[1]
                        terms.append(([i,j],[pauli1,pauli2],np.random.normal(0,0.5)))
            if n_qubits>=3:
                for _ in range(max(1,n_qubits//2)):
                    qubits = np.random.choice(n_qubits, size=3, replace=False)
                    paulis = np.random.choice(['X','Y','Z'], size=3)
                    terms.append((qubits, paulis, np.random.normal(0,0.3)))
            observable = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
            for qubits, paulis, coeff in terms:
                op_list = ['I']*n_qubits
                for q,p in zip(qubits,paulis):
                    op_list[q]=p
                mat = pauli[op_list[0]]
                for t in op_list[1:]:
                    mat = np.kron(mat, pauli[t])
                observable += coeff * mat
            trace = np.trace(observable)
            identity = np.eye(2**n_qubits, dtype=complex)
            observable = observable - (trace/(2**n_qubits))*identity
            fro_norm = np.linalg.norm(observable, 'fro')
            if fro_norm>0:
                observable = observable/fro_norm
        
        else:
            raise ValueError(f"Unknown observable type: {obstype}")
        
        return observable

    #def cul_fidelity(self,est_rho,target_rho):

    def run_simulation(self, **kwargs) -> Dict:
        """运行完整的态层析实验"""
        self.refresh_param()

        #换为本地变量访问，防止线程崩溃
        show_func = self.show_func.copy()  # 复制字典
        show_key = str(self.show)  # 复制为字符串
        if show_key in show_func:
            func = show_func[show_key]
            x, y, plot_type = func()

        self.update_derived_params()
        label = f"{self.name} ({self.show})"
        dg = DataGroup(1, label, x, y, plot_type=plot_type)
        return [[dg]]

    def see_shadow(self):
        circuit = self.create_test_circuit(self.nqubit, self.state)
        # 2. 选择态层析方法
        if self.tomo_func == 'linear':
            tomographer = LinearInversionTomography(self.nqubit, self.num_shots)
        elif self.tomo_func == 'mle':
            tomographer = MaximumLikelihoodTomography(self.nqubit, self.num_shots)
        elif self.tomo_func == 'cse_pauli':
            tomographer = ClassicalShadowEstimation(
                n_qubits=self.nqubit,
                measurement_type="local_pauli",
                num_shadow=self.num_shadow,
                num_estimate_chunks=self.num_chunk)
        elif self.tomo_func == 'cse_clifford':
            tomographer = ClassicalShadowEstimation(
                n_qubits=self.nqubit,
                measurement_type="global_clifford",
                num_shadow=self.num_shadow,
                num_estimate_chunks=self.num_chunk)
        elif self.tomo_func == 'cse_povm':
            tomographer = OptimizedCSE(
                n_qubits=self.nqubit,
                num_snapshots=self.num_shots,
                num_estimate_chunks=self.num_chunk)
        else:
            raise ValueError(f"未知的态层析方法: {self.tomo_func}")

        # 3. 设置态制备电路
        prep_circuit = circuit.copy()
        true_state = tomographer.set_state_preparation(prep_circuit)

        # 4. 解析噪声参数并构建噪声模型
        noise_model = None
        if self.noise_json_str!={}:
            noise_model = self.create_noise_model(self.noise_json_str)

        # 5. 执行态层析
        reconstructed_state = tomographer.tomography_state(noise_model)

        # 6. 评估性能
        self.state_fidelity = state_fidelity(true_state, reconstructed_state)
        observable = self.create_obs(self.obs, self.nqubit)
        self.estimate_obs = tomographer.predict_observable(observable)
        self.real_obs = np.real(np.trace(observable @ true_state.data))
        observable_error = abs(self.real_obs - self.estimate_obs)
        if abs(self.real_obs) > 1e-10:
            self.relat_error = observable_error / abs(self.real_obs) * 100
        else:
            self.relat_error = observable_error
        y_data = tomographer.chunk_estimates
        x_data = list(range(1, self.num_chunk + 1))
        return x_data, y_data, '直方图'

    def see_shots(self):
        circuit = self.create_test_circuit(self.nqubit,self.state)
        prep_circuit = circuit.copy()

        # 4. 解析噪声参数并构建噪声模型
        noise_model = None
        if self.noise_json_str!='':
            noise_model = self.create_noise_model(self.noise_json_str)
        
        x_shadow=['cse_pauli','cse_clifford','cse_povm']
        if self.tomo_func in x_shadow:
            if self.x_increase=='linear':
                x_list=np.linspace(self.num_shadow/self.num_plot, self.num_shadow, self.num_plot).astype(int)# 等差数列
            elif self.x_increase=='exp':
                x_list=np.exp(np.linspace(1, np.log(self.num_shadow), self.num_plot)).astype(int)# 等比数列
        else:
            if self.x_increase=='linear':
                x_list=np.linspace(self.num_shots/self.num_plot, self.num_shots, self.num_plot).astype(int)# 等差数列
            elif self.x_increase=='exp':
                x_list=np.exp(np.linspace(1, np.log(self.num_shots), self.num_plot)).astype(int)# 等比数列
        
        fidelity=[]
        rmse=[]
        for x_f in x_list:
            x=int(x_f)
            if self.tomo_func == 'linear':
                tomographer = LinearInversionTomography(self.nqubit, x)
            elif self.tomo_func == 'mle':
                tomographer = MaximumLikelihoodTomography(self.nqubit, x)
            elif self.tomo_func == 'cse_pauli':
                tomographer = ClassicalShadowEstimation(
                    n_qubits=self.nqubit,
                    measurement_type="local_pauli",
                    num_shadow=x,
                    num_estimate_chunks=self.num_chunk)
            elif self.tomo_func == 'cse_clifford':
                tomographer = ClassicalShadowEstimation(
                    n_qubits=self.nqubit,
                    measurement_type="global_clifford",
                    num_shadow=x,
                    num_estimate_chunks=self.num_chunk)
            elif self.tomo_func == 'cse_povm':
                tomographer = OptimizedCSE(
                    n_qubits=self.nqubit,
                    num_snapshots=x,
                    num_estimate_chunks=self.num_chunk)
            else:
                raise ValueError(f"未知的态层析方法: {self.tomo_func}")
            
            true_state=tomographer.set_state_preparation(prep_circuit)
            self.estimate_obs = tomographer.predict_observable(true_state.data,noise_model)
            fidelity.append(self.estimate_obs)

            reconstructed_state = tomographer.tomography_state(noise_model)
            # self.state_fidelity = tomographer.fidelity#state_fidelity(true_state, reconstructed_state)
            # fidelity.append(self.state_fidelity)

            # 计算矩阵差的Frobenius范数的平方，然后除以元素总数
            diff = true_state.data - reconstructed_state.data
            mse = np.sum(np.abs(diff)) / diff.size
            rmse.append(mse)#np.sqrt(mse))
        return x_list,fidelity,rmse
    
    def shots_RMSE(self):
        x_list,fidelity,rmse=self.see_shots()
        return x_list,rmse,"线图"

    def shots_fidelity(self):
        x_list,fidelity,rmse=self.see_shots()
        return x_list,fidelity,"线图"
