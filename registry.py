from src.PhysicalSimu import WaveSimulation,QuantumWellSimulation,dumpcos
from src.RSC import RSCsimu
from src.QuanStateTomo import Simu_Tomo

registry={
    "波动模拟": WaveSimulation(),
    "量子阱模拟": QuantumWellSimulation(),
    "RSC模拟":RSCsimu(),
    "dumpcos":dumpcos(),
    "量子态扫描":Simu_Tomo(),
}