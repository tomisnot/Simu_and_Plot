import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QDoubleValidator

class DataGroup:
    """数据组类，管理一组数据及其绘图属性"""
    def __init__(self, data_id, label, x_data, y_data, color=None, plot_type="线图", visible=True, plot_props=None):
        self.data_id = data_id
        self.label = label
        self.x_data = x_data
        self.y_data = y_data
        self.color = color
        self.plot_type = plot_type  # 存储绘图类型
        self.visible = visible
        self.plot_object = None  # 存储matplotlib绘图对象
        # plot_props 存放与 plot_type 相关的作图参数，例如线宽、marker、bins 等
        if plot_props is None:
            self.plot_props = self._default_props_for(plot_type)
        else:
            self.plot_props = plot_props

    def _default_props_for(self, plot_type):
        if plot_type == "线图":
            return {"linewidth": 2.0, "linestyle": "-", "marker": None}
        if plot_type == "散点图":
            return {"s": 20, "marker": "o"}
        if plot_type == "直方图":
            return {"bins": 30, "alpha": 0.7}
        if plot_type == "误差图":
            return {"capsize": 3, "fmt": "o-"}
        if plot_type == "热图":
            return {"cmap": "RdBu_r", "origin": "lower", "aspect": "auto"}
        return {}


class BasePhysicalSimulation(QObject):
    """物理模拟基类，所有具体模拟都应继承此类"""
    progress=pyqtSignal(dict)  # 定义进度更新信号，参数为进度描述字典
    def __init__(self):
        super().__init__()
        self.name = "基础模拟"
        self.description = "物理模拟基类"
        self.params = {}  # 参数定义字典
        self.param_widgets = {}  # 参数控件字典
        self.derived_widgets = {}  # 派生参数控件字典
        self.simu_fun = {}
        self.results = {}  # 存储模拟结果

    @staticmethod
    def format_scientific(value, precision=6):
        """将数值格式化为科学记数法字符串"""
        if value == 0:
            return "0.0"
        
        # 使用'g'格式，它会自动选择固定点或科学记数法
        return f"{value:.{precision}g}"
    
    @staticmethod
    def parse_scientific(text):
        """解析科学记数法字符串为浮点数"""
        try:
            # 先尝试直接解析
            return float(text)
        except ValueError:
            # 尝试解析各种科学记数法格式
            text = text.strip()
            
            # 替换常见的科学记数法表示
            text = text.replace('×10^', 'e').replace('×10⁻', 'e-').replace('×10', 'e')
            text = text.replace('E', 'e').replace('−', '-')
            text = text.replace('e+', 'e').replace('e-', 'e-')
            
            # 处理无符号指数的情况（如1.23e9）
            if 'e' in text and text[-1] != 'e' and text[-2] != 'e':
                try:
                    return float(text)
                except ValueError:
                    pass
            
            return 0.0
        
    def define_parameters(self):
        """定义模拟参数，子类必须重写此方法。支持五种类型：'type': 'scientific'|'combo'|'text'|'plaintext'|'multext'|'funcbtn'"""
        # 参数格式: {'param_name': {'value': 默认值, 'type': 类型, ...}}
        self.params = {
            'amplitude': {'value': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1,
                         'desc': '振幅', 'unit': 'a.u.', 'type': 'scientific'},
            'frequency': {'value': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1,
                         'desc': '频率', 'unit': 'Hz', 'type': 'scientific'},
            'mode': {'value': '正弦波', 'desc': '模式', 'unit': '', 'type': 'combo', 'options': ['正弦波', '阻尼振荡', '调制波']},
            'comment': {'value': '', 'desc': '备注', 'unit': '', 'type': 'text'},
        }
        
    def get_derived_parameters(self):
        """获取派生参数，子类可以重写此方法"""
        # 返回格式: {'param_name': {'value': 值, 'desc': '描述', 'unit': '单位'}}
        return {}
        
    def create_param_widgets(self, parent_widget, layout):
        """创建参数输入控件，multext类型为按钮弹窗"""
        self.param_widgets.clear()
        self.derived_widgets.clear()

        params_container = QWidget()
        params_layout = QHBoxLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)

        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)

        for param_name, param_info in self.params.items():
            param_widget = QWidget()
            param_layout = QHBoxLayout(param_widget)
            param_layout.setContentsMargins(0, 0, 0, 0)

            label_text = f"{param_info['desc']}:"
            label = QLabel(label_text)
            label.setFixedWidth(80)
            param_layout.addWidget(label)

            ptype = param_info.get('type', 'scientific')
            if ptype == 'scientific':
                line_edit = QLineEdit()
                line_edit.setFixedWidth(100)
                default_value = param_info['value']
                formatted_value = self.format_scientific(default_value)
                line_edit.setText(formatted_value)
                validator = QDoubleValidator(param_info['min'], param_info['max'], 5)
                validator.setNotation(QDoubleValidator.ScientificNotation)
                line_edit.setValidator(validator)
                line_edit.editingFinished.connect(self.update_derived_params)
                line_edit.param_name = param_name
                self.param_widgets[param_name] = line_edit
                param_layout.addWidget(line_edit)
            elif ptype == 'combo':
                combo = QComboBox()
                combo.setFixedWidth(100)
                options = param_info.get('options', [])
                combo.addItems(options)
                combo.setCurrentText(str(param_info['value']))
                combo.currentTextChanged.connect(self.update_derived_params)
                combo.param_name = param_name
                self.param_widgets[param_name] = combo
                param_layout.addWidget(combo)
            elif ptype == 'text':
                text_edit = QLineEdit()
                text_edit.setFixedWidth(100)
                text_edit.setText(str(param_info['value']))
                text_edit.editingFinished.connect(self.update_derived_params)
                text_edit.param_name = param_name
                self.param_widgets[param_name] = text_edit
                param_layout.addWidget(text_edit)
            elif ptype == 'plaintext':
                plainText = QPlainTextEdit()
                plainText.setFixedWidth(200)
                plainText.setPlainText(str(param_info['value']))
                plainText.param_name = param_name
                self.param_widgets[param_name] = plainText
                param_layout.addWidget(plainText)
            elif ptype == 'multext':
                btn = QPushButton("编辑/查看…")
                btn.setFixedWidth(100)
                btn.param_name = param_name
                btn.clicked.connect(lambda checked, pn=param_name: self._show_multext_dialog(pn))
                self.param_widgets[param_name] = btn
                param_layout.addWidget(btn)
            elif ptype == 'funcbtn':
                btn = QPushButton(param_info.get('desc', 'Button'))
                btn.setFixedWidth(100)
                btn.clicked.connect(param_info['value'])
                param_layout.addWidget(btn)

            if 'unit' in param_info:
                unit_label = QLabel(param_info['unit'])
                unit_label.setFixedWidth(40)
                param_layout.addWidget(unit_label)

            param_layout.addStretch()
            input_layout.addWidget(param_widget)

        input_layout.addStretch()
        params_layout.addWidget(input_widget)

        derived_widget = QWidget()
        derived_layout = QVBoxLayout(derived_widget)
        derived_layout.setContentsMargins(10, 0, 0, 0)

        derived_title = QLabel("派生参数")
        derived_title.setStyleSheet("font-weight: bold; color: #2196F3;")
        derived_layout.addWidget(derived_title)

        derived_params = self.get_derived_parameters()
        for param_name, param_info in derived_params.items():
            derived_param_widget = QWidget()
            derived_param_layout = QHBoxLayout(derived_param_widget)
            derived_param_layout.setContentsMargins(0, 0, 0, 0)

            label_text = f"{param_info['desc']}:"
            label = QLabel(label_text)
            label.setFixedWidth(80)
            derived_param_layout.addWidget(label)

            value_label = QLabel("")
            value_label.setFixedWidth(120)
            value_label.setStyleSheet("background-color: #f0f0f0; padding: 2px;")
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self.derived_widgets[param_name] = value_label
            derived_param_layout.addWidget(value_label)

            if 'unit' in param_info:
                unit_label = QLabel(param_info['unit'])
                unit_label.setFixedWidth(40)
                derived_param_layout.addWidget(unit_label)

            derived_param_layout.addStretch()
            derived_layout.addWidget(derived_param_widget)

        derived_layout.addStretch()
        params_layout.addWidget(derived_widget)

        layout.addWidget(params_container)

        self.update_derived_params()

    def _show_multext_dialog(self, param_name):
        """弹出多行文本编辑对话框"""
        dlg = QDialog()
        dlg.setWindowTitle(self.params[param_name].get('desc', param_name))
        dlg.resize(500, 300)
        vlayout = QVBoxLayout(dlg)
        textedit = QTextEdit()
        textedit.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 18px;
                border: 1px solid #cccccc;
            }
        """)
        textedit.setPlainText(str(self.params[param_name]['value']))
        vlayout.addWidget(textedit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        vlayout.addWidget(buttons)
        def accept():
            self.params[param_name]['value'] = textedit.toPlainText()
            dlg.accept()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dlg.reject)
        dlg.exec_()
    
    def update_derived_params(self):
        """更新派生参数显示"""
        derived_params = self.get_derived_parameters()
        for param_name, param_info in derived_params.items():
            if param_name in self.derived_widgets:
                value = param_info.get('value', 0.0)
                formatted_value = self.format_scientific(value)
                self.derived_widgets[param_name].setText(formatted_value)
    
    def get_parameters(self):
        """从控件获取参数值，科学计数法为float，combo和text为str，multext为多行文本（取自params字典）"""
        params_dict = {}
        for param_name, widget in self.param_widgets.items():
            param_info = self.params[param_name]
            ptype = param_info.get('type', 'scientific')
            if ptype == 'scientific':
                try:
                    value = self.parse_scientific(widget.text())
                    value = max(param_info['min'], min(param_info['max'], value))
                    formatted = self.format_scientific(value)
                    widget.setText(formatted)
                    params_dict[param_name] = value
                except ValueError:
                    params_dict[param_name] = param_info['value']
                    formatted = self.format_scientific(param_info['value'])
                    widget.setText(formatted)
            elif ptype == 'combo':
                params_dict[param_name] = widget.currentText()
            elif ptype == 'text':
                params_dict[param_name] = widget.text()
            elif ptype == 'plaintext':
                params_dict[param_name] = widget.toPlainText()
            elif ptype == 'multext':
                # 取自params字典
                params_dict[param_name] = param_info['value']
        self.update_derived_params()
        return params_dict
    
    def set_parameters(self, params_dict):
        """设置参数值，兼容multext类型（只更新params字典，按钮不需更新）"""
        for param_name, value in params_dict.items():
            if param_name in self.params and param_name in self.param_widgets:
                ptype = self.params[param_name].get('type', 'scientific')
                if ptype == 'scientific':
                    self.params[param_name]['value'] = value
                    formatted = self.format_scientific(value)
                    self.param_widgets[param_name].setText(formatted)
                elif ptype == 'combo':
                    self.params[param_name]['value'] = value
                    self.param_widgets[param_name].setCurrentText(str(value))
                elif ptype == 'text':
                    self.params[param_name]['value'] = value
                    self.param_widgets[param_name].setText(str(value))
                elif ptype == 'plaintext':
                    self.params[param_name]['value'] = value
                    self.param_widgets[param_name].setPlainText(str(value))
                elif ptype == 'multext':
                    self.params[param_name]['value'] = value
                    # 按钮无需更新内容
        self.update_derived_params()
    
    def run_simulation(self, **kwargs):
        """运行模拟，子类必须重写此方法。

        返回值应为 List[List[DataGroup]]。
        外层列表按子图顺序排列，内层列表表示同一子图内的多个数据组。
        """
        # 这里应该包含实际的物理模拟代码。
        x = np.linspace(0, 4*np.pi, 1000)
        y = np.sin(x)
        dg = DataGroup(1, self.name, x, y, color=(0.0, 0.0, 1.0, 1.0), plot_type="线图")
        return [[dg]]
    
class WaveSimulation(BasePhysicalSimulation):
    """波动模拟示例"""
    
    def __init__(self):
        super().__init__()
        self.name = "波动模拟"
        self.description = "模拟各种波动现象"
        self.define_parameters()
        
    def define_parameters(self):
        """定义波动模拟的参数"""
        self.params = {
            'amplitude': {'value': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1, 
                         'desc': '振幅', 'unit': 'a.u.'},
            'frequency': {'value': 1.0, 'min': 0.1, 'max': 10.0, 'step': 0.1, 
                         'desc': '频率', 'unit': 'Hz'},
            'phase': {'value': 0.0, 'min': 0.0, 'max': 2*np.pi, 'step': 0.1, 
                     'desc': '相位', 'unit': 'rad'},
            'damping': {'value': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01, 
                       'desc': '阻尼系数', 'unit': '1/s'},
            'wave_type': {'value': 0, 'min': 0, 'max': 2, 'step': 1, 
                         'desc': '波形类型', 'unit': ''},
        }
    
    def run_simulation(self, n_points=1000, **kwargs):
        """运行波动模拟"""
        params = self.get_parameters()
        
        A = params.get('amplitude', 1.0)
        f = params.get('frequency', 1.0)
        phi = params.get('phase', 0.0)
        gamma = params.get('damping', 0.1)
        wave_type = int(params.get('wave_type', 0))
        
        x = np.linspace(0, 4*np.pi, n_points)
        
        if wave_type == 0:  # 正弦波
            y = A * np.sin(2*np.pi*f*x + phi)
        elif wave_type == 1:  # 阻尼振荡
            y = A * np.exp(-gamma*x) * np.sin(2*np.pi*f*x + phi)
        else:  # 调制波
            y = A * np.sin(2*np.pi*f*x + phi) * np.cos(0.5*x)
        
        dg = DataGroup(1, self.name, x, y, color=(0.0, 0.0, 1.0, 1.0), plot_type="线图")
        return [[dg]]

class QuantumWellSimulation(BasePhysicalSimulation):
    """量子阱模拟示例"""
    
    def __init__(self):
        super().__init__()
        self.name = "量子阱模拟"
        self.description = "模拟一维无限深方势阱中的粒子"
        self.define_parameters()
        
    def define_parameters(self):
        """定义量子阱模拟的参数"""
        self.params = {
            'well_width': {'value': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1, 
                          'desc': '势阱宽度', 'unit': 'nm'},
            'energy_level': {'value': 1.0, 'min': 1.0, 'max': 10.0, 'step': 1.0, 
                           'desc': '能级', 'unit': 'n'},
            'mass': {'value': 9.11e-31, 'min': 1e-31, 'max': 1e-29, 'step': 1e-31, 
                    'desc': '粒子质量', 'unit': 'kg'},
            'normalize': {'value': 1.0, 'min': 0.0, 'max': 1.0, 'step': 1.0, 
                         'desc': '归一化', 'unit': ''},
        }
    
    def run_simulation(self, n_points=1000, **kwargs):
        """运行量子阱模拟"""
        params = self.get_parameters()
        
        L = params.get('well_width', 1.0)  # 势阱宽度
        n = params.get('energy_level', 1.0)  # 能级
        m = params.get('mass', 9.11e-31)  # 粒子质量
        hbar = 1.0545718e-34  # 约化普朗克常数
        
        x = np.linspace(0, L, n_points)
        
        # 一维无限深方势阱的波函数
        psi = np.sqrt(2/L) * np.sin(n*np.pi*x/L)
        
        # 能量本征值
        E = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
        
        # 概率密度
        prob_density = np.abs(psi)**2
        
        dg = DataGroup(1, self.name, x, prob_density, color=(0.0, 0.5, 0.0, 1.0), plot_type="线图")
        return [[dg]]

class dumpcos(BasePhysicalSimulation):
    def __init__(self):
        super().__init__()
        self.name = "dumpcos"
        self.description = "dumpcos"
        self.define_parameters()
    
    def define_parameters(self):
        #初始化实验参数
        self.params = {
            'time_steps': {'value': 2000, 'min': 1, 'max': 1e20, 'step': 1, 
                         'desc': 'time steps', 'unit': 'a.u.'},
            't_end_factor': {'value': 20, 'min': 0.0, 'max': 1e20, 'step': 0.1, 
                         'desc': 't_end factor', 'unit': 'a.u.'},
            'A': {'value': 0.253, 'min': 0.0, 'max': 1e20, 'step': 0.1, 
                         'desc': 'A', 'unit': 'a.u.'},
            'tao': {'value': 8.149, 'min': 0.0, 'max': 1e20, 'step': 0.1, 
                         'desc': 'tao', 'unit': 'us'},
            'Rabi': {'value': 61.9e3, 'min': 0.0, 'max': 1e20, 'step': 0.1, 
                         'desc': 'Rabi', 'unit': 'Hz'},
            'C': {'value': 0.364, 'min': -1e20, 'max': 1e20, 'step': 0.1, 
                         'desc': 'C', 'unit': 'a.u.'},}
        
    def run_simulation(self, **kwargs):
        params = self.get_parameters()
        time_steps=int(params.get('time_steps', 1.0))
        t_end_factor=params.get('t_end_factor', 1.0)
        A = params.get('A', 1.0)
        tao = params.get('tao', 1.0)*1e-6
        Rabi = params.get('Rabi', 1.0)
        C=params.get('C', 1.0)
        T_pi = np.pi / Rabi
        x=np.linspace(0, t_end_factor*T_pi, time_steps)
        y=A*np.exp(-x/tao)*np.cos(Rabi*x+np.pi)+C
        x=x*1e6
        dg = DataGroup(1, self.name, x, y, color=(0.5, 0.0, 0.5, 1.0), plot_type="线图")
        return [[dg]]
