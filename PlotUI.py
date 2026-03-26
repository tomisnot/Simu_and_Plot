import sys
import os
import json
from datetime import timedelta
import pickle
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QColor

from src.UIWidget import SubplotPropertiesDialog, DataGroupWidget

import matplotlib
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.use('Qt5Agg')
matplotlib.rcParams.update({
    'font.size': 20,               # 全局基础字体大小
    'axes.titlesize': 24,          # 坐标轴标题大小
    'axes.labelsize': 20,          # 坐标轴标签大小
    'xtick.labelsize': 16,         # x轴刻度标签大小
    'ytick.labelsize': 16,         # y轴刻度标签大小
    'legend.fontsize': 16,         # 图例字体大小
    'figure.titlesize': 28,        # 图标题大小
    'figure.labelsize': 16,        # 图标签大小
})

class SubplotDataManager:
    """子图数据管理器，管理一个子图的所有数据组"""
    _global_color_index = 0  # 添加全局颜色索引
    
    def __init__(self, subplot_idx):
        self.subplot_idx = subplot_idx
        self.data_groups = {}  # {data_id: DataGroup}
        self.next_data_id = 1
        # 子图属性：可由用户编辑并在重绘时应用
        self.properties = {
            'title':f"子图 {subplot_idx+1}",'xlabel':'X','ylabel':'Y',
            'xlim':None,'ylim':None,'xlog':False,'ylog':False,'grid':True,
        }
        
    def add_datagroup(self, data_group):
        """添加新数据组"""
        data_id = self.next_data_id
        data_group.data_id = data_id
        self.next_data_id += 1

        # 如果外部指定颜色/plot_props，优先使用，否则从全局colormap分配
        if data_group.color is None:
            cmap = cm.get_cmap('tab20')
            n_colors = 20
            data_group.color = cmap(SubplotDataManager._global_color_index % n_colors)
            SubplotDataManager._global_color_index += 1

        self.data_groups[data_id] = data_group
        return data_id
    
    def remove_data(self, data_id):
        """删除数据组"""
        if data_id in self.data_groups:
            del self.data_groups[data_id]
            return True
        return False
    
    def toggle_visibility(self, data_id):
        """切换数据组的可见性"""
        if data_id in self.data_groups:
            data_group = self.data_groups[data_id]
            data_group.visible = not data_group.visible
            return True, data_group.visible
        return False, False
    
    def get_data_count(self):
        """获取数据组数量"""
        return len(self.data_groups)
    
    def get_visible_data_count(self):
        """获取可见数据组数量"""
        return sum(1 for dg in self.data_groups.values() if dg.visible)
    
    def get_all_data(self):
        """获取所有数据组"""
        return list(self.data_groups.values())
    
    def clear_all(self):
        """清空所有数据组"""
        self.data_groups.clear()
        self.next_data_id = 1

    def set_properties(self, props):
        for key in props:
            if key in self.properties:
                self.properties[key] = props[key]
        
class SimulationThread(QThread):
    """模拟线程"""
    result = pyqtSignal(object)
    progress = pyqtSignal(str)
    
    def __init__(self, simulation_obj):
        super().__init__()
        self.simulation = simulation_obj
        
    def run(self):
        """运行模拟计算"""
        # 发出开始状态
        self.progress.emit(f"开始{self.simulation.name}...")

        try:
            # 调用模拟对象的run_simulation，期望返回 DataGroup 列表
            data_groups = self.simulation.run_simulation()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            self.progress.emit(f"模拟失败: {str(e)}")
            # 发射空结果表示失败
            self.result.emit([])
            return

        self.progress.emit("模拟完成")
        self.result.emit(data_groups)

class Plotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.simulations = {}  # 存储可用的模拟对象
        self.current_simulation = None  # 当前选中的模拟对象
        self.axes = []  # 存储子图对象
        self.current_ax_idx = 0  # 当前选中的子图索引
        self.data_managers = {}  # 存储每个子图的数据管理器
        self.simulation_thread = None
        self.data_group_widgets = {}  # 存储数据组控件
        self.register_simulations()
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('物理仿真参数调节与可视化')
        self.setGeometry(100, 100, 1400, 800)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = QWidget()
        self.control_layout = QVBoxLayout(control_panel)
        self.control_layout.setAlignment(Qt.AlignTop)
        
        # 模拟选择区域
        self.setup_simulation_selector(self.control_layout)
        
        # 绘图控制区域
        self.setup_plot_controls(self.control_layout)
        
        # 子图导航区域
        self.setup_subplot_navigation(self.control_layout)
        
        # 参数输入区域（由模拟类管理）
        self.setup_param_area(self.control_layout)
        
        # 数据组管理区域
        self.setup_data_group_area(self.control_layout)
        
        # 模拟控制按钮
        self.setup_simulation_controls(self.control_layout)
        
        # 添加拉伸项
        self.control_layout.addStretch()
        
        # 右侧绘图区域
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        
        # Matplotlib图形
        self.figure = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 将左右面板添加到主布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(plot_panel)
        splitter.setSizes([400, 1000])
        main_layout.addWidget(splitter)

        # 设置默认模拟
        if self.simulations:
            sim_name = list(self.simulations.keys())[0]
            self.select_simulation(sim_name)
        
    def register_simulations(self):
        """注册可用的模拟类"""
        from registry import registry
        self.simulations = registry
        
    def setup_simulation_selector(self, layout):
        """设置模拟选择区域"""
        group = QGroupBox("模拟选择")
        group_layout = QVBoxLayout()
        
        # 模拟选择下拉框
        self.sim_combo = QComboBox()
        self.sim_combo.addItems(self.simulations.keys())
        self.sim_combo.currentTextChanged.connect(self.on_simulation_changed)
        group_layout.addWidget(self.sim_combo)
        
        # 模拟描述
        self.sim_description = QLabel("选择一个模拟")
        self.sim_description.setWordWrap(True)
        self.sim_description.setStyleSheet("color: #666; font-style: italic;")
        self.sim_description.setFixedHeight(40)
        group_layout.addWidget(self.sim_description)
        
        group.setLayout(group_layout)
        layout.addWidget(group)

    def on_simulation_changed(self, sim_name):
        """模拟类型改变"""
        if sim_name in self.simulations:
            self.select_simulation(sim_name)
    
    def select_simulation(self, sim_name):
        """选择模拟"""
        self.current_simulation = self.simulations[sim_name]
        self.sim_description.setText(self.current_simulation.description)
        self.current_simulation.progress.connect(self.on_simulation_progress)
        self.update_param_area()
        
    def setup_plot_controls(self, layout):
        """设置绘图控制区域"""
        group = QGroupBox("绘图控制")
        group_layout = QVBoxLayout()
        
        # 移除了绘图类型选择下拉框
        
        # 子图布局控制
        subplot_layout = QGridLayout()
        subplot_layout.addWidget(QLabel("子图布局:"), 0, 0)
        
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 4)
        self.rows_spin.setValue(1)
        subplot_layout.addWidget(QLabel("行数:"), 0, 1)
        subplot_layout.addWidget(self.rows_spin, 0, 2)
        
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 4)
        self.cols_spin.setValue(1)
        subplot_layout.addWidget(QLabel("列数:"), 0, 3)
        subplot_layout.addWidget(self.cols_spin, 0, 4)
        
        group_layout.addLayout(subplot_layout)
        
        # 绘图操作按钮
        self.new_plot_btn = QPushButton("新建绘图布局")
        self.new_plot_btn.clicked.connect(self.create_plot_layout)
        self.new_plot_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        group_layout.addWidget(self.new_plot_btn)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
    def setup_subplot_navigation(self, layout):
        """设置子图导航区域"""
        group = QGroupBox("子图选择")
        group_layout = QVBoxLayout()
        
        # 当前子图显示
        self.subplot_display = QLabel("未创建子图")
        self.subplot_display.setAlignment(Qt.AlignCenter)
        self.subplot_display.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        self.subplot_display.setFixedHeight(40)
        group_layout.addWidget(self.subplot_display)
        
        # 子图索引选择
        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("子图索引:"))
        
        self.subplot_idx_spin = QSpinBox()
        self.subplot_idx_spin.setRange(1, 16)
        self.subplot_idx_spin.setValue(1)
        self.subplot_idx_spin.valueChanged.connect(self.on_subplot_index_changed)
        index_layout.addWidget(self.subplot_idx_spin)
        # 编辑子图属性按钮
        self.edit_props_btn = QPushButton("编辑属性")
        self.edit_props_btn.clicked.connect(self.edit_subplot_properties)
        index_layout.addWidget(self.edit_props_btn)
        
        index_layout.addStretch()
        group_layout.addLayout(index_layout)
        
        # 子图网格显示
        self.subplot_grid = QWidget()
        self.subplot_grid_layout = QGridLayout(self.subplot_grid)
        self.subplot_grid_layout.setSpacing(5)
        self.subplot_grid_layout.setContentsMargins(5, 5, 5, 5)
        group_layout.addWidget(self.subplot_grid)
        
        # 子图操作按钮
        btn_layout = QHBoxLayout()
        self.clear_current_btn = QPushButton("删除当前子图")
        self.clear_current_btn.clicked.connect(self.clear_current_subplot)
        btn_layout.addWidget(self.clear_current_btn)
        
        self.clear_all_btn = QPushButton("清空所有子图")
        self.clear_all_btn.clicked.connect(self.clear_all_subplots)
        btn_layout.addWidget(self.clear_all_btn)
        
        # 导出/导入按钮
        self.export_btn = QPushButton("导出全部数据")
        self.export_btn.clicked.connect(self.export_current_subplot_data)
        btn_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton("导入数据组")
        self.import_btn.clicked.connect(self.import_data_groups)
        btn_layout.addWidget(self.import_btn)
        group_layout.addLayout(btn_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        
    def setup_param_area(self, layout):
        """设置参数输入区域容器"""
        self.param_group = QGroupBox("参数设置")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)
        layout.addWidget(self.param_group)
        
    def update_param_area(self):
        """更新参数输入区域"""
        if not self.current_simulation:
            return
            
        # 清空现有参数控件
        for i in reversed(range(self.param_layout.count())): 
            widget = self.param_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # 更新组框标题
        self.param_group.setTitle(f"{self.current_simulation.name} - 参数设置")
        
        # 让模拟类创建参数控件
        self.current_simulation.create_param_widgets(self.param_group, self.param_layout)
        
    def setup_data_group_area(self, layout):
        """设置数据组管理区域"""
        self.data_group_container = QWidget()
        data_group_layout = QVBoxLayout(self.data_group_container)
        
        # 数据组管理标题
        self.data_group_title = QLabel("当前子图数据 (0 组)")
        self.data_group_title.setStyleSheet("font-weight: bold; color: #2196F3;")
        data_group_layout.addWidget(self.data_group_title)
        
        # 滚动区域用于显示数据组
        self.data_group_scroll = QScrollArea()
        self.data_group_scroll.setWidgetResizable(True)
        self.data_group_scroll.setMaximumHeight(300)
        
        self.data_group_content = QWidget()
        self.data_group_content_layout = QVBoxLayout(self.data_group_content)
        self.data_group_content_layout.setAlignment(Qt.AlignTop)
        
        self.data_group_scroll.setWidget(self.data_group_content)
        data_group_layout.addWidget(self.data_group_scroll)
        
        layout.addWidget(self.data_group_container)
        
    def setup_simulation_controls(self, layout):
        """设置模拟控制区域"""
        control_group = QGroupBox("模拟控制")
        control_layout = QVBoxLayout(control_group)
        
        self.solo_thread_chk=QCheckBox("独立线程")
        self.solo_thread_chk.setChecked(True)
        control_layout.addWidget(self.solo_thread_chk)
        # 模拟控制按钮
        self.run_btn = QPushButton("运行模拟")
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.run_btn.setEnabled(True)
        control_layout.addWidget(self.run_btn)
        
        # 进度显示
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("状态:"))
        self.progress_label = QLabel("准备就绪")
        self.progress_label.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.progress_label)
        self.time_label = QLabel("运行进度:00:00:00 | 00:00:00")
        self.time_label.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.time_label)
        progress_layout.addStretch()
        control_layout.addLayout(progress_layout)

        layout.addWidget(control_group)
        
    def _rebuild_plot_layout(self, rows, cols, preserve_data=False, reset_current=False):
        """内部方法：重建子图布局。

        Args:
            rows: 子图行数
            cols: 子图列数
            preserve_data: 是否保留已有数据组
            reset_current: 是否将当前选中子图重置为第一个
        """
        existing_current = self.current_ax_idx
        existing_managers = self.data_managers if preserve_data else {}

        # 清除原有图形并创建子图
        self.figure.clear()
        self.axes = []
        for i in range(rows * cols):
            ax = self.figure.add_subplot(rows, cols, i+1)
            ax.set_title(f'子图 {i+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            self.axes.append(ax)

        # 重建数据管理器
        self.data_managers = existing_managers.copy() if preserve_data else {}

        # 若 preserve_data 为 True，则保留索引有效的子图数据管理器，丢弃超范围部分
        if preserve_data:
            self.data_managers = {i: self.data_managers[i] for i in range(len(self.axes)) if i in self.data_managers}

        for i in range(len(self.axes)):
            if i not in self.data_managers:
                self.data_managers[i] = SubplotDataManager(i)

        # 更新子图索引范围
        self.subplot_idx_spin.setMaximum(rows * cols)
        if reset_current:
            self.current_ax_idx = 0
            self.subplot_idx_spin.setValue(1)
        else:
            if existing_current >= len(self.axes):
                self.current_ax_idx = len(self.axes) - 1
                self.subplot_idx_spin.setValue(len(self.axes))
            else:
                self.current_ax_idx = existing_current
                self.subplot_idx_spin.setValue(existing_current + 1)

        # 更新界面
        self.update_subplot_grid(rows, cols)
        self.update_current_subplot_display()
        self.update_data_group_display()

        # 重新绘制已有数据（如果有）
        for idx in range(len(self.axes)):
            if idx in self.data_managers and self.data_managers[idx].get_data_count() > 0:
                self.redraw_current_subplot(idx, draw=False)

        self.canvas.draw()

        if not preserve_data:
            self.status_bar.showMessage(f"已创建 {rows}x{cols} 子图布局")

    def _collect_datagroups(self):
        """按子图顺序收集所有 DataGroup 结构。"""
        return [self.data_managers[idx].get_all_data() for idx in sorted(self.data_managers.keys()) if idx in self.data_managers]

    def create_plot_layout(self):
        """创建子图布局，保留所有数据并按顺序重绘。"""
        existing_groups = self._collect_datagroups()
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()

        # 如果当前布局不足以容纳现有子图数据，则自动扩展列数
        existing_count = len(existing_groups)
        if existing_count > 0 and rows * cols < existing_count:
            cols = int(np.ceil(existing_count / rows))
            cols = min(cols, self.cols_spin.maximum())
            self.cols_spin.blockSignals(True)
            self.cols_spin.setValue(cols)
            self.cols_spin.blockSignals(False)

        self._rebuild_plot_layout(rows, cols, preserve_data=False, reset_current=True)

        # 重新添加旧数据
        self.current_ax_idx = 0
        self._add_datagroups(existing_groups)

    def ensure_subplot_count(self, required_count):
        """确保至少有足够的子图来显示指定数量的数据组"""
        if required_count <= len(self.axes):
            return

        rows = self.rows_spin.value()
        cols = int(np.ceil(required_count / rows))

        # 更新行数但不重置当前选中子图
        self.cols_spin.blockSignals(True)
        self.cols_spin.setValue(cols)
        self.cols_spin.blockSignals(False)

        self._rebuild_plot_layout(rows, cols, preserve_data=True, reset_current=False)

    def update_subplot_grid(self, rows, cols):
        """更新子图网格显示"""
        # 清除现有按钮
        for i in reversed(range(self.subplot_grid_layout.count())):
            widget = self.subplot_grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # 创建新的子图按钮网格
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                btn = QPushButton(f"{idx+1}")
                btn.setFixedSize(50, 50)
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, idx=idx: self.select_subplot(idx))
                
                # 如果是当前选中的子图，设置为选中状态
                if idx == self.current_ax_idx:
                    btn.setChecked(True)
                    btn.setStyleSheet("""
                        QPushButton {background-color: #2196F3;color: white;font-weight: bold;border: 2px solid #1976D2;}
                        QPushButton:checked {background-color: #1976D2;}""")
                else:
                    btn.setStyleSheet("""
                        QPushButton {background-color: #f0f0f0;border: 1px solid #ccc;}
                        QPushButton:hover {background-color: #e0e0e0;}""")
                
                self.subplot_grid_layout.addWidget(btn, i, j)
        
    def select_subplot(self, idx):
        """选择子图"""
        if 0 <= idx < len(self.axes):
            self.current_ax_idx = idx
            self.subplot_idx_spin.setValue(idx + 1)
            self.update_current_subplot_display()
            self.update_subplot_grid(self.rows_spin.value(), self.cols_spin.value())
            self.update_data_group_display()
        
    def on_subplot_index_changed(self, index):
        """子图索引改变"""
        idx = index - 1
        if 0 <= idx < len(self.axes):
            self.current_ax_idx = idx
            self.update_current_subplot_display()
            self.update_subplot_grid(self.rows_spin.value(), self.cols_spin.value())
            self.update_data_group_display()

    def edit_subplot_properties(self):
        """弹出子图属性对话框，编辑并应用到当前子图"""
        if self.current_ax_idx not in self.data_managers:
            QMessageBox.information(self, "信息", "请先创建子图布局")
            return

        manager = self.data_managers[self.current_ax_idx]
        dlg = SubplotPropertiesDialog(self, props=manager.properties)
        if dlg.exec_() == QDialog.Accepted:
            props = dlg.get_properties()
            manager.set_properties(props)
            # 立即应用并重绘
            self.redraw_current_subplot()
        
    def update_current_subplot_display(self):
        """更新当前子图显示"""
        if self.axes:
            rows = self.rows_spin.value()
            cols = self.cols_spin.value()
            row = self.current_ax_idx // cols + 1
            col = self.current_ax_idx % cols + 1
            self.subplot_display.setText(f"当前子图: {self.current_ax_idx+1} (第{row}行, 第{col}列)")
        else:
            self.subplot_display.setText("未创建子图")
            
    def update_data_group_display(self):
        """更新数据组显示"""
        # 清除现有数据组控件
        for widget in self.data_group_content.findChildren(QWidget):
            if isinstance(widget, DataGroupWidget):
                widget.deleteLater()
        
        self.data_group_widgets.clear()
        
        if self.current_ax_idx in self.data_managers:
            data_manager = self.data_managers[self.current_ax_idx]
            data_count = data_manager.get_data_count()
            
            # 更新标题
            self.data_group_title.setText(f"当前子图数据 ({data_count} 组)")
            
            # 添加数据组控件
            for data_group in data_manager.get_all_data():
                widget = DataGroupWidget(data_group, self)
                widget.label_changed.connect(self.update_legend_label)  # 连接信号
                self.data_group_content_layout.addWidget(widget)
                self.data_group_widgets[data_group.data_id] = widget
        else:
            self.data_group_title.setText(f"当前子图数据 (0 组)")
            
    def update_legend_label(self, data_id, new_label):
        """更新图例标签"""
        if self.current_ax_idx in self.data_managers:
            data_manager = self.data_managers[self.current_ax_idx]
            if data_id in data_manager.data_groups:
                data_group = data_manager.data_groups[data_id]
                data_group.label = new_label
                # 更新绘图对象的标签
                if data_group.plot_object:
                    if isinstance(data_group.plot_object, tuple):  # 直方图返回的是元组
                        # 对于直方图，更新图例条目的标签
                        handles, labels = self.axes[self.current_ax_idx].get_legend_handles_labels()
                        if data_group.label in labels:
                            idx = labels.index(data_group.label)
                            # 重新绘制图例
                            self.redraw_current_subplot()
                    else:  # 其他类型
                        for plot_object in data_group.plot_object:
                            plot_object.set_label(new_label)
                            self.axes[self.current_ax_idx].legend()
                            self.canvas.draw()

    def toggle_data_group(self, data_id):
        """切换数据组的可见性"""
        if self.current_ax_idx in self.data_managers:
            data_manager = self.data_managers[self.current_ax_idx]
            success, new_visible = data_manager.toggle_visibility(data_id)
            
            if success:
                # 重新绘制当前子图
                self.redraw_current_subplot()
                
    def delete_data_group(self, data_id):
        """删除数据组"""
        if self.current_ax_idx in self.data_managers:
            data_manager = self.data_managers[self.current_ax_idx]
            success = data_manager.remove_data(data_id)
            
            if success:
                # 删除控件
                if data_id in self.data_group_widgets:
                    self.data_group_widgets[data_id].deleteLater()
                    del self.data_group_widgets[data_id]
                
                # 重新绘制当前子图
                self.redraw_current_subplot()
                
                # 更新数据组显示
                self.update_data_group_display()
                
    def redraw_current_subplot(self, idx=None, draw=True):
        """重新绘制指定子图（默认当前子图）。"""
        if idx is None:
            idx = self.current_ax_idx
        if idx >= len(self.axes):
            return

        ax = self.axes[idx]
        ax.clear()

        # 应用子图管理器的属性（若存在）
        manager = self.data_managers.get(idx, None)
        if manager:
            ax.set_title(manager.properties['title'])
            ax.set_xlabel(manager.properties['xlabel'])
            ax.set_ylabel(manager.properties['ylabel'])
            if manager.properties['xlim'] is not None:
                ax.set_xlim(manager.properties['xlim'])
            if manager.properties['ylim'] is not None:
                ax.set_ylim(manager.properties['ylim'])
            if manager.properties['xlog']:
                ax.set_xscale('log')
            if manager.properties['ylog']:
                ax.set_yscale('log')
            if manager.properties['grid']:
                ax.grid(True, which="both", alpha=0.3)
            else:
                ax.grid(False)
        else:
            ax.set_title(f'子图 {idx+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)

        if idx in self.data_managers:
            data_manager = self.data_managers[idx]

            for data_group in data_manager.get_all_data():
                if data_group.visible:
                    plot_type = data_group.plot_type
                    # 基于 plot_props 构建绘图参数
                    props = data_group.plot_props if hasattr(data_group, 'plot_props') else {}
                    color = data_group.color
                    if plot_type == "线图":
                        plot_kwargs = {"label": data_group.label, "color": color}
                        if props.get('linewidth') is not None:
                            plot_kwargs['linewidth'] = props.get('linewidth')
                        if props.get('linestyle'):
                            plot_kwargs['linestyle'] = props.get('linestyle')
                        if props.get('marker'):
                            plot_kwargs['marker'] = props.get('marker')
                        data_group.plot_object = ax.plot(
                            data_group.x_data, data_group.y_data,
                            **plot_kwargs
                        )
                        ax.margins(0)
                    elif plot_type == "散点图":
                        s = props.get('s', 20)
                        marker = props.get('marker', 'o')
                        data_group.plot_object = ax.scatter(
                            data_group.x_data, data_group.y_data,
                            s=s, marker=marker, label=data_group.label, color=color
                        )
                        ax.margins(0)
                    elif plot_type == "直方图":
                        bins = props.get('bins', 30)
                        alpha = props.get('alpha', 0.7)
                        data_group.plot_object = ax.hist(
                            data_group.y_data, bins=bins, alpha=alpha,
                            label=data_group.label, color=color
                        )
                    elif plot_type == "误差图":
                        y_err = 0.1 * np.abs(data_group.y_data)
                        cap = props.get('capsize', 3)
                        fmt = props.get('fmt', 'o-')
                        data_group.plot_object = ax.errorbar(
                            data_group.x_data, data_group.y_data,
                            yerr=y_err, fmt=fmt, capsize=cap,
                            label=data_group.label, color=color
                        )
                    elif plot_type == "热图":
                        vmin, vmax = float(data_group.y_data.min()), float(data_group.y_data.max())
                        if vmin == vmax:
                            vmin -= 1e-12
                            vmax += 1e-12
                        if vmin < 0 < vmax:
                            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                        else:
                            norm = Normalize(vmin=vmin, vmax=vmax)
                        cmap_name = props.get('cmap', 'RdBu_r')
                        origin = props.get('origin', 'lower')
                        aspect = props.get('aspect', 'auto')
                        im = ax.imshow(data_group.y_data, cmap=cmap_name, norm=norm, origin=origin,
                                       aspect=aspect, extent=data_group.x_data)

                        # cbar = self.figure.colorbar(im, ax=ax)
                        # cbar.set_label('Δn', fontsize=12)
                        # cbar.formatter = matplotlib.FuncFormatter(lambda x, p: f'{x:.3f}')
                        # cbar.update_ticks()

                    else:  # 默认为线图
                        data_group.plot_object = ax.plot(
                            data_group.x_data, data_group.y_data, 
                            label=data_group.label, color=data_group.color
                        )

        # 添加图例
        if idx in self.data_managers and self.data_managers[idx].get_visible_data_count() > 0:
            ax.legend()

        if draw:
            self.canvas.draw()
            
    def clear_current_subplot(self):
        """删除当前子图（含数据），并重建布局、按顺序重绘剩余数据。"""
        if not self.axes:
            return

        idx = self.current_ax_idx
        all_groups = self._collect_datagroups()

        if idx < 0 or idx >= len(all_groups):
            return

        # 删除当前子图对应的数据组
        all_groups.pop(idx)

        # 计算新子图数量
        new_count = len(all_groups)
        if new_count == 0:
            self._rebuild_plot_layout(1, 1, preserve_data=False, reset_current=True)
            self.update_data_group_display()
            self.status_bar.showMessage("已删除当前子图，剩余0个子图")
            return

        # 重新计算布局，尽量保持当前行数，同时保证能容纳所有子图
        rows = min(self.rows_spin.value(), new_count)
        cols = int(np.ceil(new_count / rows))
        cols = max(1, min(cols, self.cols_spin.maximum()))
        self.rows_spin.blockSignals(True)
        self.cols_spin.blockSignals(True)
        self.rows_spin.setValue(rows)
        self.cols_spin.setValue(cols)
        self.rows_spin.blockSignals(False)
        self.cols_spin.blockSignals(False)

        self._rebuild_plot_layout(rows, cols, preserve_data=False, reset_current=True)

        # 重新按顺序添加剩余数据
        self.current_ax_idx = 0
        self._add_datagroups(all_groups)

        # 更新显示与状态
        self.update_data_group_display()
        self.status_bar.showMessage(f"已删除子图 {idx+1}，剩余 {new_count} 个子图")
            
    def clear_all_subplots(self):
        """清空所有子图"""
        if not self.axes:
            return
            
        for idx, ax in enumerate(self.axes):
            # 清除数据管理器
            if idx in self.data_managers:
                self.data_managers[idx].clear_all()
            
            # 清除子图内容
            ax.clear()
            ax.set_title(f'子图 {idx+1}')
            ax.grid(True, alpha=0.3)
        
        # 更新数据组显示
        self.update_data_group_display()
        
        self.canvas.draw()
        self.status_bar.showMessage("已清空所有子图")

    def _sanitize_filename(self, s):
        keep = " ._()-[]"
        return ''.join(c for c in s if c.isalnum() or c in keep).rstrip()

    def _add_datagroups(self, data_groups):
        """添加 DataGroups"""
        # 向后兼容：若传入的是 List[DataGroup]，则包装成 List[List[DataGroup]]
        if isinstance(data_groups, list) and data_groups and not isinstance(data_groups[0], list):
            data_groups = [data_groups]

        # 确保有足够的子图来展示所有返回的数据组
        required = self.current_ax_idx + len(data_groups)
        self.ensure_subplot_count(required)

        # 将 data_groups 依次添加到对应的 SubplotDataManager
        for offset, dg_list in enumerate(data_groups):
            idx = self.current_ax_idx + offset
            if idx not in self.data_managers:
                continue

            manager = self.data_managers[idx]
            for dg in dg_list:
                manager.add_datagroup(dg)

            # 只重新绘制当前子图和新增子图（不清空已有数据）
            self.redraw_current_subplot(idx)

        # 更新当前子图显示（保持当前选中子图）
        self.update_data_group_display()

    def export_current_subplot_data(self):
        """导出当前所有子图的 DataGroup，统一格式（List[List[DataGroup]]）。"""
        if not self.data_managers:
            QMessageBox.information(self, "信息", "当前没有数据可导出")
            return

        data_all = [self.data_managers[idx].get_all_data() for idx in sorted(self.data_managers.keys())]

        export_dir = os.path.join(os.getcwd(), 'datagroup')
        os.makedirs(export_dir, exist_ok=True)
        fpath, _ = QFileDialog.getSaveFileName(self, "保存数据组", os.path.join(export_dir, 'datagroups.dgdata'), "DataGroup 文件 (*.dgdata)")
        if not fpath:
            return

        try:
            with open(fpath, 'wb') as f:
                pickle.dump(data_all, f)
            QMessageBox.information(self, "导出完成", f"已导出 {len(data_all)} 个子图数据（DataGroup 格式）")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"写入失败: {e}")

    def import_data_groups(self):
        """导入刚好与模拟结果协议一致的 DataGroup 格式: List[List[DataGroup]]。"""
        if not self.axes:
            QMessageBox.information(self, "信息", "请先创建子图布局")
            return

        start_dir = os.path.join(os.getcwd(), 'datagroup')
        fpath, _ = QFileDialog.getOpenFileName(self, "选择要导入的数据组文件", start_dir, "DataGroup 文件 (*.dgdata)")
        if not fpath:
            return

        try:
            with open(fpath, 'rb') as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, list) or (loaded and not isinstance(loaded[0], list)):
                raise ValueError("文件格式不符合：应为 List[List[DataGroup]]")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"加载失败: {e}")
            return

        self._add_datagroups(loaded)

        QMessageBox.information(self, "导入完成", f"已导入 {len(loaded)} 个子图数据，每组内包含 DataGroup")
        
    def run_simulation(self):
        """运行模拟"""
        if not self.current_simulation:
            QMessageBox.warning(self, "警告", "请先选择模拟类型！")
            return
            
        if not self.axes:
            QMessageBox.warning(self, "警告", "请先创建子图布局！")
            return
            
        # 禁用运行按钮，防止重复点击
        self.run_btn.setEnabled(False)
        self.progress_label.setText("模拟运行中...")
        self.progress_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        
        if self.solo_thread_chk.isChecked():
            # 创建并启动模拟线程
            self.simulation_thread = SimulationThread(self.current_simulation)  # 移除plot_type参数
            # 连接自定义结果信号，避免与 QThread.finished 冲突
            self.simulation_thread.result.connect(self.on_simulation_finished)
            #self.simulation_thread.progress.connect(self.on_simulation_progress)
            self.simulation_thread.start()
        else:
            data_groups = self.current_simulation.run_simulation()
            if not isinstance(data_groups, list):
                data_groups = [data_groups]
            self.on_simulation_finished(data_groups)
        
    def on_simulation_progress(self, time):
        """模拟进度更新"""
        elapsed_time=time.get('elapsed')/1000
        elapsed_str=str(timedelta(elapsed_time)).split('.')[0]
        remaining_time=elapsed_time*(time.get('total')-time.get('n'))/time.get('n')
        remaining_str=str(timedelta(remaining_time)).split('.')[0]
        self.time_label.setText("模拟进度:"+elapsed_str+'|'+remaining_str)
        
    def on_simulation_finished(self, data_groups):
        """模拟完成，处理结果

        data_groups: List[List[DataGroup]]
            外层列表：按子图序号顺序，内层列表：同一个子图的多个 DataGroup
        """
        if not data_groups:
            # 模拟失败
            self.progress_label.setText("模拟失败")
            self.progress_label.setStyleSheet("color: #F44336;")
            self.run_btn.setEnabled(True)
            return

        if not self.axes or self.current_ax_idx >= len(self.axes):
            return

        self._add_datagroups(data_groups)

        # 更新状态
        self.progress_label.setText("模拟完成")
        self.progress_label.setStyleSheet("color: #4CAF50;")
        self.run_btn.setEnabled(True)
        self.status_bar.showMessage(f"已从子图 {self.current_ax_idx+1} 起依次添加 {len(data_groups)} 组数据")
    
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用样式
    palette = app.palette()
    palette.setColor(palette.Window, QColor(240, 240, 240))  # 浅灰色背景
    palette.setColor(palette.WindowText, Qt.black)
    palette.setColor(palette.Base, QColor(255, 255, 255))
    palette.setColor(palette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(palette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ToolTipText, Qt.black)
    palette.setColor(palette.Text, Qt.black)
    palette.setColor(palette.Button, QColor(240, 240, 240))
    palette.setColor(palette.ButtonText, Qt.black)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(66, 165, 245))
    palette.setColor(palette.HighlightedText, Qt.white)
    app.setPalette(palette)
    
    window = Plotter()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()