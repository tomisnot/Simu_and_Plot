from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib import cm

class ColorPickerDialog(QDialog):
    """颜色选择对话框，显示一组常用颜色供选择"""
    def __init__(self, parent=None, initial_color=None):
        super().__init__(parent)
        self.setWindowTitle("选择颜色")
        self.setModal(True)
        self.selected = None

        layout = QVBoxLayout(self)
        grid = QGridLayout()

        cmap = cm.get_cmap('tab20')
        n = 20
        for i in range(n):
            rgba = cmap(i)
            r, g, b, a = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), rgba[3]
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(f"background-color: rgba({r}, {g}, {b}, {a}); border: 1px solid #ccc;")
            btn.clicked.connect(lambda checked, c=rgba: self._choose(c))
            grid.addWidget(btn, i//10, i%10)

        layout.addLayout(grid)

        custom = QPushButton("自定义...")
        custom.clicked.connect(self._custom_color)
        layout.addWidget(custom)

        btns = QHBoxLayout()
        ok = QPushButton("确定")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

        # 预置初始颜色（如果提供）
        if initial_color is not None:
            self.selected = initial_color

    def _choose(self, rgba):
        self.selected = rgba
        self.accept()

    def _custom_color(self):
        q = QColorDialog.getColor()
        if q.isValid():
            r, g, b, a = q.red()/255.0, q.green()/255.0, q.blue()/255.0, q.alpha()/255.0
            self.selected = (r, g, b, a)
            self.accept()

    def get_color(self):
        return self.selected

class SubplotPropertiesDialog(QDialog):
    def __init__(self, parent=None, props=None):
        super().__init__(parent)
        self.setWindowTitle("子图属性")
        self.setModal(True)
        layout = QVBoxLayout(self)

        range_layout = QGridLayout()
        self.title_edit = QLineEdit()
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()
        range_layout.addWidget(QLabel("标题:"), 0, 0)
        range_layout.addWidget(self.title_edit, 0, 1)
        range_layout.addWidget(QLabel("X 标签:"), 0, 2)
        range_layout.addWidget(self.xlabel_edit, 0, 3)
        range_layout.addWidget(QLabel("Y 标签:"), 0, 4)
        range_layout.addWidget(self.ylabel_edit, 0, 5)
        
        range_layout.addWidget(QLabel("X 最小"), 1, 0)
        self.xmin = QLineEdit()
        range_layout.addWidget(self.xmin, 1, 1)
        range_layout.addWidget(QLabel("X 最大"), 1, 2)
        self.xmax = QLineEdit()
        range_layout.addWidget(self.xmax, 1, 3)
        self.xlog = QCheckBox("X 对数坐标")
        range_layout.addWidget(self.xlog, 1, 4)

        range_layout.addWidget(QLabel("Y 最小"), 2, 0)
        self.ymin = QLineEdit()
        range_layout.addWidget(self.ymin, 2, 1)
        range_layout.addWidget(QLabel("Y 最大"), 2, 2)
        self.ymax = QLineEdit()
        range_layout.addWidget(self.ymax, 2, 3)
        self.ylog = QCheckBox("Y 对数坐标")
        range_layout.addWidget(self.ylog, 2, 4)

        self.grid = QCheckBox("显示网格")
        range_layout.addWidget(self.grid, 3, 0)

        layout.addLayout(range_layout)

        btns = QHBoxLayout()
        ok = QPushButton("确定")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

        if props:
            self.title_edit.setText(props.get('title', ''))
            self.xlabel_edit.setText(props.get('xlabel', ''))
            self.ylabel_edit.setText(props.get('ylabel', ''))
            xlim = props.get('xlim')
            ylim = props.get('ylim')
            if xlim:
                self.xmin.setText(str(xlim[0]))
                self.xmax.setText(str(xlim[1]))
            if ylim:
                self.ymin.setText(str(ylim[0]))
                self.ymax.setText(str(ylim[1]))
            self.xlog.setChecked(props.get('xlog', False))
            self.ylog.setChecked(props.get('ylog', False))
            self.grid.setChecked(props.get('grid', True))

    def get_properties(self):
        def to_float_or_none(s):
            s = s.strip()
            if s == '':
                return None
            try:
                return float(s)
            except Exception:
                return None

        xmn = to_float_or_none(self.xmin.text())
        xmx = to_float_or_none(self.xmax.text())
        ymn = to_float_or_none(self.ymin.text())
        ymx = to_float_or_none(self.ymax.text())

        xlim = (xmn, xmx) if xmn is not None and xmx is not None else None
        ylim = (ymn, ymx) if ymn is not None and ymx is not None else None

        return {
            'title': self.title_edit.text().strip(),
            'xlabel': self.xlabel_edit.text().strip(),
            'ylabel': self.ylabel_edit.text().strip(),
            'xlim': xlim,
            'ylim': ylim,
            'xlog': self.xlog.isChecked(),
            'ylog': self.ylog.isChecked(),
        }

class DataGroupPropertiesDialog(QDialog):
    """根据数据组的 plot_type 显示对应的可编辑属性"""
    def __init__(self, parent=None, plot_type="线图", props=None):
        super().__init__(parent)
        self.setWindowTitle("数据组属性")
        self.setModal(True)
        self.plot_type = plot_type
        self.props = props or {}
        self._widgets = {}

        layout = QVBoxLayout(self)

        # 动态构建表单
        form = QGridLayout()
        row = 0
        def add_row(label_text, widget):
            nonlocal row
            form.addWidget(QLabel(label_text), row, 0)
            form.addWidget(widget, row, 1)
            row += 1
            return widget

        if plot_type == "线图":
            lw = QLineEdit(str(self.props.get('linewidth', 2.0)))
            ls = QLineEdit(str(self.props.get('linestyle', '-')))
            mk = QLineEdit(str(self.props.get('marker', '')))
            self._widgets['linewidth'] = lw
            self._widgets['linestyle'] = ls
            self._widgets['marker'] = mk
            add_row("线宽:", lw)
            add_row("线型:", ls)
            add_row("Marker:", mk)

        elif plot_type == "散点图":
            s = QLineEdit(str(self.props.get('s', 20)))
            mk = QLineEdit(str(self.props.get('marker', 'o')))
            self._widgets['s'] = s
            self._widgets['marker'] = mk
            add_row("点大小:", s)
            add_row("Marker:", mk)

        elif plot_type == "直方图":
            bins = QLineEdit(str(self.props.get('bins', 30)))
            alpha = QLineEdit(str(self.props.get('alpha', 0.7)))
            self._widgets['bins'] = bins
            self._widgets['alpha'] = alpha
            add_row("bins:", bins)
            add_row("透明度:", alpha)

        elif plot_type == "误差图":
            cap = QLineEdit(str(self.props.get('capsize', 3)))
            fmt = QLineEdit(str(self.props.get('fmt', 'o-')))
            self._widgets['capsize'] = cap
            self._widgets['fmt'] = fmt
            add_row("capsize:", cap)
            add_row("格式(fmt):", fmt)

        elif plot_type == "热图":
            cmap_edit = QLineEdit(str(self.props.get('cmap', 'RdBu_r')))
            origin_edit = QLineEdit(str(self.props.get('origin', 'lower')))
            self._widgets['cmap'] = cmap_edit
            self._widgets['origin'] = origin_edit
            add_row("cmap:", cmap_edit)
            add_row("origin:", origin_edit)

        layout.addLayout(form)

        btns = QHBoxLayout()
        ok = QPushButton("确定")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def get_properties(self):
        out = {}
        for k, w in self._widgets.items():
            text = w.text().strip()
            if text == '':
                continue
            # 尝试转换为数值
            try:
                if '.' in text:
                    val = float(text)
                else:
                    val = int(text)
            except Exception:
                val = text
            out[k] = val
        return out


class DataGroupWidget(QWidget):
    """数据组控件，用于显示和操作单个数据组"""
    label_changed = pyqtSignal(int, str)  # 添加标签改变信号
    
    def __init__(self, data_group, parent=None):
        super().__init__(parent)
        self.data_group = data_group
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        
        # 使用可点击的按钮显示颜色，点击弹出颜色选择
        color_btn = QPushButton()
        color_btn.setFixedSize(20, 20)
        color_btn.setCursor(Qt.PointingHandCursor)
        color_btn.setToolTip("点击更改颜色")
        rgba = self.data_group.color
        color_btn.setStyleSheet(f"background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]}); border: 1px solid #ccc;")
        color_btn.clicked.connect(self.on_color_clicked)
        layout.addWidget(color_btn)
        self._color_btn = color_btn
        
        # 数据标签（可编辑的QLineEdit）
        self.label_edit = QLineEdit(self.data_group.label)
        self.label_edit.setMaximumWidth(150)
        self.label_edit.setToolTip("点击编辑数据组名称，按Enter确认")
        self.label_edit.editingFinished.connect(self.on_label_edited)
        layout.addWidget(self.label_edit)
        
        layout.addStretch()
        
        # 隐藏/显示按钮
        self.toggle_btn = QPushButton("隐藏" if self.data_group.visible else "显示")
        self.toggle_btn.setFixedSize(50, 20)
        self.toggle_btn.setStyleSheet("font-size: 10px;")
        self.toggle_btn.clicked.connect(self.on_toggle_clicked)
        layout.addWidget(self.toggle_btn)
        
        # 属性按钮（根据 plot_type 编辑作图参数）
        self.props_btn = QPushButton("属性")
        self.props_btn.setFixedSize(50, 20)
        self.props_btn.setStyleSheet("font-size: 10px;")
        self.props_btn.clicked.connect(self.on_props_clicked)
        layout.addWidget(self.props_btn)
        
        # 删除按钮
        self.delete_btn = QPushButton("删除")
        self.delete_btn.setFixedSize(50, 20)
        self.delete_btn.setStyleSheet("font-size: 10px; background-color: #f44336; color: white;")
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        layout.addWidget(self.delete_btn)
        
        self.setLayout(layout)
        self.setMaximumHeight(30)
        
    def on_label_edited(self):
        """标签编辑完成"""
        new_label = self.label_edit.text().strip()
        if new_label and new_label != self.data_group.label:
            self.data_group.label = new_label
            self.label_changed.emit(self.data_group.data_id, new_label)
        
    def on_toggle_clicked(self):
        """隐藏/显示按钮点击"""
        self.toggle_btn.setText("显示" if self.data_group.visible else "隐藏")
        self.window().toggle_data_group(self.data_group.data_id)
        
    def on_delete_clicked(self):
        """删除按钮点击"""
        self.window().delete_data_group(self.data_group.data_id)

    def on_color_clicked(self):
        """弹出颜色选择对话框并应用选择的颜色"""
        dlg = ColorPickerDialog(self, initial_color=self.data_group.color)
        if dlg.exec_() == QDialog.Accepted:
            new_rgba = dlg.get_color()
            if new_rgba is not None:
                self.data_group.color = new_rgba
                r, g, b, a = new_rgba
                self._color_btn.setStyleSheet(f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a}); border: 1px solid #ccc;")
                # 立即重绘当前子图
                self.window().redraw_current_subplot()

    def on_props_clicked(self):
        """弹出数据组属性编辑对话框，保存并重绘"""
        dlg = DataGroupPropertiesDialog(self, plot_type=self.data_group.plot_type, props=self.data_group.plot_props)
        if dlg.exec_() == QDialog.Accepted:
            new_props = dlg.get_properties()
            # 合并到 data_group.plot_props
            self.data_group.plot_props.update(new_props)
            self.window().redraw_current_subplot()
