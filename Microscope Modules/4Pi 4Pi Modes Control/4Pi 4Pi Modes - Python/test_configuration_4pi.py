#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import json  # noqa

from os import path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout, QLabel, QWidget,
    QScrollArea, QSlider, QDoubleSpinBox, QFrame,
    QApplication, QMainWindow, QPushButton,
    )
from matplotlib.figure import Figure

from matplotlib.backends.backend_qt5agg import FigureCanvas

from dmplot import DMPlot
import matplotlib.pyplot as plt  # noqa


class RelSlider:

    def __init__(self, val, cb):
        self.old_val = None
        self.fto100mul = 100
        self.cb = cb

        self.sba = QDoubleSpinBox()
        self.sba.setMinimum(-1000)
        self.sba.setMaximum(1000)
        self.sba.setDecimals(6)
        self.sba.setToolTip('Effective value')
        self.sba.setValue(val)
        self.sba_color(val)
        self.sba.setSingleStep(1.25e-3)

        self.qsr = QSlider(Qt.Horizontal)
        self.qsr.setMinimum(-100)
        self.qsr.setMaximum(100)
        self.qsr.setValue(0)
        self.qsr.setToolTip('Drag to apply relative delta')

        self.sbm = QDoubleSpinBox()
        self.sbm.setMinimum(0.01)
        self.sbm.setMaximum(1000)
        self.sbm.setSingleStep(1.25e-3)
        self.sbm.setToolTip('Maximum relative delta')
        self.sbm.setDecimals(2)
        self.sbm.setValue(4.0)

        def sba_cb():
            def f():
                self.block()
                val = self.sba.value()
                self.sba_color(val)
                self.cb(val)
                self.unblock()
            return f

        def qs1_cb():
            def f(t):
                self.block()

                if self.old_val is None:
                    self.qsr.setValue(0)
                    self.unblock()
                    return

                val = self.old_val + self.qsr.value()/100*self.sbm.value()
                self.sba.setValue(val)
                self.sba_color(val)
                self.cb(val)

                self.unblock()
            return f

        def qs1_end():
            def f():
                self.block()
                self.qsr.setValue(0)
                self.old_val = None
                self.unblock()
            return f

        def qs1_start():
            def f():
                self.block()
                self.old_val = self.get_value()
                self.unblock()
            return f

        self.sba_cb = sba_cb()
        self.qs1_cb = qs1_cb()
        self.qs1_start = qs1_start()
        self.qs1_end = qs1_end()

        self.sba.valueChanged.connect(self.sba_cb)
        self.qsr.valueChanged.connect(self.qs1_cb)
        self.qsr.sliderPressed.connect(self.qs1_start)
        self.qsr.sliderReleased.connect(self.qs1_end)

    def sba_color(self, val):
        if abs(val) > 1e-4:
            self.sba.setStyleSheet("font-weight: bold;")
        else:
            self.sba.setStyleSheet("font-weight: normal;")
        # self.sba.update()

    def block(self):
        self.sba.blockSignals(True)
        self.qsr.blockSignals(True)
        self.sbm.blockSignals(True)

    def unblock(self):
        self.sba.blockSignals(False)
        self.qsr.blockSignals(False)
        self.sbm.blockSignals(False)

    def enable(self):
        self.sba.setEnabled(True)
        self.qsr.setEnabled(True)
        self.sbm.setEnabled(True)

    def disable(self):
        self.sba.setEnabled(False)
        self.qsr.setEnabled(False)
        self.sbm.setEnabled(False)

    def fto100(self, f):
        return int((f + self.m2)/(2*self.m2)*self.fto100mul)

    def get_value(self):
        return self.sba.value()

    def set_value(self, v):
        self.sba_color(v)
        return self.sba.setValue(v)

    def add_to_layout(self, l1, ind1, ind2):
        l1.addWidget(self.sba, ind1, ind2)
        l1.addWidget(self.qsr, ind1, ind2 + 1)
        l1.addWidget(self.sbm, ind1, ind2 + 2)

    def remove_from_layout(self, l1):
        l1.removeWidget(self.sba)
        l1.removeWidget(self.qsr)
        l1.removeWidget(self.sbm)

        self.sba.setParent(None)
        self.qsr.setParent(None)
        self.sbm.setParent(None)

        self.sba.valueChanged.disconnect(self.sba_cb)
        self.qsr.valueChanged.disconnect(self.qs1_cb)
        self.qsr.sliderPressed.disconnect(self.qs1_start)
        self.qsr.sliderReleased.disconnect(self.qs1_end)

        self.sba_cb = None
        self.qs1_cb = None
        self.qs1_start = None
        self.qs1_end = None

        self.sb = None
        self.qsr = None


class DMWindow(QMainWindow):

    def __init__(self, app, C, modes):
        super().__init__()
        self.C = C
        self.modes = modes
        self.dmplot0 = DMPlot()
        self.dmplot1 = DMPlot()
        self.u = np.zeros(C.shape[0])
        self.z = np.zeros(C.shape[1])

        fig0 = FigureCanvas(Figure(figsize=(2, 2)))
        ax0 = fig0.figure.subplots(1, 1)
        ima0 = self.dmplot0.draw(ax0, self.u[:140])
        ax0.axis('off')

        fig1 = FigureCanvas(Figure(figsize=(2, 2)))
        ax1 = fig1.figure.subplots(1, 1)
        ima1 = self.dmplot1.draw(ax1, self.u[140:])
        ax1.axis('off')

        def update():
            self.u = np.dot(self.C, self.z)
            ima0.set_data(self.dmplot0.compute_pattern(self.u[:140]))
            ax0.figure.canvas.draw()
            ima1.set_data(self.dmplot1.compute_pattern(self.u[140:]))
            ax1.figure.canvas.draw()

        def make_callback(i):
            def f(r):
                self.z[i] = r
                update()
            return f

        scroll = QScrollArea()
        scroll.setWidget(QWidget())
        scroll.setWidgetResizable(True)
        lay = QGridLayout(scroll.widget())
        for i in range(len(modes)):
            lab = QLabel(modes[i])
            slider = RelSlider(0., make_callback(i))
            lay.addWidget(lab, i, 0)
            slider.add_to_layout(lay, i, 1)

        breset = QPushButton('reset')

        def reset_fun():
            self.z *= 0.
            update()

        breset.clicked.connect(reset_fun)

        main = QFrame()
        top = QGridLayout()
        top.addWidget(fig0, 0, 0)
        top.addWidget(fig1, 0, 1)
        top.addWidget(scroll, 1, 0, 1, 2)
        top.addWidget(breset, 2, 0)
        main.setLayout(top)
        self.setCentralWidget(main)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    fname = path.join(
        '..', '4Pi 4Pi Modes - Default Files', 'config.json')
    with open(fname, 'r') as f:
        conf = json.load(f)

    dm0 = DMPlot()
    dm1 = DMPlot()

    C = np.array(conf['Matrix'])
    modes = conf['Modes']

    zwindow = DMWindow(app, C, modes)
    zwindow.show()

    sys.exit(app.exec_())
