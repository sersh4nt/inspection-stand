from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class CheckerWidget(QWidget):
    def __init__(self, parent=None):
        super(CheckerWidget, self).__init__(parent)
        self.status = QLabel(self.centralwidget)
        self.status.setObjectName("status")
        self.verticalLayout.addWidget(self.status)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.scratch_indicator = QLabel(self.centralwidget)
        self.scratch_indicator.setObjectName("scratch_indicator")
        self.verticalLayout_2.addWidget(self.scratch_indicator)
        self.hole_indicator = QLabel(self.centralwidget)
        self.hole_indicator.setObjectName("hole_indicator")
        self.verticalLayout_2.addWidget(self.hole_indicator)
        self.leg_indicator = QLabel(self.centralwidget)
        self.leg_indicator.setObjectName("leg_indicator")
        self.verticalLayout_2.addWidget(self.leg_indicator)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_3.addWidget(self.label_5)
        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.label_8 = QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_3.addWidget(self.label_8)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout_3)


