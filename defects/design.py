# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'defects/main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from libs.camera import CameraWidget


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1013, 772)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.main_view = CameraWidget(0, parent=self.centralwidget)
        self.main_view.setObjectName("main_view")
        self.horizontalLayout.addWidget(self.main_view)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.status = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.status.sizePolicy().hasHeightForWidth())
        self.status.setSizePolicy(sizePolicy)
        self.status.setObjectName("status")
        self.verticalLayout.addWidget(self.status)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.scratches = QtWidgets.QLabel(self.centralwidget)
        self.scratches.setObjectName("scratch_inidcator")
        self.gridLayout.addWidget(self.scratches, 0, 0, 1, 1)
        self.holes = QtWidgets.QLabel(self.centralwidget)
        self.holes.setObjectName("holes_indicator")
        self.gridLayout.addWidget(self.holes, 1, 0, 1, 1)
        self.legs = QtWidgets.QLabel(self.centralwidget)
        self.legs.setObjectName("legs_indicator")
        self.gridLayout.addWidget(self.legs, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.defect_view = QtWidgets.QLabel(self.centralwidget)
        self.defect_view.setObjectName("defect_view")
        self.defect_view.setSizePolicy(sizePolicy)
        self.verticalLayout.addWidget(self.defect_view)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.prev_defect_button = QtWidgets.QPushButton(self.centralwidget)
        self.prev_defect_button.setObjectName("prev_defect_button")
        self.horizontalLayout_2.addWidget(self.prev_defect_button)
        self.next_defect_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_defect_button.setObjectName("next_defect_button")
        self.horizontalLayout_2.addWidget(self.next_defect_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 10)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 10)
        self.verticalLayout.setStretch(5, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.setStretch(0, 4)
        self.horizontalLayout.setStretch(1, 1)
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1013, 21))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Поиск дефектов"))
        self.main_view.setText(_translate("mainWindow", "TextLabel"))
        self.label_3.setText(_translate("mainWindow", "Статус детали:"))
        self.label_5.setText(_translate("mainWindow", "Наличие царапин"))
        self.label_6.setText(_translate("mainWindow", "Наличие отверстий"))
        self.label_8.setText(_translate("mainWindow", "Погнутые ножки"))
        self.label_7.setText(_translate("mainWindow", "Фото возможного дефекта:"))
        self.defect_view.setText(_translate("mainWindow", "TextLabel"))
        self.prev_defect_button.setText(_translate("mainWindow", "Предыдущий\nдефект"))
        self.next_defect_button.setText(_translate("mainWindow", "Следующий\nдефект"))
