import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget,QVBoxLayout
from PyQt5.QtWidgets import QGridLayout, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 tabs - pythonspot.com'
        self.left = 0
        self.top = 0
        self.width = 700
        self.height = 550
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        
        self.show()
    
class MyTableWidget(QWidget):
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(700,550)
        
        # Add tabs
        self.tabs.addTab(self.tab1,"Input")
        self.tabs.addTab(self.tab2,"Dataset")
        
        self.InputWidget()
        
        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def InputWidget(self):
        # Create first tab
        layout = QGridLayout()
        layout.setVerticalSpacing(0)
        upload = QPushButton("Upload File")
        reset = QPushButton("Reset")
        title = QLabel("Gender Recognition with Local Binary Pattern & Linear SVM")
        imagePriview = QLabel()
        imagePriview.setPixmap(QPixmap("upload.png"))
        imagePriview.setFixedSize(100,100)
        layout.addWidget(QLabel("Gender Recognition with Local Binary Pattern & Linear SVM"), 0,0)
        layout.addWidget(imagePriview, 1,0)
        layout.addWidget(upload, 2,0)
        layout.addWidget(reset, 2,1)

        self.tab1.setLayout(layout)

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())