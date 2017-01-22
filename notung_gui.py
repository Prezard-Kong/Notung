from __future__ import unicode_literals
from PyQt4 import QtGui, QtCore
from gui.main_window import Ui_mainWindow
from gui.style_window import Ui_styleWindow
import sys
from search import ImageSearcher
from model import NeuralStyle, FastNeuralStyle


def change_value(edit, args, dst, category):
    def change_value_impl():
        if not edit.text().isEmpty():
            try:
                args[dst] = category(edit.text())
            except ValueError:
                pass
        elif category is str:
            args[dst] = None
    edit.textChanged.connect(change_value_impl)


class MainWindow(QtGui.QDialog, Ui_mainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.searcher = ImageSearcher()
        self.setupUi(self)
        self.Next.setEnabled(False)
        self.Previous.setEnabled(False)
        self.style_window = StyleWindow()
        self.args = {'inquery num': 10, 'semantic query': None}
        self.names = []
        self.img_query = None
        self.img_inqueries = []
        self.caption_inqueries = []
        self.idx = 0
        self.inqeury_num.setText(QtCore.QString(str(self.args['inquery num'])))
        change_value(self.inqeury_num, self.args, 'inquery num', int)
        self.neuralStyle.clicked.connect(self.style_window.show)
        self.addFolder_db.clicked.connect(self.add_folder)
        self.addFiles_db.clicked.connect(self.add_files)
        self.selectFlie_imgsearch.clicked.connect(self.select_img_query)
        change_value(self.lineEdit_semsearch, self.args, 'semantic query', str)
        self.build_db.clicked.connect(self.build_database)
        self.imgsearch.clicked.connect(self.img_search)
        self.semsearch.clicked.connect(self.caption_search)
        self.Next.clicked.connect(lambda: self.press_next_previous(False))
        self.Previous.clicked.connect(lambda: self.press_next_previous(True))

    def add_folder(self):
        folder = QtGui.QFileDialog().getExistingDirectory(self, 'Add Folder')
        tmp = str(folder)
        self.names.append(tmp)
        self.show_files()

    def add_files(self):
        files = QtGui.QFileDialog().getOpenFileNames(self, 'Add Files')
        for f in files:
            self.names.append(str(f))
        self.show_files()

    def show_files(self):
        tmp = '; '.join(self.names)
        self.lineEdit_db.setText(QtCore.QString(tmp))

    def select_img_query(self):
        self.img_query = QtGui.QFileDialog().getOpenFileName(self, 'Select File')
        self.lineEdit_imgserach.setText(self.img_query)

    def build_database(self):
        QtGui.QMessageBox().information(self, 'Information', 'Please wait...')
        self.searcher.build_database(self.names)
        QtGui.QMessageBox().information(self, 'Information', 'Done')

    def img_search(self):
        if self.img_query is not None:
            tmp = str(self.img_query)
            QtGui.QMessageBox().information(self, 'Information', 'Please wait...')
            self.img_inqueries, self.caption_inqueries = self.searcher.image_similarity_search(
                tmp, self.args['inquery num'])
            QtGui.QMessageBox().information(self, 'Information', 'Done')
            if self.img_inqueries is None:
                self.img_inqueries = []
                self.caption_inqueries = []
                return
            self.idx = 0
            self.show_result()
        else:
            QtGui.QMessageBox().information(self, 'Information', 'Please select a file!')

    def caption_search(self):
        if self.args['semantic query'] is not None:
            QtGui.QMessageBox().information(self, 'Information', 'Please wait...')
            self.img_inqueries, self.caption_inqueries = self.searcher.semantic_similarity_search(
                self.args['semantic query'], self.args['inquery num'])
            QtGui.QMessageBox().information(self, 'Information', 'Done')
            if self.img_inqueries is None:
                self.img_inqueries = []
                self.caption_inqueries = []
                return
            self.idx = 0
            self.show_result()
        else:
            QtGui.QMessageBox().information(self, 'Information', 'Please input keywords!')

    def show_result(self):
        if len(self.img_inqueries) > 0:
            if self.idx < len(self.img_inqueries)-1:
                self.Next.setEnabled(True)
            else:
                self.Next.setEnabled(False)
            if self.idx > 0:
                self.Previous.setEnabled(True)
            else:
                self.Previous.setEnabled(False)
            self.lineEdit_path.setText(QtCore.QString(self.img_inqueries[self.idx]))
            pixmap = QtGui.QPixmap()
            pixmap.load(self.img_inqueries[self.idx])
            scene = QtGui.QGraphicsScene(self.graphicsView)
            item = QtGui.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.graphicsView.setSceneRect(scene.sceneRect())
            self.graphicsView.setScene(scene)
            self.caption.setText(QtCore.QString(self.caption_inqueries[self.idx]))

    def press_next_previous(self, previous=True):
        if previous:
            self.idx -= 1
        else:
            self.idx += 1
        self.show_result()


class StyleWindow(QtGui.QDialog, Ui_styleWindow):

    def __init__(self):
        super(StyleWindow, self).__init__()
        self.setupUi(self)
        self.img_path = None
        self.style_path = None
        self.args = {'learning rate': 1e-3, 'content weight': 5.0, 'style weight': 1e4,
                     'tv weight': 1e-3, 'width': 512, 'height': 512, 'max steps': 1000}
        tmp = {'learning rate': (self.learningRate, float), 'content weight': (self.contentWeight, float),
               'style weight': (self.styleWeight, float), 'tv weight': (self.tvWeight, float),
               'width': (self.imgWidth, int), 'height': (self.imgHeight, int), 'max steps': (self.maxSteps, int)}
        for k, v in tmp.items():
            v[0].setText(QtCore.QString(str(self.args[k])))
            change_value(v[0], self.args, k, v[1])
        self.selectFIle.clicked.connect(self.select_img)
        self.selectStyle.clicked.connect(self.select_style)
        self.start.clicked.connect(self.start_transform)

    def select_img(self):
        self.img_path = QtGui.QFileDialog().getOpenFileName(self, 'Select File')
        self.filePath.setText(self.img_path)

    def select_style(self):
        self.style_path = QtGui.QFileDialog().getOpenFileName(self, 'Select Style')
        self.stylePath.setText(self.style_path)

    def start_transform(self):
        if self.img_path is not None:
            img_path = str(self.img_path)
            if self.slow.isChecked() and self.style_path is not None:
                style_path = str(self.style_path)
                if self.method.currentIndex() == 0:
                    neural_style = NeuralStyle(img_width=self.args['width'],
                                               img_height=self.args['height'],
                                               content_weight=self.args['content weight'],
                                               style_weight=self.args['content weight'],
                                               tv_weight=self.args['tv weight'],
                                               SGD=False)
                else:
                    neural_style = NeuralStyle(img_width=self.args['width'],
                                               img_height=self.args['height'],
                                               content_weight=self.args['content weight'],
                                               style_weight=self.args['content weight'],
                                               tv_weight=self.args['tv weight'],
                                               SGD=False,
                                               max_iters=self.args['max steps'],
                                               learning_rate=self.args['learning rate'])
                QtGui.QMessageBox().information(self, 'Information', 'Please wait...')
                neural_style.evaluate(img_path, style_path)
                QtGui.QMessageBox().information(self, 'Information', 'Done')
                self.show_img()
            else:
                neural_style = FastNeuralStyle(img_width=self.args['width'],
                                               img_height=self.args['height'],
                                               content_weight=self.args['content weight'],
                                               style_weight=self.args['content weight'],
                                               tv_weight=self.args['tv weight'],
                                               learning_rate=self.args['learning rate'])
                neural_style.evaluate(img_path)
                self.show_img()

    def show_img(self):
        pixmap = QtGui.QPixmap()
        pixmap.load('./data/img.jpg')
        scene = QtGui.QGraphicsScene(self.graphicsView)
        item = QtGui.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView.setSceneRect(scene.sceneRect())
        self.graphicsView.setScene(scene)


def main():
    app = QtGui.QApplication(sys.argv)
    app.setStyle('fusion')
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()