import sys
import numpy as np
import os.path
import cv2
import math
import os
import pandas as pd
from PyQt5.QtWidgets import QSpinBox, QApplication, QFileDialog, QMessageBox, QTableView, QWidget, QGridLayout, QLineEdit, QPushButton, \
    QLabel, QProgressBar, QCheckBox
import numpy as np
from pathlib import Path

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle('Droplets segmentation')
        self.setGeometry(100, 100, 400, 100)

        self.prog_bar = QProgressBar(self)
        self.prog_bar.setGeometry(50, 50, 300, 30)
        self.prog_bar.setMinimum(0)

        layout = QGridLayout()
        self.setLayout(layout)

        # directory selection
        dir_btn = QPushButton('Open')
        dir_btn.clicked.connect(self.open_dir_dialog)
        self.dir_name_edit = QLineEdit()

        # input parameters

        self.spinbox1 = QSpinBox()
        self.spinbox1.valueChanged.connect(self.Set_parameters1)
        self.blurr_block = QLineEdit()

        self.spinbox2 = QSpinBox()
        self.spinbox2.valueChanged.connect(self.Set_parameters2)
        self.block_size = QLineEdit()

        self.spinbox3 = QSpinBox()
        self.spinbox3.valueChanged.connect(self.Set_parameters3)
        self.C = QLineEdit()


        start_btn = QPushButton('Calculate')
        start_btn.clicked.connect(self.droplets_segmentation)

        layout.addWidget(QLabel('Folder:'), 1, 0)
        layout.addWidget(self.dir_name_edit, 1, 1)
        layout.addWidget(dir_btn, 1, 2)
        layout.addWidget(start_btn, 5, 2)
        layout.addWidget(QLabel('Progress:'), 5, 0)
        layout.addWidget(self.prog_bar, 5, 1)

        layout.addWidget(self.spinbox1, 2, 1)
        layout.addWidget(self.blurr_block, 2, 2)
        layout.addWidget(self.spinbox2, 3, 1)
        layout.addWidget(self.block_size, 3, 2)
        layout.addWidget(self.spinbox3, 4, 1)
        layout.addWidget(self.C, 4, 2)

        layout.addWidget(QLabel('blurr block size:'), 2, 0)
        layout.addWidget(QLabel('Adaptive threshold block size:'), 3, 0)
        layout.addWidget(QLabel('C:'), 4, 0)

        self.show()

    def open_dir_dialog(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Choose folder")
        if dir_name:
            path = Path(dir_name)
            self.dir_name_edit.setText(str(path))

    def Set_parameters1(self):
        value = self.spinbox1.value()
        self.blurr_block.setText(str(value))

    def Set_parameters2(self):
        value = self.spinbox2.value()
        self.block_size.setText(str(value))

    def Set_parameters3(self):
        value = self.spinbox3.value()
        self.C.setText(str(value))

    def droplets_segmentation(self):
        self.prog_bar.setValue(0)
        dir_photo = self.dir_name_edit.text()
        m = int(self.blurr_block.text())
        blurr_block = (m, m)
        #block_size = self.block_size
        self.check_dir(dir_photo)
        total_file = sum(len(files) for address, dirs, files in os.walk(dir_photo))
        self.prog_bar.setMaximum(total_file - 2)


        def contrast_and_blurr(img, blurr_block):
            """
            INPUT:
            img - unprocessed RGB image
            OUTPUT: gray_cs - gray normalized and blurred image
            """
            # turn to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # substract noise
            gray = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
            # blur
            gray = cv2.GaussianBlur(gray, blurr_block, 0)
            # normalization
            gray_cs = cv2.normalize(gray, None, 0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            gray_cs = cv2.GaussianBlur(gray_cs, blurr_block, 0)
            #print(gray_cs)

            return gray_cs

        def calculate_drops_area(img, photo_number, blockSize, C):
            '''
            INPUT:
            img - preprocessed image
            blockSize - int, odd numbers only, e.x.: 3, 5, 7, ... - pixel neighborhood size for binary segmentation
            C - int, constant to be substracted before binary segmentation

            OUTPUT:
            drop_areas - np.array of shape(k) with calculated areas for each drop
            circularity - np.array of shape(k) with circularity parameters calculated for each drop
            '''

            img = img.astype(np.uint8)
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize,
                                           C)  # cv2.ADAPTIVE_THRESH_GAUSSIAN_C cv2.ADAPTIVE_THRESH_MEAN_C
            # (T, thresh) = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)# | cv2.THRESH_OTSU)
            thresh = thresh.astype(np.uint8)
            cv2.imwrite('processed_photos_tr1_ph4/' + str(photo_number) + 'bw' + '.jpg', thresh)

            # detect contours in the mask
            cnts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
            f = img.shape[0]
            l = img.shape[1]
            contours = np.zeros([f, l])

            # show_image(cnts, 'cnts')
            drop_areas = []
            circularity = []

            for k in range(len(cnts)):

                area = cv2.contourArea(cnts[k])
                length = cv2.arcLength(cnts[k], True)

                if (area > 7.0) and (area < 300.0):
                    circ = 4 * math.pi * area / (length ** 2.0)

                    if circ > 0.5:
                        for coords in cnts[k]:
                            contours[coords[0][1], coords[0][0]] = 255
                        drop_areas.append(area)
                        circularity.append(circ)

            contours = contours.astype(np.uint8)

            return drop_areas, circularity

        def calculate_for_all_photos(address, files, blockSize, C, blurr_block):
            '''
            INPUT:
            photo_number - int, number of photos
            blockSize - int, odd numbers only, e.x.: 3, 5, 7, ... - pixel neighborhood size for binary segmentation
            C - int, constant to be substracted before binary segmentation

            OUTPUT:
            photo_numbers - np.array of size (k), photo numbers of each processed drop
            area_all - np.array of size (k), area values for all photos
            circularity_all - np.array of size (k), circularity values of all drops
            '''
            area_all = []
            circularity_all = []
            photo_numbers = []
            n = 0
            for file in files:
                n += 1
                full_path_img = os.path.join(address, file)
                print(n, full_path_img)
                img = cv2.imread(full_path_img)
                gray_cs = contrast_and_blurr(img, blurr_block)
                print(gray_cs)
                cv2.imwrite('processed_photos_tr1_ph4/' + str(n) + 'contr' + '.jpg', gray_cs)

                areas, circularity = calculate_drops_area(gray_cs, n, blockSize, C)
                photo_numbers = np.concatenate((photo_numbers, n * np.ones(len(areas))))
                circularity_all = np.concatenate((circularity_all, circularity))
                area_all = np.concatenate((area_all, areas))

                print('area, pxl:', areas)
                print('circularity:', circularity)

            return photo_numbers, area_all, circularity_all

        with open('submission.csv', 'w', newline='') as csvfile:
            for address, dirs, files in os.walk(dir_photo):
                for name in files:
                    value = self.prog_bar.value()
                    self.prog_bar.setValue(value + 1)
                    full_path_img = os.path.join(address, name)

            for address, dirs, files in os.walk(dir_photo):
                value = self.prog_bar.value()
                self.prog_bar.setValue(value + 1)

                blockSize = 13  # should be an odd number: 3, 5, 7,...
                C = -2  # a value to be substracted
                #blurr_block = (13, 13)

                if not os.path.exists('processed_photos_tr1_ph4'): os.mkdir('processed_photos_tr1_ph4')

                photo_numbers, area, circularity = calculate_for_all_photos(address, files, blockSize, C,
                                                                            blurr_block)

                df = pd.DataFrame({'ind': photo_numbers, 'area': area, 'circularity': circularity}, index=None)
                df1 = df.groupby('ind').mean()

                df.to_csv('submission.csv', index=False)

        dlg_complite = QMessageBox(self)
        dlg_complite.setWindowTitle("Recognition is successful!")
        dlg_complite.setText(
            f"{total_file} files were processed! Results were saved at submission.csv")
        button = dlg_complite.exec()

    def check_dir(self, dir_photo):
        if not dir_photo:
            dlg_no_dir = QMessageBox(self)
            dlg_no_dir.setWindowTitle("Attention!")
            dlg_no_dir.setText("Empty path!")
            button = dlg_no_dir.exec()
        elif not os.path.isdir(dir_photo):
            dlg_no_is_dir = QMessageBox(self)
            dlg_no_is_dir.setWindowTitle("Attention!")
            dlg_no_is_dir.setText("Folder does not exist!")
            button = dlg_no_is_dir.exec()
        elif not os.listdir(dir_photo):
            dlg_no_listdir = QMessageBox(self)
            dlg_no_listdir.setWindowTitle("Attention!")
            dlg_no_listdir.setText("Selected folder has no files!")
            button = dlg_no_listdir.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
