import sys
from PyQt5 import uic, QtWidgets
import pandas as pd
import math
import LogReg
import SVMFIle
import TreeClass
import RandomForestFile
from threading import Thread
import os.path

Form, _ = uic.loadUiType("Project1.ui")

class Ui(QtWidgets.QDialog, Form):
        def __init__(self):
            super(Ui, self).__init__()
            self.setupUi(self)
          #  self.criterionButton.clicked.connect(self.criterion)
            self.groupButton.clicked.connect(self.group)

        def criterion(self):
            codePation = self.codePatient.text()
            value = Logic().calculateCriterion(codePation)
            self.valueCriterion_2.setText(str(value))
            group = Logic().compare(value)
            self.groupField.setText(group)

        def getRhytmValue(self):
            if  self.rhytmComboBox.currentText() == "Синусовый ритм":
                return 1
            elif self.rhytmComboBox.currentText() == "Фибрилляция предсердий":
                return 2
            else: return 3

        def getChangeRhytmValue(self):
            if self.changeRhytmComboBox.currentText() == "Да":
                return 1
            else:
                return 0

        def getModel(self):
            if self.changeRhytmComboBox.currentText() == "Да":
                return 1
            else:
                return 0

        def group(self):
            age = int(self.age.text())
            sex = 1 if self.sexComboBox.currentText() == "Ж" else 0
            days = int(self.days.text())
          #  criteri = float(self.criteri.text())
            rhytmC = getRhytmValue(self)
            rhytm =  getChangeRhytmValue(self)
            model = self.modelComboBox(self)

            result = Logic().predict(model, age, sex, days, criteri, rhytmC, rhytm)
            self.groupF.setText("С вероятностью " + str(result) + " пациент принадлежит к группе 1")



        '''    if self.logisticRegression.isChecked():
                result = Logic().predict("регрессия", age, sex, days, criteri,rhytmC, rhytm)
                self.groupF.setText("С вероятностью " + str(result) + " пациент принадлежит к группе 1")

            if self.svm.isChecked():
                result = Logic().predict("опорные вектора", age, sex, days, criteri, rhytmC, rhytm)
                self.groupF.setText("С вероятностью" + str(result) + "пациент принадлежит к группе 1")

            if self.tree.isChecked():
                result = Logic().predict("деревья", age, sex, days, criteri, rhytmC, rhytm)
                self.groupF.setText("С вероятностью" + str(result) + "пациент принадлежит к группе 1")

            if self.randomForest.isChecked():
                result = Logic().predict("лес", age, sex, days, criteri, rhytmC, rhytm)
                self.groupF.setText("С вероятностью" + str(result) + "пациент принадлежит к группе 1")'''

class Logic():
        def calculateCriterion(self, codePation):
            table = pd.read_excel("EKG.xlsx")
            QT = table.QT.loc[table['Код пациента'] == int(codePation)].apply((lambda x: x**3)).sum()
            RR = table.RR.loc[table['Код пациента'] == int(codePation)].apply((lambda x: x ** 3)).sum()
            criterion = math.log(RR, QT)
            return criterion

        def compare(self, criterion):
            const = 1.0952899999998708
            if criterion > const:
                return 'Основная группа'
            else: return 'Контрольная группа'

        def predict(self, method, age, sex, days, criteri,rhytmc, rhytm):
            if method == "Логистическая регрессия":
                return LogReg.Predict(age, sex, days, criteri,rhytmc, rhytm)
            if method == "Метод опорных векторов":
                return SVMFIle.Predict(age, sex, days, criteri,rhytmc, rhytm)
            if method == "Дерево решений":
                return TreeClass.Predict(age, sex, days, criteri,rhytmc, rhytm)
            if method == "Случайный лес":
                return RandomForestFile.Predict(age, sex, days, criteri,rhytmc, rhytm)


if __name__ == "__main__":

    def log_uncaught_exceptions(ex_cls, ex, tb):
        text = '{}: {}:\n'.format(ex_cls.__name__, ex)
        import traceback
        text += ''.join(traceback.format_tb(tb))

        print(text)
        QtWidgets.QMessageBox.critical(None, 'Error', text)
        quit()

    sys.excepthook = log_uncaught_exceptions
    if not os.path.isfile('logReg') or not os.path.isfile('SVM') or not os.path.isfile('randomForest') or not os.path.isfile('Tree'):
        thread1 = Thread(target=LogReg.train())
        thread2 = Thread(target=SVMFIle.train())
        thread3 = Thread(target=TreeClass.train())
        thread4 = Thread(target=RandomForestFile.train())

        thread1.start()
        thread2.start()
        thread3.start()
        thread4.start()
        thread1.join()
        thread2.join()
        thread3.join()
        thread4.join()

    app = QtWidgets.QApplication(sys.argv)
    w = Ui()
    w.show()
    sys.exit(app.exec_())
