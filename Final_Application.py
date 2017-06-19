import os  # importing the different python modules for future programming
import sqlite3

import cv2
import cv2.face
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

print("Application is online")  # This Application is developed using the pyqt5 GUI developer software called designer


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(548, 218)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.calendarWidget = QtWidgets.QCalendarWidget(self.centralwidget)
        self.calendarWidget.setGeometry(QtCore.QRect(280, 0, 272, 149))
        self.calendarWidget.setObjectName("calendarWidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 261, 16))
        self.label.setFrameShape(QtWidgets.QFrame.Panel)

        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 30, 111, 16))
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(160, 30, 111, 22))
        self.pushButton.setObjectName("pushButton")

        self.pushButton.clicked.connect(self.detect)

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(0, 60, 151, 16))
        self.label_3.setObjectName("label_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(160, 60, 111, 22))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_2.clicked.connect(self.dataset)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, 90, 141, 16))
        self.label_4.setObjectName("label_4")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(160, 90, 111, 22))
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_3.clicked.connect(self.train)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(0, 120, 141, 16))
        self.label_5.setObjectName("label_5")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(160, 120, 111, 22))
        self.pushButton_4.setObjectName("pushButton_4")

        self.pushButton_4.clicked.connect(self.recognize)

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(0, 150, 141, 16))
        self.label_6.setObjectName("label_6")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(160, 150, 111, 22))
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton_5.clicked.connect(self.send_mail)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 548, 19))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.triggered.connect(QApplication.quit)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def detect(self):
        print("starting the detection program ")
        # location of cascade classfier
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        cam = cv2.VideoCapture(0)  # capture the frames
        # running the loop again and again
        while True:
            ret, img = cam.read()  # return the frames the cam
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the img to gray
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:  # draw rectangle on the detected face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("img", img)  # showing the img

            # 1/waitkey = number of frames persecond
            if cv2.waitKey(30) & 0xFF == ord("c"):
                break
        cam.release()  # release the camera
        cv2.destroyAllWindows()  # destroying the all open windows
        print("Ending the face detection")

        # dataset program function

    def dataset(self):
        print("Creating pictures for dataset")

        faceDetect = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(0)

        # function for accessing the student database(sqlite3)
        def insertOrUpdate(Id, Name):
            conn = sqlite3.connect("Face_Reg/Facebase.db")
            cmd = "SELECT * FROM students WHERE ID=" + str(Id)
            cursor = conn.execute(cmd)
            isRecordExist = 0
            for row in cursor:
                isRecordExist = 1
            if isRecordExist == 1:  # ID already present then update
                cmd = "UPDATE students SET Name=" + str(Name) + " WHERE ID=" + str(Id)
            else:  # Insert new ID
                cmd = "INSERT INTO students(ID,Name) Values(" + str(Id) + "," + str(Name) + ")"
            conn.execute(cmd)
            conn.commit()
            conn.close()

        id = int(input("Enter user id no: "))
        name = str(input("Enter name: "))
        insertOrUpdate(id, name)
        sampleno = 0

        while True:
            ret, img = cam.read()
            faces = faceDetect.detectMultiScale(img, 1.3, 5)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            norm = cv2.equalizeHist(gray)  # equalize the intensity
            for (x, y, w, h) in faces:  # cut the faces and store it
                sampleno += 1
                w_rm = int(0.2 * w / 2)
                cv2.imwrite("dataSet/User." + str(id) + "." + str(sampleno) + ".jpg",
                            norm[y:y + h, x + w_rm:x + w - w_rm])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # 1/waitkey = number of frames persecond
                cv2.waitKey(100)
            cv2.waitKey(1)
            cv2.imshow("face", img)
            if sampleno > 100:  # no of samples
                break
        cam.release()
        cv2.destroyAllWindows()
        print("Images are taken")

        # train program function

    def train(self):
        print("Train the images for future recognization")

        recognizer = cv2.face.createLBPHFaceRecognizer()
        path = "dataSet"  # location of images to be trained

        def getImageswithID(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            IDs = []
            for imagePath in imagePaths:
                faceImg = Image.open(imagePath).convert("L")
                facenp = np.array(faceImg, "uint8")  # convert the image into array
                ID = int(os.path.split(imagePath)[-1].split(".")[1])
                faces.append(facenp)
                IDs.append(ID)
                cv2.imshow("training images", facenp)  # showing the images which are to be train
                cv2.waitKey(10)
            return np.array(IDs), faces  # return the images as numpy array

        IDs, faces = getImageswithID(path)
        recognizer.train(faces, np.array(IDs))
        recognizer.save("recognizer/trainningData.yml")  # convert the img into dataset repository
        cv2.destroyAllWindows()
        print("Training is over")

        # recognize program function

    def recognize(self):
        print("Finding the faces... ")

        def make_directory(directory):
            if not os.path.exists(directory):
                print("creating project " + directory)
                os.makedirs(directory)

        make_directory("Attendance")

        def make_data_files(project_name):
            queue = project_name + "/attendance.txt"
            if not os.path.isfile(queue):
                write_file(queue, "")

        def write_file(path, data):
            f = open(path, "w")
            f.write(data)
            f.close

        make_data_files("Attendance")

        def append_to_file(path, data):
            with open(path, "a") as file:
                file.write(data + "\n")

        def resetdatabase():
            conn = sqlite3.connect("Face_Reg/Facebase.db")
            cmd = "UPDATE students SET presentORnot=" + str(0)
            conn.execute(cmd)
            conn.commit()
            conn.close()

        def update_presentORnot(id):
            conn = sqlite3.connect("Face_Reg/Facebase.db")
            cmd = "UPDATE students SET presentORnot=" + str(1) + " WHERE ID=" + str(id)
            conn.execute(cmd)
            conn.commit()
            conn.close()

        def write_in_file():
            conn = sqlite3.connect("Face_Reg/Facebase.db")
            cmd = "SELECT ID,Name,Rollno FROM students WHERE presentORnot=" + str(1)
            cursor = conn.execute(cmd)
            profile = None
            for row in cursor:
                profile = row
            conn.close()
            return str(profile)

        # access the database and displays the information for the respective ID
        def getprofiles(id):
            conn = sqlite3.connect(
                "Facebase.db")  # connect to the database
            cmd = "SELECT * FROM students WHERE ID=" + str(id)
            cursor = conn.execute(cmd)
            profile = None
            for row in cursor:
                profile = row
            conn.close()
            return profile

        faceDetect = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(0)
        rec = cv2.face.createLBPHFaceRecognizer()  # LBPH face recognizer for detecting the trained faces
        rec.load(
            "trainningData.yml")  # load the trained data
        id = 0
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for display the information
        while cam.isOpened():
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                collecter = cv2.face.MinDistancePredictCollector()  # finding the minimum distance
                rec.predict(gray[y:y + h, x:x + w], collecter)  # give the collector to the predictor
                conf = collecter.getDist()  # from collector get confidence
                pre = collecter.getLabel()  # from confidence get the labels
                update_presentORnot(pre)
                profile = getprofiles(pre)
                # q = round(conf, 1)
                # print(q)
                threshold = 55  # applying the threshold value
                if conf < threshold:  # checks the confidence level
                    if profile is not None:
                        cv2.putText(img, str(profile[1]), (x, y + w + 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(img, str(profile[2]), (x, y + w + 60), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(img, str(profile[3]), (x, y + w + 90), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, str("unknown"), (x, y + w + 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("face", img)
            # 1/waitkey = number of frames persecond
            if cv2.waitKey(1) & 0xff == ord("c"):
                break
        append_to_file("Attendance/attendance.txt", write_in_file())
        resetdatabase()
        cam.release()
        cv2.destroyAllWindows()
        print("Ends the program")

    def send_mail(self):
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        fromaddr = "your email"
        toaddr = "your email"

        username = "email name"
        password = "password"
        filename = 'Attendance/attendance.txt'
        f = open(filename)
        msg = MIMEMultipart()
        attachment = MIMEText(f.read())
        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = 'Test5'
        msg.attach(attachment)
        server = smtplib.SMTP('smtp.gmail.com', 25)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(username, password)
        server.sendmail(fromaddr, toaddr, msg.as_string())
        server.quit()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Attendance Management Application"))
        self.label_2.setText(_translate("MainWindow", "1.Detect the face"))
        self.pushButton.setText(_translate("MainWindow", "Detect"))
        self.label_3.setText(_translate("MainWindow", "2.Take pic\'s for dataset"))
        self.pushButton_2.setText(_translate("MainWindow", "Dataset Creater"))
        self.label_4.setText(_translate("MainWindow", "3.Training the Dataset"))
        self.pushButton_3.setText(_translate("MainWindow", "Training data"))
        self.label_5.setText(_translate("MainWindow", "4.Recognize the  face"))
        self.pushButton_4.setText(_translate("MainWindow", "Recognize"))
        self.label_6.setText(_translate("MainWindow", "5.Mail the Text"))
        self.pushButton_5.setText(_translate("MainWindow", "Mail"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionExit.setText(_translate("MainWindow", "exit"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
