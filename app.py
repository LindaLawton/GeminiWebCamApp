import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from gemini import sample_generate_text_image_content
import asyncio
from qasync import QEventLoop, asyncSlot

cap = cv2.VideoCapture(0)  # 0 refers to the default webcam


class MainWindow(QWidget):
    def __init__(self, loop=None):
        super().__init__()
        self.initUI()

        # Start the timer in the constructor
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_webcam_feed)
        self.timer.start(50)  # Update the webcam feed every 50 milliseconds

        self.loop = loop or asyncio.get_event_loop()

    def initUI(self):
        hbox = QHBoxLayout(self)

        # webcam frame
        webcam_frame = QFrame(self)
        webcam_frame.setFrameShape(QFrame.StyledPanel)

        # Create a QLabel to display the webcam feed
        self.label = QLabel(webcam_frame)
        self.label.setGeometry(0, 0, 640, 480)

        bottom = QFrame()
        bottom.setFrameShape(QFrame.StyledPanel)

        self.textedit = QTextEdit()
        self.textedit.setFontPointSize(16)
        # Create a button
        button = QPushButton("Send", self)

        splitter2 = QSplitter(Qt.Vertical)
        splitter2.addWidget(self.textedit)
        splitter2.addWidget(button)

        splitter1 = QSplitter(Qt.Horizontal)

        splitter1.addWidget(webcam_frame)
        splitter1.addWidget(splitter2)
        splitter1.setSizes([400, 400])

        hbox.addWidget(splitter1)

        self.setLayout(hbox)
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))

        self.setGeometry(100, 100, 1280, 480)
        self.setWindowTitle('Daimto Gemini video demo')

        # Connect button click event to a function
        button.clicked.connect(self.capture_image_async)

        # self.show()

    @asyncSlot()
    async def capture_image_async(self):
        # Implement your desired capture functionality here
        # e.g., save the current QImage to a file

        try:
            ret, frame = cap.read()

            if ret:
                encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()

                # Read the text from the textedit
                text = self.textedit.toPlainText()

                # Split the text into lines
                lines = text.splitlines()

                # Get the last line
                last_line = lines[-1]

                self.textedit.append("...")
                # TODO make this async at some point
                response = await sample_generate_text_image_content(last_line, encoded_frame)

                self.textedit.append(f"Gemini: {response}")

                return response

        except Exception as e:
            print(e)

    def update_webcam_feed(self):
        ret, frame = cap.read()
        if ret:
            # Convert OpenCV frame to QImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            # Display the QImage on the QLabel
            self.label.setPixmap(QPixmap.fromImage(frame))


def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow(loop)
    window.update_webcam_feed()  # Call the update function once
    window.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
