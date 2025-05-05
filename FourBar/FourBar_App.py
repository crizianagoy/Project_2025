#region imports
from FourBar_GUI import Ui_Form
from FourBarLinkage_MVC import FourBarLinkage_Controller
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import math
import sys
import numpy as np
import scipy as sp
from scipy import optimize
from copy import deepcopy as dc
import logging
#endregion

# Set up logging to capture errors and debugging information
logging.basicConfig(
    filename='../../finalandproject/Project__2025/FourBar/fourbar_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

#region class definitions
class MainWindow(Ui_Form, qtw.QWidget):
    """
    Main window class for the Four-Bar Linkage simulation application.

    Inherits from Ui_Form (auto-generated GUI layout) and QWidget. Handles setup, user interaction,
    and delegates logic to the FourBarLinkage_Controller.
    """

    def __init__(self):
        """Initialize the main window, set up UI widgets and connect logic."""
        super().__init__()
        self.setupUi(self)  # Set up the UI from Qt Designer

        #region UserInterface setup
        # Add spin boxes and labels for dynamic simulation parameters
        self._initControls()

        # Setup the controller with references to key widgets
        widgets = [
            self.gv_Main, self.nud_InputAngle, self.lbl_OutputAngle_Val, self.nud_Link1Length,
            self.nud_Link3Length, self.spnd_Zoom, self.nud_MinAngle, self.nud_MaxAngle,
            self.nud_Damping, self.nud_Mass, self.nud_Spring
        ]
        self.FBL_C = FourBarLinkage_Controller(widgets)

        # Setup graphics and initial linkage configuration
        self.FBL_C.setupGraphics()
        self.FBL_C.buildScene()

        # Store angle values for update logic
        self.prevAlpha = self.FBL_C.FBL_M.InputLink.angle
        self.prevBeta = self.FBL_C.FBL_M.OutputLink.angle

        self.lbl_OutputAngle_Val.setText("{:0.3f}".format(self.FBL_C.FBL_M.OutputLink.AngleDeg()))
        self.nud_Link1Length.setValue(self.FBL_C.FBL_M.InputLink.length)
        self.nud_Link3Length.setValue(self.FBL_C.FBL_M.OutputLink.length)

        # Connect GUI events to controller logic
        self._connectSignals()

        # Install event filter for mouse interactions
        self.FBL_C.FBL_V.scene.installEventFilter(self)
        self.mouseDown = False  # Mouse press state

        self.show()  # Display the application window

    def _initControls(self):
        """Initialize and layout the user input controls on the horizontal layout."""
        # Create and label UI controls
        self._addLabeledSpinbox("nud_MinAngle", "Min Angle", -360, 360, 0)
        self._addLabeledSpinbox("nud_MaxAngle", "Max Angle", -360, 360, 180)
        self._addLabeledSpinbox("nud_Damping", "Damping Coeff", 0, 1000, 50)
        self._addLabeledSpinbox("nud_Mass", "Mass", 0.1, 100, 1)
        self._addLabeledSpinbox("nud_Spring", "Spring Const", 0, 1000, 100)

        # Add buttons for simulation control
        self.btn_Simulate = qtw.QPushButton("Start Simulation", self)
        self.horizontalLayout.addWidget(self.btn_Simulate)

        self.btn_PauseResume = qtw.QPushButton("Pause/Resume", self)
        self.horizontalLayout.addWidget(self.btn_PauseResume)

        self.btn_ResetTracers = qtw.QPushButton("Reset Tracers", self)
        self.horizontalLayout.addWidget(self.btn_ResetTracers)

    def _addLabeledSpinbox(self, name, label, min_val, max_val, default):
        """Helper to create a spinbox with an associated label and insert it into the layout."""
        spinbox = qtw.QDoubleSpinBox(self)
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setObjectName(name)
        setattr(self, name, spinbox)
        label_widget = qtw.QLabel(label, self)
        self.horizontalLayout.addWidget(label_widget)
        self.horizontalLayout.addWidget(spinbox)

    def _connectSignals(self):
        """Connect GUI signals to the appropriate controller update methods."""
        self.spnd_Zoom.valueChanged.connect(self.setZoom)
        self.nud_Link1Length.valueChanged.connect(self.setInputLinkLength)
        self.nud_Link3Length.valueChanged.connect(self.setOutputLinkLength)
        self.nud_MinAngle.valueChanged.connect(self.updateAngleLimits)
        self.nud_MaxAngle.valueChanged.connect(self.updateAngleLimits)
        self.nud_Damping.valueChanged.connect(self.updateDamping)
        self.nud_Mass.valueChanged.connect(self.updateMass)
        self.nud_Spring.valueChanged.connect(self.updateSpring)
        self.btn_Simulate.clicked.connect(self.startSimulation)
        self.btn_PauseResume.clicked.connect(self.pauseResumeSimulation)
        self.btn_ResetTracers.clicked.connect(self.resetTracers)

    # --------- UI Interaction Methods ---------
    def setInputLinkLength(self):
        """Pass new input link length to the controller."""
        self.FBL_C.setInputLinkLength()

    def setOutputLinkLength(self):
        """Pass new output link length to the controller."""
        self.FBL_C.setOutputLinkLength()

    def updateAngleLimits(self):
        """Ensure valid angle bounds and update model limits accordingly."""
        try:
            min_angle, max_angle = self.nud_MinAngle.value(), self.nud_MaxAngle.value()
            if min_angle > max_angle:
                min_angle, max_angle = max_angle, min_angle
                self.nud_MinAngle.setValue(min_angle)
                self.nud_MaxAngle.setValue(max_angle)
            self.FBL_C.setAngleLimits(min_angle, max_angle)
        except Exception as e:
            logging.error(f"Error in updateAngleLimits: {str(e)}")

    def updateDamping(self):
        try:
            self.FBL_C.setDampingCoefficient(self.nud_Damping.value())
        except Exception as e:
            logging.error(f"Error in updateDamping: {str(e)}")

    def updateMass(self):
        try:
            self.FBL_C.setMass(self.nud_Mass.value())
        except Exception as e:
            logging.error(f"Error in updateMass: {str(e)}")

    def updateSpring(self):
        try:
            self.FBL_C.setSpringConstant(self.nud_Spring.value())
        except Exception as e:
            logging.error(f"Error in updateSpring: {str(e)}")

    def startSimulation(self):
        """Initialize simulation with user-defined parameters, warn if underdamped."""
        try:
            initial_angle = self.nud_InputAngle.value()
            m, k, c = self.nud_Mass.value(), self.nud_Spring.value(), self.nud_Damping.value()
            zeta = c / (2 * math.sqrt(k * m))  # Damping ratio
            if zeta < 0.5:
                qtw.QMessageBox.warning(
                    self, "Simulation Warning",
                    f"The system may oscillate excessively (damping ratio = {zeta:.2f}). "
                    "Try increasing damping or decreasing spring constant."
                )
            self.FBL_C.startSimulation(initial_angle, m, k, c)
        except Exception as e:
            logging.error(f"Error in startSimulation: {str(e)}")

    def pauseResumeSimulation(self):
        """Toggle simulation between paused and running states."""
        try:
            self.FBL_C.pauseResumeSimulation()
            self.btn_PauseResume.setText("Pause" if self.FBL_C.is_simulation_running else "Resume")
        except Exception as e:
            logging.error(f"Error in pauseResumeSimulation: {str(e)}")

    def resetTracers(self):
        """Reset all visual tracer paths in the simulation."""
        try:
            self.FBL_C.resetTracers()
        except Exception as e:
            logging.error(f"Error in resetTracers: {str(e)}")

    def setZoom(self):
        """Scale the graphics view for zoom effect."""
        try:
            self.gv_Main.resetTransform()
            self.gv_Main.scale(self.spnd_Zoom.value(), self.spnd_Zoom.value())
        except Exception as e:
            logging.error(f"Error in setZoom: {str(e)}")

    def mouseMoveEvent(self, a0: qtg.QMouseEvent):
        """Display cursor coordinates in the window title."""
        try:
            w = app.widgetAt(a0.globalPos())
            name = 'none' if w is None else w.objectName()
            self.setWindowTitle(f"{a0.x()},{a0.y()} {name}")
        except Exception as e:
            logging.error(f"Error in mouseMoveEvent: {str(e)}")

    def eventFilter(self, obj, event):
        """Custom event handler for scene interactions like drag, zoom, and click."""
        try:
            if obj == self.FBL_C.FBL_V.scene:
                if event.type() == qtc.QEvent.GraphicsSceneMouseMove:
                    scenePos = event.scenePos()
                    self.setWindowTitle(f"screen x = {event.screenPos().x()}, screen y = {event.screenPos().y()}: "
                                        f"scene x = {scenePos.x():.2f}, scene y = {scenePos.y():.2f}")
                    if self.mouseDown:
                        self.FBL_C.moveLinkage(scenePos)
                elif event.type() == qtc.QEvent.GraphicsSceneWheel:
                    if event.delta() > 0:
                        self.spnd_Zoom.stepUp()
                    else:
                        self.spnd_Zoom.stepDown()
                elif event.type() == qtc.QEvent.GraphicsSceneMousePress and event.button() == qtc.Qt.LeftButton:
                    self.mouseDown = True
                elif event.type() == qtc.QEvent.GraphicsSceneMouseRelease:
                    self.mouseDown = False
                    self.startSimulation()
            return super(MainWindow, self).eventFilter(obj, event)
        except Exception as e:
            logging.error(f"Error in eventFilter: {str(e)}")
            return False
#endregion

#region function calls
if __name__ == '__main__':
    """Entry point for the Four-Bar Linkage GUI application."""
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    mw.setWindowTitle('Four Bar Linkage')
    sys.exit(app.exec())
#endregion
