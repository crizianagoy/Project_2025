#region imports
import PyQt5.QtGui as qtg
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import math
from scipy import optimize
from copy import deepcopy as dc
from scipy.integrate import odeint
import numpy as np
import logging
#endregion

# Set up logging to capture errors and debugging information
logging.basicConfig(
    filename='fourbar_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

#region four bar linkage classes in MVC Pattern
"""
The Model-View-Controller (MVC) design pattern divides responsibilities between:
- Model: Retains the state of the four-bar linkage.
- View: Displays the state of the linkage in the UI.
- Controller: Mediates updates between the model and view based on user interactions.
"""

#region model and supporting classes
class RigidLink(qtw.QGraphicsItem):
    """A rigid link in the four-bar linkage, represented as a graphical item.

    Attributes:
        stPt (QPointF): Starting point of the link.
        enPt (QPointF): Ending point of the link.
        length (float): Length of the link.
        angle (float): Angle of the link in radians.
        radius (float): Radius for drawing the link's ends.
        mass (float): Mass of the link.
        pen (QPen): Pen for drawing the link.
        brush (QBrush): Brush for filling the link.
        name (str): Name of the link (e.g., "Input", "Coupler").
        label_pen (QPen): Pen for drawing the label.
        rect (QRectF): Bounding rectangle for the link.
        transform (QTransform): Transformation for positioning the link.
    """

    def __init__(self, stX=0, stY=0, enX=1, enY=1, radius=10, mass=10, parent=None, pen=None, brush=None, name='RigidLink',
                 label_pen=qtg.QPen(qtc.Qt.black)):
        """Initialize a rigid link with start and end points.

        Args:
            stX (float): X-coordinate of the start point.
            stY (float): Y-coordinate of the start point.
            enX (float): X-coordinate of the end point.
            enY (float): Y-coordinate of the end point.
            radius (float): Radius for drawing the ends of the link.
            mass (float): Mass of the link.
            parent (QGraphicsItem, optional): Parent item.
            pen (QPen, optional): Pen for drawing the link.
            brush (QBrush, optional): Brush for filling the link.
            name (str): Name of the link.
            label_pen (QPen): Pen for drawing the label.
        """
        super().__init__(parent)
        self.pen = pen
        self.label_pen = label_pen
        self.brush = brush
        self.name = name
        self.stPt = qtc.QPointF(stX, stY)
        self.enPt = qtc.QPointF(enX, enY)
        self.radius = radius
        self.mass = mass
        self.angle = self.linkAngle()
        self.rect = qtc.QRectF(-self.radius, -self.radius, self.length + self.radius, self.radius)
        self.transform = qtg.QTransform()
        self.transform.reset()

    def boundingRect(self):
        """Return the bounding rectangle of the link after transformation.

        Returns:
            QRectF: The transformed bounding rectangle.
        """
        boundingRect = self.transform.mapRect(self.rect)
        return boundingRect

    def deltaY(self):
        """Calculate the vertical distance between start and end points.

        Returns:
            float: The difference in y-coordinates.
        """
        self.DY = self.enPt.y() - self.stPt.y()
        return self.DY

    def deltaX(self):
        """Calculate the horizontal distance between start and end points.

        Returns:
            float: The difference in x-coordinates.
        """
        self.DX = self.enPt.x() - self.stPt.x()
        return self.DX

    def linkLength(self):
        """Calculate the length of the link.

        Returns:
            float: The Euclidean distance between start and end points.
        """
        self.length = math.sqrt(math.pow(self.deltaX(), 2) + math.pow(self.deltaY(), 2))
        return self.length

    def linkAngle(self):
        """Calculate the angle of the link relative to the horizontal.

        Returns:
            float: The angle in radians.
        """
        self.linkLength()
        if self.length == 0.0:
            self.angle = 0
        else:
            self.angle = math.acos(self.DX / self.length)
            self.angle *= -1 if (self.DY > 0) else 1  # Adjust sign based on y-direction
        self.rangeAngle()
        return self.angle

    def rangeAngle(self):
        """Normalize the angle to the range [0, 2π)."""
        while (self.angle < 0):
            self.angle += 2 * math.pi
        while (self.angle > 2 * math.pi):
            self.angle -= 2 * math.pi

    def AngleDeg(self):
        """Convert the link angle to degrees.

        Returns:
            float: The angle in degrees.
        """
        return self.angle * 180 / math.pi

    def paint(self, painter, option, widget=None):
        """Draw the link with its label and pivot points.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        path = qtg.QPainterPath()
        len = self.linkLength()
        angLink = self.linkAngle() * 180 / math.pi
        rectSt = qtc.QRectF(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)
        rectEn = qtc.QRectF(self.length - self.radius, -self.radius, 2 * self.radius, 2 * self.radius)
        centerLinePen = qtg.QPen()
        centerLinePen.setStyle(qtc.Qt.DashDotLine)
        r, g, b, a = self.pen.color().getRgb()
        centerLinePen.setColor(qtg.QColor(r, g, b, 128))
        centerLinePen.setWidth(1)
        p1 = qtc.QPointF(0, 0)
        p2 = qtc.QPointF(len, 0)
        painter.setPen(centerLinePen)
        painter.drawLine(p1, p2)  # Draw the centerline of the link
        path.arcMoveTo(rectSt, 90)
        path.arcTo(rectSt, 90, 180)
        path.lineTo(self.length, self.radius)
        path.arcMoveTo(rectEn, 270)
        path.arcTo(rectEn, 270, 180)
        path.lineTo(0, -self.radius)
        if self.pen is not None:
            painter.setPen(self.pen)
        if self.label_pen is None:
            self.label_pen = qtg.QPen(qtg.QColor('red'))
        if self.brush is not None:
            painter.setBrush(self.brush)
        painter.drawPath(path)  # Draw the link shape
        pivotStart = qtc.QRectF(-self.radius / 6, -self.radius / 6, self.radius / 3, self.radius / 3)
        pivotEnd = qtc.QRectF(self.length - self.radius / 6, -self.radius / 6, self.radius / 3, self.radius / 3)
        painter.drawEllipse(pivotStart)  # Draw pivot at start
        painter.drawEllipse(pivotEnd)  # Draw pivot at end
        self.rect = qtc.QRectF(-self.radius, -self.radius, self.length + 2 * self.radius, 2 * self.radius)
        painter.setBrush(qtg.QBrush(qtc.Qt.black))
        painter.setPen(self.label_pen)
        painter.setFont(qtg.QFont("Times", self.radius))
        painter.drawText(self.rect, qtc.Qt.AlignCenter, self.name)  # Draw the link name
        self.transform.reset()
        self.transform.translate(self.stPt.x(), self.stPt.y())
        self.transform.rotate(-angLink)  # Rotate to align with the link angle
        self.setTransform(self.transform)
        self.transform.reset()
        # Set tooltip with link details
        stTT = self.name + "\nstart: ({:0.3f}, {:0.3f})\nend:({:0.3f},{:0.3f})\nlength: {:0.3f}\nangle: {:0.3f}".format(
            self.stPt.x(), self.stPt.y(), self.enPt.x(), self.enPt.y(), self.length, self.angle * 180 / math.pi)
        self.setToolTip(stTT)

class RigidPivotPoint(qtw.QGraphicsItem):
    """A pivot point in the four-bar linkage, represented as a graphical item.

    Attributes:
        x (float): X-coordinate of the pivot.
        y (float): Y-coordinate of the pivot.
        pt (QPointF): Position of the pivot.
        height (float): Height of the pivot graphic.
        width (float): Width of the pivot graphic.
        radius (float): Radius for drawing the pivot circle.
        rect (QRectF): Bounding rectangle for the pivot.
        rotationAngle (float): Rotation angle of the pivot.
        name (str): Name of the pivot (e.g., "RP 0").
        pen (QPen): Pen for drawing the pivot.
        brush (QBrush): Brush for filling the pivot.
        transformation (QTransform): Transformation for positioning the pivot.
    """

    def __init__(self, ptX=0, ptY=0, pivotHeight=10, pivotWidth=10, parent=None, pen=None, brush=None, rotation=0,
                 name='RigidPivotPoint', label_pen=None):
        """Initialize a pivot point with position and dimensions.

        Args:
            ptX (float): X-coordinate of the pivot.
            ptY (float): Y-coordinate of the pivot.
            pivotHeight (float): Height of the pivot graphic.
            pivotWidth (float): Width of the pivot graphic.
            parent (QGraphicsItem, optional): Parent item.
            pen (QPen, optional): Pen for drawing the pivot.
            brush (QBrush, optional): Brush for filling the pivot.
            rotation (float): Initial rotation angle in degrees.
            name (str): Name of the pivot.
            label_pen (QPen, optional): Pen for drawing the label.
        """
        super().__init__(parent)
        self.x = ptX
        self.y = ptY
        self.pt = qtc.QPointF(ptX, ptY)
        self.pen = pen if pen is not None else qtg.QPen(qtg.QColor('black'))
        self.brush = brush
        self.height = pivotHeight
        self.width = pivotWidth
        self.radius = min(self.height, self.width) / 4
        self.rect = qtc.QRectF(self.x - self.width / 2, self.y - self.radius, self.width, self.height + self.radius)
        self.rotationAngle = rotation
        self.name = name
        self.transformation = qtg.QTransform()
        stTT = self.name + "\nx={:0.3f}, y={:0.3f}".format(self.x, self.y)
        self.setToolTip(stTT)

    def boundingRect(self):
        """Return the bounding rectangle of the pivot after transformation.

        Returns:
            QRectF: The transformed bounding rectangle.
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def rotate(self, angle):
        """Rotate the pivot by the specified angle.

        Args:
            angle (float): Angle in degrees.
        """
        self.rotationAngle = angle

    def paint(self, painter, option, widget=None):
        """Draw the pivot point with its label and support structure.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        path = qtg.QPainterPath()
        radius = min(self.height, self.width) / 2
        name = self.name
        H = math.sqrt(math.pow(self.width / 2, 2) + math.pow(self.height, 2))
        phi = math.asin(radius / H)
        theta = math.asin(self.height / H)
        ang = math.pi - phi - theta
        l = H * math.cos(phi)
        x1 = self.width / 2
        y1 = self.height
        path.moveTo(x1, y1)
        x2 = l * math.cos(ang)
        y2 = l * math.sin(ang)
        path.lineTo(x1 + x2, y1 - y2)
        pivotRect = qtc.QRectF(-radius, -radius, 2 * radius, 2 * radius)
        stAng = math.pi / 2 - phi - theta
        spanAng = math.pi - 2 * stAng
        path.arcTo(pivotRect, stAng * 180 / math.pi, spanAng * 180 / math.pi)
        x4 = -self.width / 2
        y4 = +self.height
        path.lineTo(x4, y4)
        if self.pen is not None:
            painter.setPen(self.pen)
        if self.brush is not None:
            painter.setBrush(self.brush)
        painter.drawPath(path)  # Draw the pivot shape
        pivotPtRect = qtc.QRectF(-radius / 4, -radius / 4, radius / 2, radius / 2)
        painter.drawEllipse(pivotPtRect)  # Draw the pivot point
        x5 = -self.width
        x6 = +self.width
        painter.drawLine(x5, y4, x6, y4)  # Draw the base line
        penOutline = qtg.QPen(qtc.Qt.NoPen)
        hatchbrush = qtg.QBrush(qtc.Qt.BDiagPattern)
        brushTransform = qtg.QTransform()
        brushTransform.scale(0.5, 0.5)
        hatchbrush.setTransform(brushTransform)
        painter.setPen(penOutline)
        painter.setBrush(hatchbrush)
        painter.setFont(qtg.QFont("Arial", 3))
        support = qtc.QRectF(x5, y4, self.width * 2, self.height)
        painter.drawRect(support)  # Draw the support structure
        painter.setBrush(self.brush)
        painter.setPen(self.pen)
        painter.drawText(support, qtc.Qt.AlignCenter, name)  # Draw the pivot name
        self.rect = qtc.QRectF(-self.width, -self.radius, self.width * 2, self.height * 2 + self.radius)
        self.transformation.reset()
        self.transformation.translate(self.x, self.y)
        self.transformation.rotate(self.rotationAngle)
        self.setTransform(self.transformation)
        self.transformation.reset()

class Tracer(qtw.QGraphicsItem):
    """A tracer that tracks the path of a point in the four-bar linkage.

    Attributes:
        pts (list[QPointF]): List of points in the tracer path.
        rect (QRectF): Bounding rectangle for the tracer.
        pen (QPen): Pen for drawing the tracer path.
        penOutline (QPen): Pen for drawing the current point marker.
    """

    def __init__(self, x=0, y=0, pen=None, penOutline=qtg.QPen(qtc.Qt.black)):
        """Initialize a tracer starting at the given position.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            pen (QPen, optional): Pen for drawing the path.
            penOutline (QPen, optional): Pen for drawing the current point marker.
        """
        super().__init__()
        self.pts = [qtc.QPointF(x, y)]
        self.rect = qtc.QRectF(-5, -5, 10, 10)
        self.pen = pen
        self.penOutline = penOutline

    def boundingRect(self):
        """Return the bounding rectangle of the tracer.

        Returns:
            QRectF: The bounding rectangle.
        """
        bounding_rect = self.rect
        return bounding_rect

    def lastPt(self):
        """Return the last point in the tracer path.

        Returns:
            QPointF: The last point.
        """
        return self.pts[len(self.pts) - 1]

    def paint(self, painter, option, widget=None):
        """Draw the tracer path and a marker at the current point.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        if self.pen is not None:
            painter.setPen(self.pen)
        path = qtg.QPainterPath()
        if len(self.pts) > 0:
            path.moveTo(self.pts[0])
        for i in range(1, len(self.pts)):
            path.lineTo(self.pts[i])
        painter.drawPath(path)  # Draw the tracer path
        pt = self.pts[len(self.pts) - 1]
        painter.setPen(self.penOutline)
        painter.drawEllipse(qtc.QRectF(pt.x() - 2.5, pt.y() - 2.5, 5, 5))  # Draw the current point marker

class LinearSpring(qtw.QGraphicsItem):
    """A linear spring connecting two points in the four-bar linkage.

    Attributes:
        stPt (QPointF): Starting point of the spring.
        enPt (QPointF): Ending point of the spring.
        freeLength (float): Natural length of the spring.
        length (float): Current length of the spring.
        DL (float): Displacement from the free length.
        centerPt (QPointF): Midpoint of the spring.
        force (float): Current force exerted by the spring.
        coilsWidth (float): Width of the spring coils.
        coilsLength (float): Length of the spring coils.
        rect (QRectF): Bounding rectangle for the spring.
        pen (QPen): Pen for drawing the spring.
        name (str): Name of the spring.
        label (str, optional): Additional label text.
        k (float): Spring constant.
        nCoils (int): Number of coils in the spring.
        transformation (QTransform): Transformation for positioning the spring.
    """

    def __init__(self, ptSt=qtc.QPointF(0, 0), ptEn=qtc.QPointF(1, 1), coilsWidth=10, coilsLength=30, parent=None,
                 pen=None, name='Spring', label=None, k=10, nCoils=6):
        """Initialize a linear spring with start and end points.

        Args:
            ptSt (QPointF): Starting point of the spring.
            ptEn (QPointF): Ending point of the spring.
            coilsWidth (float): Width of the spring coils.
            coilsLength (float): Length of the spring coils.
            parent (QGraphicsItem, optional): Parent item.
            pen (QPen, optional): Pen for drawing the spring.
            name (str): Name of the spring.
            label (str, optional): Additional label text.
            k (float): Spring constant.
            nCoils (int): Number of coils in the spring.
        """
        super().__init__(parent)
        self.stPt = ptSt
        self.enPt = ptEn
        self.freeLength = self.getLength()
        self.DL = self.length - self.freeLength
        self.centerPt = (self.stPt + self.enPt) / 2.0
        self.force = 0
        self.pen = pen
        self.coilsWidth = coilsWidth
        self.coilsLength = coilsLength
        self.top = - self.coilsLength / 2
        self.left = -self.coilsWidth / 2
        self.rect = qtc.QRectF(self.left, self.top, self.coilsWidth, self.coilsLength)
        self.name = name
        self.label = label
        self.k = k
        self.nCoils = nCoils
        self.transformation = qtg.QTransform()
        stTT = self.name + "\nx={:0.1f}, y={:0.1f}\nk = {:0.1f}".format(self.centerPt.x(), self.centerPt.y(), self.k)
        self.setToolTip(stTT)

    def setk(self, k=None):
        """Set the spring constant and update the tooltip.

        Args:
            k (float, optional): New spring constant.
        """
        if k is not None:
            self.k = k
            stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nk = {:0.3f}".format(self.stPt.x(), self.stPt.y(), self.k)
            self.setToolTip(stTT)

    def boundingRect(self):
        """Return the bounding rectangle of the spring after transformation.

        Returns:
            QRectF: The transformed bounding rectangle.
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def getLength(self):
        """Calculate the current length of the spring.

        Returns:
            float: The Euclidean distance between start and end points.
        """
        p = self.enPt - self.stPt
        self.length = math.sqrt(p.x() ** 2 + p.y() ** 2)
        return self.length

    def getForce(self):
        """Calculate the force exerted by the spring.

        Returns:
            float: The force (k * displacement).
        """
        self.force = self.k * self.getDL()
        return self.force

    def getDL(self):
        """Calculate the displacement from the free length.

        Returns:
            float: The displacement (current length - free length).
        """
        self.DL = self.length - self.freeLength
        return self.DL

    def getAngleDeg(self):
        """Calculate the angle of the spring relative to the horizontal.

        Returns:
            float: The angle in degrees.
        """
        p = self.enPt - self.stPt
        self.angleRad = math.atan2(p.y(), p.x())
        self.angleDeg = 180.0 / math.pi * self.angleRad
        return self.angleDeg

    def paint(self, painter, option, widget=None):
        """Draw the spring with its coils, connectors, and force label.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        try:
            self.transformation.reset()
            if self.pen is not None:
                painter.setPen(self.pen)
            self.getLength()
            self.getAngleDeg()
            self.getDL()
            ht = self.coilsWidth
            wd = self.coilsLength + self.DL
            top = -ht / 2
            left = -wd / 2
            right = wd / 2
            self.rect = qtc.QRectF(left, top, wd, ht)
            painter.drawLine(qtc.QPointF(left, 0), qtc.QPointF(left, ht / 2))  # Draw left connector
            dX = wd / (self.nCoils)
            for i in range(self.nCoils):
                # Draw the zigzag pattern of the spring coils
                painter.drawLine(qtc.QPointF(left + i * dX, ht / 2), qtc.QPointF(left + (i + 0.5) * dX, -ht / 2))
                painter.drawLine(qtc.QPointF(left + (i + 0.5) * dX, -ht / 2), qtc.QPointF(left + (i + 1) * dX, ht / 2))
            painter.drawLine(qtc.QPointF(right, ht / 2), qtc.QPointF(right, 0))  # Draw right connector
            painter.drawLine(qtc.QPointF(-self.length / 2, 0), qtc.QPointF(left, 0))  # Draw left extension
            painter.drawLine(qtc.QPointF(right, 0), qtc.QPointF(self.length / 2, 0))  # Draw right extension
            nodeRad = 2
            stRec = qtc.QRectF(-self.length / 2 - nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
            enRec = qtc.QRectF(self.length / 2 - nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
            painter.drawEllipse(stRec)  # Draw start node
            painter.drawEllipse(enRec)  # Draw end node
            self.transformation.translate(self.stPt.x(), self.stPt.y())
            self.transformation.rotate(self.angleDeg)
            self.transformation.translate(self.length / 2, 0)
            self.setTransform(self.transformation)
            self.transformation.reset()
            font = painter.font()
            font.setPointSize(6)
            painter.setFont(font)
            text = "k = {:0.1f} N/m".format(self.k)
            text += ", F = {:0.2f} N".format(self.force)
            fm = qtg.QFontMetrics(painter.font())
            centerPt = (self.stPt + self.enPt) / 2, 0
            painter.drawText(qtc.QPointF(-fm.width(text) / 2.0, fm.height() / 2.0), text)  # Draw force label
            if self.label is not None:
                font.setPointSize(12)
                painter.setFont(font)
                painter.drawText(qtc.QPointF((self.coilsWidth / 2.0) + 10, 0), self.label)
            self.transformation.reset()
        except Exception as e:
            logging.error(f"Error in LinearSpring.paint: {str(e)}")

class DashPot(qtw.QGraphicsItem):
    """A dashpot (damper) connecting two points in the four-bar linkage.

    Attributes:
        stPt (QPointF): Starting point of the dashpot.
        enPt (QPointF): Ending point of the dashpot.
        freeLength (float): Natural length of the dashpot.
        length (float): Current length of the dashpot.
        DL (float): Displacement from the free length.
        centerPt (QPointF): Midpoint of the dashpot.
        force (float): Current force exerted by the dashpot.
        Width (float): Width of the dashpot graphic.
        Length (float): Length of the dashpot graphic.
        conn1Len (float): Length of the first connector.
        conn2Len (float): Length of the second connector.
        rect (QRectF): Bounding rectangle for the dashpot.
        pen (QPen): Pen for drawing the dashpot.
        name (str): Name of the dashpot.
        label (str, optional): Additional label text.
        c (float): Damping coefficient.
        transformation (QTransform): Transformation for positioning the dashpot.
    """

    def __init__(self, ptSt=qtc.QPointF(0, 0), ptEn=qtc.QPointF(1, 1), dpWidth=10, dpLength=30, parent=None, pen=None,
                 name='Dashpot', label=None, c=10):
        """Initialize a dashpot with start and end points.

        Args:
            ptSt (QPointF): Starting point of the dashpot.
            ptEn (QPointF): Ending point of the dashpot.
            dpWidth (float): Width of the dashpot graphic.
            dpLength (float): Length of the dashpot graphic.
            parent (QGraphicsItem, optional): Parent item.
            pen (QPen, optional): Pen for drawing the dashpot.
            name (str): Name of the dashpot.
            label (str, optional): Additional label text.
            c (float): Damping coefficient.
        """
        super().__init__(parent)
        self.stPt = ptSt
        self.enPt = ptEn
        self.freeLength = self.getLength()
        self.DL = self.length - self.freeLength
        self.centerPt = (self.stPt + self.enPt) / 2.0
        self.force = 0
        self.pen = pen
        self.Width = dpWidth
        self.Length = dpLength
        self.conn1Len = self.freeLength - self.Length
        self.conn1Len /= 2
        self.conn2Len = self.freeLength / 2
        self.top = - self.Length / 2
        self.left = -self.Width / 2
        self.rect = qtc.QRectF(self.left, self.top, self.Width, self.Length)
        self.name = name
        self.label = label
        self.c = c
        self.transformation = qtg.QTransform()
        stTT = self.name + "\nx={:0.1f}, y={:0.1f}\nc = {:0.1f}".format(self.centerPt.x(), self.centerPt.y(), self.c)
        self.setToolTip(stTT)

    def setc(self, c=None):
        """Set the damping coefficient and update the tooltip.

        Args:
            c (float, optional): New damping coefficient.
        """
        if c is not None:
            self.c = c
            stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nc = {:0.3f}".format(self.stPt.x(), self.stPt.y(), self.c)
            self.setToolTip(stTT)

    def boundingRect(self):
        """Return the bounding rectangle of the dashpot after transformation.

        Returns:
            QRectF: The transformed bounding rectangle.
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def getLength(self):
        """Calculate the current length of the dashpot.

        Returns:
            float: The Euclidean distance between start and end points.
        """
        p = self.enPt - self.stPt
        self.length = math.sqrt(p.x() ** 2 + p.y() ** 2)
        return self.length

    def getDL(self):
        """Calculate the displacement from the free length.

        Returns:
            float: The displacement (current length - free length).
        """
        self.DL = self.length - self.freeLength
        return self.DL

    def getAngleDeg(self):
        """Calculate the angle of the dashpot relative to the horizontal.

        Returns:
            float: The angle in degrees.
        """
        p = self.enPt - self.stPt
        self.angleRad = math.atan2(p.y(), p.x())
        self.angleDeg = 180.0 / math.pi * self.angleRad
        return self.angleDeg

    def paint(self, painter, option, widget=None):
        """Draw the dashpot with its piston, connectors, and force label.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        try:
            self.transformation.reset()
            if self.pen is not None:
                painter.setPen(self.pen)
            self.getLength()
            self.getAngleDeg()
            self.getDL()
            ht = self.Width
            wd = self.Length
            top = -ht / 2
            left = self.conn1Len
            right = left + wd
            self.rect = qtc.QRectF(left, top, wd, ht)
            painter.drawLine(qtc.QPointF(left, -ht / 2), qtc.QPointF(left, ht / 2))  # Draw left wall
            painter.drawLine(qtc.QPointF(left, -ht / 2), qtc.QPointF(right, -ht / 2))  # Draw top edge
            painter.drawLine(qtc.QPointF(left, ht / 2), qtc.QPointF(right, ht / 2))  # Draw bottom edge
            painter.drawLine(qtc.QPointF(left + wd / 2 + self.DL, ht / 2 * 0.95),
                            qtc.QPointF(left + wd / 2 + self.DL, -ht / 2 * 0.95))  # Draw piston
            painter.drawLine(qtc.QPointF(0, 0), qtc.QPointF(left, 0))  # Draw left connector
            painter.drawLine(qtc.QPointF(left + wd / 2 + self.DL, 0),
                            qtc.QPointF(self.conn2Len + left + wd / 2 + self.DL, 0))  # Draw right connector
            nodeRad = 2
            stRec = qtc.QRectF(-nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
            enRec = qtc.QRectF(self.length - nodeRad, -nodeRad, 2 * nodeRad, 2 * nodeRad)
            painter.drawEllipse(stRec)  # Draw start node
            painter.drawEllipse(enRec)  # Draw end node
            self.transformation.translate(self.stPt.x(), self.stPt.y())
            self.transformation.rotate(self.angleDeg)
            self.setTransform(self.transformation)
            font = painter.font()
            font.setPointSize(6)
            painter.setFont(font)
            text = "c = {:0.1f} Ns/m, F = {:0.2f} N".format(self.c, self.force)
            fm = qtg.QFontMetrics(painter.font())
            painter.drawText(qtc.QPointF(-fm.width(text) / 2.0, fm.height() / 1.5), text)  # Draw force label
            self.transformation.reset()
        except Exception as e:
            logging.error(f"Error in DashPot.paint: {str(e)}")

class FourBarLinkage_Model:
    """Model for the four-bar linkage, managing its components and state.

    Attributes:
        GroundLink (RigidLink): The fixed frame link.
        InputLink (RigidLink): The input link (driven by user or simulation).
        DragLink (RigidLink): The coupler link.
        OutputLink (RigidLink): The output link.
        Pivot0 (RigidPivotPoint): Pivot point for the Input link.
        Pivot1 (RigidPivotPoint): Pivot point for the Output link.
        Spring (LinearSpring): Spring affecting the linkage motion.
        DashPot (DashPot): Dashpot affecting the linkage motion.
        Tracer0 (Tracer): Tracer for the Output link end.
        Tracer1 (Tracer): Tracer for the Input link end.
        Tracer2 (Tracer): Tracer for the midpoint of the Coupler.
        Tracer3 (Tracer): Tracer for a point on the Coupler.
        controller (FourBarLinkage_Controller): Reference to the controller.
    """

    def __init__(self):
        """Initialize the four-bar linkage model with default components."""
        self.GroundLink = RigidLink()
        self.InputLink = RigidLink()
        self.DragLink = RigidLink()
        self.OutputLink = RigidLink()
        self.Pivot0 = RigidPivotPoint()
        self.Pivot1 = RigidPivotPoint()
        self.Spring = LinearSpring()
        self.DashPot = DashPot()
        self.Tracer0 = Tracer()
        self.Tracer1 = Tracer()
        self.Tracer2 = Tracer()
        self.Tracer3 = Tracer()
        self.controller = None

    def setInputLength(self, L=10):
        """Set the length of the Input link and update dependent components.

        Args:
            L (float): New length of the Input link.
        """
        try:
            # Update the end point of the Input link based on the new length
            self.InputLink.enPt.setX(self.InputLink.stPt.x() + math.cos(self.InputLink.angle) * L)
            self.InputLink.enPt.setY(self.InputLink.stPt.y() - math.sin(self.InputLink.angle) * L)
            self.InputLink.linkLength()
            self.DragLink.stPt.setX(self.InputLink.enPt.x())
            self.DragLink.stPt.setY(self.InputLink.enPt.y())
        except Exception as e:
            logging.error(f"Error in setInputLength: {str(e)}")

    def setOutputLength(self, L=10):
        """Set the length of the Output link and update dependent components.

        Args:
            L (float): New length of the Output link.
        """
        try:
            # Update the end point of the Output link based on the new length
            self.OutputLink.enPt.setX(self.OutputLink.stPt.x() + math.cos(self.OutputLink.angle) * L)
            self.OutputLink.enPt.setY(self.OutputLink.stPt.y() - math.sin(self.OutputLink.angle) * L)
            self.OutputLink.linkLength()
            self.DragLink.enPt.setX(self.OutputLink.enPt.x())
            self.DragLink.enPt.setY(self.OutputLink.enPt.y())
        except Exception as e:
            logging.error(f"Error in setOutputLength: {str(e)}")

    def moveLinkage(self, pt=qtc.QPointF(0, 0)):
        """Update the positions of the linkage components based on a new point.

        Uses a numerical solver to ensure kinematic consistency.

        Args:
            pt (QPointF): New position for the Input link's end point.
        """
        try:
            l1 = self.InputLink.length
            l3 = self.OutputLink.length
            x = pt.x()
            y = pt.y()
            # Calculate the Input link angle based on the new position
            if (x == self.InputLink.stPt.x()):
                self.angle1 = math.pi / 2 if y <= self.InputLink.stPt.y() else math.pi * 3.0 / 2.0
            else:
                self.angle1 = math.atan(-(y - self.InputLink.stPt.y()) / (x - self.InputLink.stPt.x()))
                self.angle1 += math.pi if x < self.InputLink.stPt.x() else 0
            # Calculate the initial guess for the Output link angle
            if (self.OutputLink.enPt.x() == self.OutputLink.stPt.x()):
                self.angle2 = math.pi / 2 if self.OutputLink.enPt.y() < self.OutputLink.stPt.y() else math.pi * 3.0 / 2.0
            else:
                self.angle2 = math.atan(-(self.OutputLink.enPt.y() - self.DragLink.stPt.y()) / (
                        self.OutputLink.enPt.x() - self.OutputLink.stPt.x()))
                self.angle2 += math.pi if self.OutputLink.enPt.x() < self.OutputLink.stPt.x() else 0
            self.InputLink.enPt.setX(self.InputLink.stPt.x() + math.cos(self.angle1) * l1)
            self.InputLink.enPt.setY(self.InputLink.stPt.y() - math.sin(self.angle1) * l1)
            x1 = self.InputLink.enPt.x()
            y1 = self.InputLink.enPt.y()
            self.lTest = self.DragLink.length

            def fn1(angle2):
                """Helper function for the numerical solver to find the Output link angle.

                Args:
                    angle2 (float): Angle of the Output link in radians.

                Returns:
                    float: Difference between desired and actual Coupler length.
                """
                x2 = self.OutputLink.stPt.x() + l3 * math.cos(angle2)
                y2 = self.OutputLink.stPt.y() - l3 * math.sin(angle2)
                self.lTest = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
                return self.DragLink.length - self.lTest

            result = optimize.fsolve(fn1, [self.angle2])  # Solve for the Output link angle
            if abs(self.lTest - self.DragLink.length) > 0.001:
                # If the solver fails, revert to previous angles
                self.angle2 = getattr(self, 'prevBeta', self.angle2)
                self.angle1 = getattr(self, 'prevAlpha', self.angle1)
                self.InputLink.enPt.setX(self.InputLink.stPt.x() + math.cos(self.angle1) * l1)
                self.InputLink.enPt.setY(self.InputLink.stPt.y() - math.sin(self.angle1) * l1)
            else:
                self.angle2 = result[0]
                self.prevAlpha = self.angle1
                self.prevBeta = self.angle2

            # Update the Output link position
            self.OutputLink.enPt.setX(self.OutputLink.stPt.x() + l3 * math.cos(self.angle2))
            self.OutputLink.enPt.setY(self.OutputLink.stPt.y() - l3 * math.sin(self.angle2))
            pt1 = dc(self.InputLink.enPt)
            pt0 = dc(self.OutputLink.enPt)
            ptMid = (pt0 + pt1) / 2
            # Update tracer points
            self.Tracer1.pts.append(pt1)
            self.Tracer0.pts.append(pt0)
            self.Tracer2.pts.append(ptMid)
            self.Tracer3.pts.append(ptMid + 0.5 * (pt0 - ptMid))
            self.Spring.enPt = dc(self.Tracer3.lastPt())
            self.Spring.getForce()
            self.DashPot.enPt = dc(self.Tracer3.lastPt())
            if self.controller:
                self.DashPot.force = self.controller.dashpot_force
            # Limit tracer points to prevent memory issues
            if len(self.Tracer1.pts) >= 1000:
                self.Tracer0.pts = self.Tracer0.pts[1:]
                self.Tracer1.pts = self.Tracer1.pts[1:]
                self.Tracer2.pts = self.Tracer2.pts[1:]
                self.Tracer3.pts = self.Tracer3.pts[1:]
            self.DragLink.stPt = self.InputLink.enPt
            self.DragLink.enPt = self.OutputLink.enPt
        except Exception as e:
            logging.error(f"Error in moveLinkage: {str(e)}")
#endregion

#region view
class FourBarLinkage_View:
    """View for the four-bar linkage, handling the graphical representation.

    Attributes:
        gv_Main (QGraphicsView): The main graphics view widget.
        scene (QGraphicsScene): The scene containing the linkage graphics.
        penThick (QPen): Pen for thick lines.
        penMed (QPen): Pen for medium lines.
        penLink (QPen): Pen for drawing links.
        penGridLines (QPen): Pen for drawing the grid.
        penTracer (QPen): Pen for drawing tracers.
        penTracerIcon (QPen): Pen for drawing tracer icons.
        brushFill (QBrush): Brush for filling shapes.
        brushHatch (QBrush): Brush for hatched patterns.
        brushGrid (QBrush): Brush for the grid background.
        brushLink (QBrush): Brush for filling links.
        brushPivot (QBrush): Brush for filling pivots.
    """

    def __init__(self, gv_Main):
        """Initialize the view with the main graphics view.

        Args:
            gv_Main (QGraphicsView): The main graphics view widget.
        """
        self.gv_Main = gv_Main

    def setupGraphics(self):
        """Set up the graphics scene and viewport."""
        self.scene = qtw.QGraphicsScene()
        self.scene.setObjectName("MyScene")
        self.scene.setSceneRect(-200, -200, 400, 400)  # Set the scene size
        self.gv_Main.setScene(self.scene)
        self.setupPensAndBrushes()
        self.gv_Main.setViewportUpdateMode(qtw.QGraphicsView.FullViewportUpdate)

    def setupPensAndBrushes(self):
        """Initialize pens and brushes for drawing the linkage components."""
        self.penThick = qtg.QPen(qtc.Qt.darkGreen)
        self.penThick.setWidth(5)
        self.penMed = qtg.QPen(qtc.Qt.darkBlue)
        self.penMed.setStyle(qtc.Qt.SolidLine)
        self.penMed.setWidth(2)
        self.penLink = qtg.QPen(qtg.QColor("orange"))
        self.penLink.setWidth(1)
        self.penGridLines = qtg.QPen()
        self.penGridLines.setWidth(1)
        self.penGridLines.setColor(qtg.QColor.fromHsv(197, 144, 228, 128))
        self.penTracer = qtg.QPen(qtc.Qt.blue)
        self.penTracerIcon = qtg.QPen(qtc.Qt.black)
        self.brushFill = qtg.QBrush(qtc.Qt.darkRed)
        self.brushHatch = qtg.QBrush()
        self.brushHatch.setStyle(qtc.Qt.DiagCrossPattern)
        self.brushGrid = qtg.QBrush(qtg.QColor.fromHsv(87, 98, 245, 128))
        self.brushLink = qtg.QBrush(qtg.QColor.fromHsv(35, 255, 255, 64))
        self.brushPivot = qtg.QBrush(qtg.QColor.fromHsv(0, 0, 128, 255))

    def BuildScene(self, FBL_M):
        """Build the four-bar linkage scene with all components.

        Args:
            FBL_M (FourBarLinkage_Model): The four-bar linkage model.
        """
        try:
            self.scene.clear()
            # Draw the background grid
            self.drawAGrid(DeltaX=10, DeltaY=10, Height=400, Width=400, Pen=self.penGridLines, Brush=self.brushGrid)
            FBL_M.DragLink.stPt = qtc.QPointF(-100, -60)
            FBL_M.DragLink.enPt = qtc.QPointF(100, -150)
            # Draw the pivot points
            FBL_M.Pivot0 = self.drawPivot(-100, 0, 10, 20)
            FBL_M.Pivot0.setTransformOriginPoint(qtc.QPointF(FBL_M.Pivot0.x, FBL_M.Pivot0.y))
            FBL_M.Pivot0.rotate(0)
            FBL_M.Pivot0.name = "RP 0"
            FBL_M.Pivot1 = self.drawPivot(60, 0, 10, 20)
            FBL_M.Pivot1.setTransformOriginPoint(qtc.QPointF(FBL_M.Pivot1.x, FBL_M.Pivot1.y))
            FBL_M.Pivot1.rotate(0)
            FBL_M.Pivot1.name = "RP 1"
            # Draw the Ground link (Frame)
            FBL_M.GroundLink = self.drawLinkage(FBL_M.Pivot0.x, FBL_M.Pivot0.y, FBL_M.Pivot1.x, FBL_M.Pivot1.y,
                                               radius=5, pen=self.penGridLines, brush=self.brushGrid)
            FBL_M.InputLink = self.drawLinkage(FBL_M.Pivot0.x, FBL_M.Pivot0.y, FBL_M.DragLink.stPt.x(),
                                               FBL_M.DragLink.stPt.y(), 5)
            FBL_M.DragLink = self.drawLinkage(FBL_M.DragLink.stPt.x(), FBL_M.DragLink.stPt.y(),
                                             FBL_M.DragLink.enPt.x(), FBL_M.DragLink.enPt.y(), 5)
            FBL_M.OutputLink = self.drawLinkage(FBL_M.Pivot1.x, FBL_M.Pivot1.y, FBL_M.DragLink.enPt.x(),
                                               FBL_M.DragLink.enPt.y(), 5)
            # Set names for the links
            FBL_M.GroundLink.name = "Frame"
            FBL_M.InputLink.name = "Input"
            FBL_M.DragLink.name = "Coupler"
            FBL_M.OutputLink.name = "Output"
            # Initialize tracers
            FBL_M.Tracer0 = Tracer(x=100, y=-150, pen=self.penTracer)
            self.scene.addItem(FBL_M.Tracer0)
            FBL_M.Tracer1 = Tracer(x=-100, y=-60, pen=self.penTracer)
            self.scene.addItem(FBL_M.Tracer1)
            FBL_M.Tracer2 = Tracer(0, 0, pen=self.penTracer)
            FBL_M.Tracer2.pts[0] = (FBL_M.Tracer0.pts[0] + FBL_M.Tracer1.pts[0]) / 2
            self.scene.addItem(FBL_M.Tracer2)
            FBL_M.Tracer3 = Tracer(0, 0, pen=self.penTracer)
            FBL_M.Tracer3.pts[0] = (FBL_M.Tracer0.pts[0] + FBL_M.Tracer2.pts[0]) / 2
            self.scene.addItem(FBL_M.Tracer3)
            # Add spring and dashpot
            FBL_M.Spring = LinearSpring(FBL_M.Pivot1.pt, FBL_M.Tracer3.pts[0], 20, 50)
            self.scene.addItem(FBL_M.Spring)
            FBL_M.DashPot = DashPot(FBL_M.Pivot1.pt, FBL_M.Tracer3.lastPt(), 10, 80)
            self.scene.addItem(FBL_M.DashPot)
        except Exception as e:
            logging.error(f"Error in BuildScene: {str(e)}")

    def drawAGrid(self, DeltaX=10, DeltaY=10, Height=200, Width=200, CenterX=0, CenterY=0, Pen=None, Brush=None, SubGrid=None):
        """Draw a grid in the background of the scene.

        Args:
            DeltaX (float): Horizontal spacing between grid lines.
            DeltaY (float): Vertical spacing between grid lines.
            Height (float): Height of the grid.
            Width (float): Width of the grid.
            CenterX (float): X-coordinate of the grid center.
            CenterY (float): Y-coordinate of the grid center.
            Pen (QPen, optional): Pen for drawing the grid lines.
            Brush (QBrush, optional): Brush for filling the grid background.
            SubGrid (optional): Not used.
        """
        height = self.scene.sceneRect().height() if Height is None else Height
        width = self.scene.sceneRect().width() if Width is None else Width
        left = self.scene.sceneRect().left() if CenterX is None else (CenterX - width / 2.0)
        right = self.scene.sceneRect().right() if CenterX is None else (CenterX + width / 2.0)
        top = self.scene.sceneRect().top() if CenterY is None else (CenterY - height / 2.0)
        bottom = self.scene.sceneRect().bottom() if CenterY is None else (CenterY + height / 2.0)
        Dx = DeltaX
        Dy = DeltaY
        pen = qtg.QPen() if Pen is None else Pen
        if Brush is not None:
            rect = self.drawARectangle(left, top, width, height)
            rect.setBrush(Brush)
            rect.setPen(pen)
        x = left
        while x <= right:
            lVert = self.drawALine(x, top, x, bottom)
            lVert.setPen(pen)  # Draw vertical grid lines
            x += Dx
        y = top
        while y <= bottom:
            lHor = self.drawALine(left, y, right, y)
            lHor.setPen(pen)  # Draw horizontal grid lines
            y += Dy

    def drawARectangle(self, leftX, topY, widthX, heightY, pen=None, brush=None):
        """Draw a rectangle in the scene.

        Args:
            leftX (float): X-coordinate of the top-left corner.
            topY (float): Y-coordinate of the top-left corner.
            widthX (float): Width of the rectangle.
            heightY (float): Height of the rectangle.
            pen (QPen, optional): Pen for drawing the rectangle.
            brush (QBrush, optional): Brush for filling the rectangle.

        Returns:
            QGraphicsRectItem: The drawn rectangle.
        """
        rect = qtw.QGraphicsRectItem(leftX, topY, widthX, heightY)
        if brush is not None:
            rect.setBrush(brush)
        if pen is not None:
            rect.setPen(pen)
        self.scene.addItem(rect)
        return rect

    def drawALine(self, stX, stY, enX, enY, pen=None):
        """Draw a line in the scene.

        Args:
            stX (float): X-coordinate of the start point.
            stY (float): Y-coordinate of the start point.
            enX (float): X-coordinate of the end point.
            enY (float): Y-coordinate of the end point.
            pen (QPen, optional): Pen for drawing the line.

        Returns:
            QGraphicsLineItem: The drawn line.
        """
        if pen is None:
            pen = self.penMed
        line = qtw.QGraphicsLineItem(stX, stY, enX, enY)
        line.setPen(pen)
        self.scene.addItem(line)
        return line

    def polarToRect(self, centerX, centerY, radius, angleDeg=0):
        """Convert polar coordinates to rectangular coordinates.

        Args:
            centerX (float): X-coordinate of the center.
            centerY (float): Y-coordinate of the center.
            radius (float): Radius from the center.
            angleDeg (float): Angle in degrees.

        Returns:
            tuple: (x, y) coordinates.
        """
        angleRad = angleDeg * 2.0 * math.pi / 360.0
        return centerX + radius * math.cos(angleRad), centerY + radius * math.sin(angleRad)

    def drawACircle(self, centerX, centerY, Radius, angle=0, brush=None, pen=None):
        """Draw a circle in the scene.

        Args:
            centerX (float): X-coordinate of the center.
            centerY (float): Y-coordinate of the center.
            Radius (float): Radius of the circle.
            angle (float): Not used.
            brush (QBrush, optional): Brush for filling the circle.
            pen (QPen, optional): Pen for drawing the circle.

        Returns:
            QGraphicsEllipseItem: The drawn circle.
        """
        ellipse = qtw.QGraphicsEllipseItem(centerX - Radius, centerY - Radius, 2 * Radius, 2 * Radius)
        if pen is not None:
            ellipse.setPen(pen)
        if brush is not None:
            ellipse.setBrush(brush)
        self.scene.addItem(ellipse)
        return ellipse

    def drawASquare(self, centerX, centerY, Size, brush=None, pen=None):
        """Draw a square in the scene.

        Args:
            centerX (float): X-coordinate of the center.
            centerY (float): Y-coordinate of the center.
            Size (float): Size of the square.
            brush (QBrush, optional): Brush for filling the square.
            pen (QPen, optional): Pen for drawing the square.

        Returns:
            QGraphicsRectItem: The drawn square.
        """
        sqr = qtw.QGraphicsRectItem(centerX - Size / 2.0, centerY - Size / 2.0, Size, Size)
        if pen is not None:
            sqr.setPen(pen)
        if brush is not None:
            sqr.setBrush(brush)
        self.scene.addItem(sqr)
        return sqr

    def drawATriangle(self, centerX, centerY, Radius, angleDeg=0, brush=None, pen=None):
        """Draw a triangle in the scene.

        Args:
            centerX (float): X-coordinate of the center.
            centerY (float): Y-coordinate of the center.
            Radius (float): Radius defining the size of the triangle.
            angleDeg (float): Rotation angle in degrees.
            brush (QBrush, optional): Brush for filling the triangle.
            pen (QPen, optional): Pen for drawing the triangle.

        Returns:
            QGraphicsPolygonItem: The drawn triangle.
        """
        pts = []
        x, y = self.polarToRect(centerX, centerY, Radius, 0 + angleDeg)
        pts.append(qtc.QPointF(x, y))
        x, y = self.polarToRect(centerX, centerY, Radius, 120 + angleDeg)
        pts.append(qtc.QPointF(x, y))
        x, y = self.polarToRect(centerX, centerY, Radius, 240 + angleDeg)
        pts.append(qtc.QPointF(x, y))
        x, y = self.polarToRect(centerX, centerY, Radius, 0 + angleDeg)
        pts.append(qtc.QPointF(x, y))
        pg = qtg.QPolygonF(pts)
        PG = qtw.QGraphicsPolygonItem(pg)
        if brush is not None:
            PG.setBrush(brush)
        if pen is not None:
            PG.setPen(pen)
        self.scene.addItem(PG)
        return PG

    def drawLinkage(self, stX=0, stY=0, enX=1, enY=1, radius=10, pen=None, brush=None, name='RigidLink'):
        """Draw a rigid link in the scene.

        Args:
            stX (float): X-coordinate of the start point.
            stY (float): Y-coordinate of the start point.
            enX (float): X-coordinate of the end point.
            enY (float): Y-coordinate of the end point.
            radius (float): Radius for drawing the ends of the link.
            pen (QPen, optional): Pen for drawing the link.
            brush (QBrush, optional): Brush for filling the link.
            name (str): Name of the link.

        Returns:
            RigidLink: The drawn link.
        """
        if pen is None:
            pen = self.penLink
        if brush is None:
            brush = self.brushLink
        RL = RigidLink(stX, stY, enX, enY, radius=radius, pen=pen, brush=brush, name=name)
        self.scene.addItem(RL)
        return RL

    def drawPivot(self, ptX=0, ptY=0, Height=10, Width=10, pen=None, brush=None):
        """Draw a pivot point in the scene.

        Args:
            ptX (float): X-coordinate of the pivot.
            ptY (float): Y-coordinate of the pivot.
            Height (float): Height of the pivot graphic.
            Width (float): Width of the pivot graphic.
            pen (QPen, optional): Pen for drawing the pivot.
            brush (QBrush, optional): Brush for filling the pivot.

        Returns:
            RigidPivotPoint: The drawn pivot point.
        """
        if pen is None:
            pen = self.penMed
        if brush is None:
            brush = self.brushPivot
        PP = RigidPivotPoint(ptX, ptY, Height, Width, pen=pen, brush=brush)
        self.scene.addItem(PP)
        return PP
#endregion

#region controller
class FourBarLinkage_Controller:
    """Controller for the four-bar linkage, managing user interactions and updates.

    Attributes:
        FBL_M (FourBarLinkage_Model): The four-bar linkage model.
        FBL_V (FourBarLinkage_View): The four-bar linkage view.
        nud_input_angle (QDoubleSpinBox): Widget for the Input link angle.
        lbl_output_angle (QLabel): Widget for displaying the Output link angle.
        nud_link1_length (QDoubleSpinBox): Widget for the Input link length.
        nud_link3_length (QDoubleSpinBox): Widget for the Output link length.
        spnd_Zoom (QDoubleSpinBox): Widget for zoom level.
        nud_min_angle (QDoubleSpinBox): Widget for the minimum angle limit.
        nud_max_angle (QDoubleSpinBox): Widget for the maximum angle limit.
        nud_damping (QDoubleSpinBox): Widget for the damping coefficient.
        nud_mass (QDoubleSpinBox): Widget for the mass.
        nud_spring (QDoubleSpinBox): Widget for the spring constant.
        is_simulation_running (bool): Whether the simulation is currently running.
        timer (QTimer): Timer for updating the simulation.
        t (numpy.ndarray): Time array for the simulation.
        prev_time (float): Previous time step for velocity calculation.
        prev_position (float): Previous position for velocity calculation.
        prev_length (float): Previous dashpot length for velocity calculation.
        dashpot_force (float): Current force exerted by the dashpot.
    """

    def __init__(self, widgets):
        """Initialize the controller with UI widgets.

        Args:
            widgets (list): List of UI widgets in the following order:
                - gv_Main (QGraphicsView)
                - nud_input_angle (QDoubleSpinBox)
                - lbl_output_angle (QLabel)
                - nud_link1_length (QDoubleSpinBox)
                - nud_link3_length (QDoubleSpinBox)
                - spnd_Zoom (QDoubleSpinBox)
                - nud_min_angle (QDoubleSpinBox)
                - nud_max_angle (QDoubleSpinBox)
                - nud_damping (QDoubleSpinBox)
                - nud_mass (QDoubleSpinBox)
                - nud_spring (QDoubleSpinBox)
        """
        self.FBL_M = FourBarLinkage_Model()
        self.FBL_V = FourBarLinkage_View(widgets[0])
        self.nud_input_angle = widgets[1]
        self.lbl_output_angle = widgets[2]
        self.nud_link1_length = widgets[3]
        self.nud_link3_length = widgets[4]
        self.spnd_Zoom = widgets[5]
        self.nud_min_angle = widgets[6]
        self.nud_max_angle = widgets[7]
        self.nud_damping = widgets[8]
        self.nud_mass = widgets[9]
        self.nud_spring = widgets[10]
        self.is_simulation_running = False
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.t = np.linspace(0, 10, 1000)  # Time array for simulation
        self.prev_time = 0
        self.prev_position = 0
        self.prev_length = 0
        self.dashpot_force = 0
        self.FBL_M.controller = self

    def setupGraphics(self):
        """Set up the graphics view and scene."""
        self.FBL_V.setupGraphics()

    def buildScene(self):
        """Build the initial four-bar linkage scene."""
        self.FBL_V.BuildScene(self.FBL_M)

    def setInputLinkLength(self):
        """Update the Input link length based on the UI input."""
        try:
            length = self.nud_link1_length.value()
            self.FBL_M.setInputLength(length)
            # Recompute the position based on the current angle
            angle = math.radians(self.nud_input_angle.value())
            pt = qtc.QPointF(
                self.FBL_M.InputLink.stPt.x() + length * math.cos(angle),
                self.FBL_M.InputLink.stPt.y() - length * math.sin(angle)
            )
            self.moveLinkage(pt)
        except Exception as e:
            logging.error(f"Error in setInputLinkLength: {str(e)}")

    def setOutputLinkLength(self):
        """Update the Output link length based on the UI input."""
        try:
            length = self.nud_link3_length.value()
            self.FBL_M.setOutputLength(length)
            # Recompute the position to maintain kinematic consistency
            angle = math.radians(self.nud_input_angle.value())
            pt = qtc.QPointF(
                self.FBL_M.InputLink.stPt.x() + self.FBL_M.InputLink.length * math.cos(angle),
                self.FBL_M.InputLink.stPt.y() - self.FBL_M.InputLink.length * math.sin(angle)
            )
            self.moveLinkage(pt)
        except Exception as e:
            logging.error(f"Error in setOutputLinkLength: {str(e)}")

    def setAngleLimits(self, min_angle, max_angle):
        """Set the minimum and maximum angle limits for the Input link.

        Args:
            min_angle (float): Minimum angle in degrees.
            max_angle (float): Maximum angle in degrees.
        """
        try:
            angle = self.FBL_M.InputLink.AngleDeg()
            if angle < min_angle:
                angle = min_angle
            elif angle > max_angle:
                angle = max_angle
            self.nud_input_angle.setValue(angle)
            # Update the linkage position based on the constrained angle
            pt = qtc.QPointF(
                self.FBL_M.InputLink.stPt.x() + self.FBL_M.InputLink.length * math.cos(math.radians(angle)),
                self.FBL_M.InputLink.stPt.y() - self.FBL_M.InputLink.length * math.sin(math.radians(angle))
            )
            self.moveLinkage(pt)
        except Exception as e:
            logging.error(f"Error in setAngleLimits: {str(e)}")

    def setDampingCoefficient(self, c):
        """Set the damping coefficient for the dashpot.

        Args:
            c (float): New damping coefficient.
        """
        try:
            self.FBL_M.DashPot.setc(c)
        except Exception as e:
            logging.error(f"Error in setDampingCoefficient: {str(e)}")

    def setMass(self, m):
        """Set the mass of the linkage system.

        Args:
            m (float): New mass value.
        """
        try:
            if m <= 0:
                logging.warning("Mass must be positive. Setting to default value 1.0")
                m = 1.0
                self.nud_mass.setValue(m)
            self.FBL_M.InputLink.mass = m
        except Exception as e:
            logging.error(f"Error in setMass: {str(e)}")

    def setSpringConstant(self, k):
        """Set the spring constant for the spring.

        Args:
            k (float): New spring constant.
        """
        try:
            self.FBL_M.Spring.setk(k)
        except Exception as e:
            logging.error(f"Error in setSpringConstant: {str(e)}")

    def moveLinkage(self, pt):
        """Move the linkage to a new position and update the UI.

        Args:
            pt (QPointF): New position for the Input link's end point.
        """
        try:
            self.FBL_M.moveLinkage(pt)
            self.nud_input_angle.setValue(self.FBL_M.InputLink.AngleDeg())
            self.lbl_output_angle.setText("{:0.2f}".format(self.FBL_M.OutputLink.AngleDeg()))
            self.FBL_V.scene.update()  # Force scene update to redraw the linkage
        except Exception as e:
            logging.error(f"Error in moveLinkage: {str(e)}")

    def state_equations(self, state, t, m, k, c):
        """Define the state equations for the dynamic simulation.

        Args:
            state (list): Current state [angle, angular velocity].
            t (float): Current time.
            m (float): Mass of the system.
            k (float): Spring constant.
            c (float): Damping coefficient.

        Returns:
            list: Derivatives [angular velocity, angular acceleration].
        """
        try:
            theta, omega = state
            I = m * self.FBL_M.InputLink.length ** 2  # Moment of inertia
            if I <= 0:
                raise ValueError("Moment of inertia must be positive")
            # Calculate forces
            spring_force = self.FBL_M.Spring.getForce()
            # Use the dashpot force computed from length change rate
            dashpot_force = self.dashpot_force
            # Equation of motion: I * alpha = -k * (theta - equilibrium) - c * omega
            # Equilibrium is at 90 degrees (pi/2 radians)
            equilibrium_angle = math.pi / 2
            alpha = (-k * (theta - equilibrium_angle) - dashpot_force) / I
            return [omega, alpha]
        except Exception as e:
            logging.error(f"Error in state_equations: {str(e)}")
            return [0, 0]

    def startSimulation(self, initial_angle, m, k, c):
        """Start the dynamic simulation of the linkage.

        Args:
            initial_angle (float): Initial angle of the Input link in degrees.
            m (float): Mass of the system.
            k (float): Spring constant.
            c (float): Damping coefficient.
        """
        try:
            self.is_simulation_running = True
            initial_angle_rad = math.radians(initial_angle)
            initial_state = [initial_angle_rad, 0]  # [angle, angular velocity]
            # Solve the differential equations
            solution = odeint(self.state_equations, initial_state, self.t, args=(m, k, c))
            self.simulation_data = solution
            self.current_step = 0
            self.prev_time = 0
            self.prev_position = initial_angle_rad
            self.prev_length = self.FBL_M.DashPot.freeLength
            self.timer.start(10)  # Update every 10 ms
        except Exception as e:
            logging.error(f"Error in startSimulation: {str(e)}")

    def update_simulation(self):
        """Update the simulation state at each time step."""
        try:
            if self.current_step < len(self.simulation_data) and self.is_simulation_running:
                theta = self.simulation_data[self.current_step][0]
                current_time = self.t[self.current_step]
                # Get previous dashpot length
                prev_length = self.prev_length
                # Update the linkage position
                pt = qtc.QPointF(
                    self.FBL_M.InputLink.stPt.x() + self.FBL_M.InputLink.length * math.cos(theta),
                    self.FBL_M.InputLink.stPt.y() - self.FBL_M.InputLink.length * math.sin(theta)
                )
                self.moveLinkage(pt)
                # Get current dashpot length
                current_length = self.FBL_M.DashPot.getLength()
                # Calculate velocity (rate of change of length)
                if current_time != self.prev_time:
                    velocity = (current_length - prev_length) / (current_time - self.prev_time)
                else:
                    velocity = 0
                self.dashpot_force = self.FBL_M.DashPot.c * velocity
                self.prev_length = current_length
                self.prev_time = current_time
                self.prev_position = theta
                self.current_step += 1
            else:
                self.timer.stop()
                self.is_simulation_running = False
        except Exception as e:
            logging.error(f"Error in update_simulation: {str(e)}")

    def pauseResumeSimulation(self):
        """Toggle between pausing and resuming the simulation."""
        try:
            if self.is_simulation_running:
                self.timer.stop()
                self.is_simulation_running = False
            else:
                self.timer.start(10)
                self.is_simulation_running = True
        except Exception as e:
            logging.error(f"Error in pauseResumeSimulation: {str(e)}")

    def resetTracers(self):
        """Reset the tracer paths to their initial points."""
        try:
            # Reset tracer points to their initial positions
            self.FBL_M.Tracer0.pts = [self.FBL_M.Tracer0.pts[0]]
            self.FBL_M.Tracer1.pts = [self.FBL_M.Tracer1.pts[0]]
            self.FBL_M.Tracer2.pts = [self.FBL_M.Tracer2.pts[0]]
            self.FBL_M.Tracer3.pts = [self.FBL_M.Tracer3.pts[0]]
            # Update the scene to reflect the reset
            self.FBL_V.scene.update()
        except Exception as e:
            logging.error(f"Error in resetTracers: {str(e)}")
#endregion
#endregion