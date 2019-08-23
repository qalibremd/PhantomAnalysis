import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import sys
import platform

#import phantom_analysis modules
import dicom_util
import scalar_analysis
import voi_analysis
import phantom_definitions
import thermometry

MOST_RECENT_INPUT_DIRECTORY_SETTING = "SlicerPhantomAnalysisMostRecentInputDirectory"

COLOR_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voi_labels.ctbl")
ADC_COLORS_OFFSET = 24


class phantom(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "phantom analysis"
        self.parent.categories = ["phantom"]
        self.parent.dependencies = []
        self.parent.contributors = ["Katie Manduca (http://boulderlabs.com)"]
        self.parent.helpText = """
            A module to locate a VOI in a phantom DWI and calculate stats on that VOI.
            """
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
            """ # replace with organization, grant and thanks.

#
# phantomWidget
#

class phantomWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Instantiate and connect widgets ...

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        # directory selection
        self.inputDirectoryButton = qt.QPushButton("Select a Directory")
        self.inputDirectory = None
        parametersFormLayout.addRow("DICOM Input:", self.inputDirectoryButton)

        # phantom type selection
        self.phantomModelSelector = qt.QGroupBox()

        vbox = qt.QVBoxLayout()
        self.phantom_radios = []
        for name in sorted(phantom_definitions.PHANTOM_CATALOG.keys()):
            rb = qt.QRadioButton(name)
            self.phantom_radios.append(rb)
            vbox.addWidget(rb)
        self.phantomModelSelector.setLayout(vbox)
        parametersFormLayout.addRow("Select phantom model: ", self.phantomModelSelector)

        #
        # output DWI volume selector
        #
        self.outputDWISelector = slicer.qMRMLNodeComboBox()
        self.outputDWISelector.nodeTypes = ["vtkMRMLDiffusionWeightedVolumeNode"]
        self.outputDWISelector.selectNodeUponCreation = True
        self.outputDWISelector.addEnabled = True
        self.outputDWISelector.removeEnabled = True
        self.outputDWISelector.renameEnabled = True
        self.outputDWISelector.noneEnabled = False
        self.outputDWISelector.showHidden = False
        self.outputDWISelector.showChildNodeTypes = False
        self.outputDWISelector.setMRMLScene( slicer.mrmlScene )
        self.outputDWISelector.setToolTip( "Pick the output to the algorithm." )
        parametersFormLayout.addRow("Output DWI Volume: ", self.outputDWISelector)

        #
        # output scalar volume selector
        #
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.renameEnabled = True
        self.outputSelector.noneEnabled = False
        self.outputSelector.showHidden = False
        self.outputSelector.showChildNodeTypes = False
        self.outputSelector.setMRMLScene( slicer.mrmlScene )
        self.outputSelector.setToolTip( "Pick the output to the algorithm." )
        parametersFormLayout.addRow("Output Scalar Volume: ", self.outputSelector)

        #
        # output VOI selector
        #
        self.outputVOISelector = slicer.qMRMLNodeComboBox()
        self.outputVOISelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.outputVOISelector.selectNodeUponCreation = True
        self.outputVOISelector.addEnabled = True
        self.outputVOISelector.removeEnabled = True
        self.outputVOISelector.renameEnabled = True
        self.outputVOISelector.noneEnabled = False
        self.outputVOISelector.showHidden = False
        self.outputVOISelector.showChildNodeTypes = False
        self.outputVOISelector.setMRMLScene( slicer.mrmlScene )
        self.outputVOISelector.setToolTip( "Pick the output VOI to the algorithm." )
        parametersFormLayout.addRow("Output VOI: ", self.outputVOISelector)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Run the algorithm."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # connections
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.inputDirectoryButton.connect('clicked(bool)', self.onInputDirectory)
        self.phantomModelSelector.connect('clicked(bool)', self.onSelect)
        self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.outputDWISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
        self.outputVOISelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

        # Refresh Apply button state
        self.onSelect()

        # Add space to display feedback from program
        self.statusLabel = qt.QPlainTextEdit()
        self.statusLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        parametersFormLayout.addRow(self.statusLabel)

        # Add vertical spacer
        self.layout.addStretch(1)

    def cleanup(self):
        pass

    def onSelect(self):
        self.applyButton.enabled = self.inputDirectory and \
                                   self.outputSelector.currentNode() and \
                                   self.outputDWISelector.currentNode() and \
                                   self.outputVOISelector.currentNode()

    def onApplyButton(self):
        self.statusLabel.plainText = ''
        logic = phantomLogic()
        logic.logCallback = self.addLog

        for rb in self.phantom_radios:
            if rb.isChecked():
                phantom_name = rb.text
        scalar_type = logic.run(self.inputDirectory, self.outputDWISelector.currentNode(), self.outputSelector.currentNode(), self.outputVOISelector.currentNode(), phantom_name)

        if self.outputVOISelector.currentNode().GetDisplayNode() == None:
            self.outputVOISelector.currentNode().CreateDefaultDisplayNodes()
            logging.info("Created Default Display Node")
        #set the color table
        color_table = slicer.util.loadColorTable(COLOR_TABLE, returnNode=True)[1]
        self.outputVOISelector.currentNode().GetDisplayNode().SetAndObserveColorNodeID(color_table.GetID())
        logging.info("Set Color Table for VOI")

        # set background to scalar image, foreground to voi map, and opacity to 0.7
        slicer.util.setSliceViewerLayers(background=self.outputSelector.currentNode(), foreground=self.outputVOISelector.currentNode(), foregroundOpacity=0.7)
        logging.info("Set slicer views")

        # set display colors for T1 and ADC cases
        if scalar_type == "T1" :
            displayNode = self.outputSelector.currentNode().GetDisplayNode()
            displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeGrey')
            displayNode.AutoWindowLevelOff()
            displayNode.SetWindow(1000)
            displayNode.SetLevel(400)

        elif scalar_type == "ADC":
            displayNode = self.outputSelector.currentNode().GetDisplayNode()
            displayNode.SetAndObserveColorNodeID('vtkMRMLColorTableNodeGrey')
            displayNode.AutoWindowLevelOn()

        # set 3D view
        layoutManager = slicer.app.layoutManager()
        for sliceViewName in layoutManager.sliceViewNames():
            controller = layoutManager.sliceWidget(sliceViewName).sliceController()
            controller.setSliceVisible(True)
        logging.info("Set 3D view")
        
        # center the views
        slicer.util.resetSliceViews()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDWidget.threeDView().resetFocalPoint()
        logging.info("Centered views")

        self.addLog("Finished")

    def onInputDirectory(self):
        # open a file selection dialog, starting where last left off
        inputDirectory = qt.QFileDialog.getExistingDirectory(
            None,
            None,
            qt.QSettings().value(MOST_RECENT_INPUT_DIRECTORY_SETTING),
            qt.QFileDialog.ShowDirsOnly
        )
        if inputDirectory:
            self.inputDirectory = inputDirectory
            qt.QSettings().setValue(MOST_RECENT_INPUT_DIRECTORY_SETTING, self.inputDirectory)

            button_text = self.inputDirectory
            max_button_text_len = 32
            if len(button_text) > max_button_text_len + len('...'):
                button_text = "..." + button_text[-max_button_text_len:]
            self.inputDirectoryButton.setText(button_text)
            self.inputDirectoryButton.setToolTip(self.inputDirectory)
        self.onSelect()

    def addLog(self, text):
        # Append text to log window
        self.statusLabel.appendPlainText(text)
        slicer.app.processEvents() # force update

#
# phantomLogic
#

class phantomLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """
    def addLog(self, text):
        logging.info(text)
        if self.logCallback:
            self.logCallback(text)

    def hasImageData(self,volumeNode):
        """This is an example logic method that
        returns true if the passed in volume
        node has valid image data
        """
        if not volumeNode:
            logging.debug('hasImageData failed: no volume node')
            return False
        if volumeNode.GetImageData() is None:
            logging.debug('hasImageData failed: no image data in volume node')
            return False
        return True

    def run(self, inputDirectory, outputDWIVolume, outputVolume, outputVOI, phantom_name):
        """
        Run the actual algorithm
        """
        self.addLog('Processing started')

        # check for valid inputs and outputs
        assert inputDirectory and outputDWIVolume and outputDWIVolume.IsA("vtkMRMLDiffusionWeightedVolumeNode")

        ijktoras = vtk.vtkMatrix4x4()
        ijktoras.SetElement(2,2,-1)
        outputDWIVolume.SetIJKToRASMatrix(ijktoras)
        outputVolume.SetIJKToRASMatrix(ijktoras)
        outputVOI.SetIJKToRASMatrix(ijktoras)

        # read directory
        self.addLog("Reading Directory")
        dir_output = dicom_util.read_dicomdir(inputDirectory)
        dwi_array = dir_output["dwi"]
        image_coordinate_system = dir_output["image_coordinate_system"]

        # update coordinate system of output volumes
        origin = (image_coordinate_system.pixel_x0_left_cm * -10, image_coordinate_system.pixel_y0_posterior_cm * -10, image_coordinate_system.min_superior_cm * -10)
        spacing = (image_coordinate_system.pixel_spacing_cm * 10, image_coordinate_system.pixel_spacing_cm * 10, image_coordinate_system.spacing_between_slices_cm * 10)

        outputDWIVolume.SetOrigin(origin)
        outputVolume.SetOrigin(origin)
        outputVOI.SetOrigin(origin)
        outputDWIVolume.SetSpacing(spacing)
        outputVolume.SetSpacing(spacing)
        outputVOI.SetSpacing(spacing)

        # Get phantom definition based on model and scalar type
        phantom_def = phantom_definitions.PHANTOM_CATALOG[phantom_name][dir_output["scalar_type"]]
        self.addLog("Using Phantom Definition {}".format(phantom_def["config"]["definition_name"]))

        # Get VOIs
        self.addLog('Generating VOIs')
        voi_dict = voi_analysis.get_vois(dwi_array, image_coordinate_system, phantom_def)
        label_map = voi_dict["label_map"].astype(dtype=np.float32)

        # create output scalar volume
        if dir_output["scalar_type"] == "ADC":
            b_values = dir_output["bvalues"]
            self.addLog('Calculating ADC')
            scalar_map = scalar_analysis.calculate_adc(dwi_array, b_values)
            # Update colors on label map
            label_map[label_map != 0] += ADC_COLORS_OFFSET     # adjust colors for Slicer color table

        elif dir_output["scalar_type"] == "T1":
            # Check if system is Windows
            WINDOWS = True if platform.system() == 'Windows' else False

            # If this is newer phantom, report the temperature
            if phantom_def["config"]["thermometry"]:
                temp = thermometry.get_temperature(dwi_array, image_coordinate_system)
                self.addLog('Temperature: {}'.format(temp))

            # By default Slicer can't use multiprocessing module,
            # Fix by temporarily reset stdin so that Slicer can use pool, reset below
            original_stdin = sys.stdin
            sys.stdin = open(os.devnull)
            try:
                self.addLog('Depending on operating system T1 calculation may take up to a few minutes, please be patient...')
                scalar_map = scalar_analysis.calculate_t1(dwi_array, dir_output["alphas"], dir_output["rep_time_seconds"], use_pool= not WINDOWS, clamp=(0, 4000), threshold=5)
            finally:
                sys.stdin.close()
                sys.stdin = original_stdin

        voi_map = label_map
        # update the output volumes
        slicer.util.updateVolumeFromArray(outputDWIVolume, dwi_array)
        slicer.util.updateVolumeFromArray(outputVolume, scalar_map)
        slicer.util.updateVolumeFromArray(outputVOI, voi_map)

        self.addLog('Processing completed')
        return dir_output["scalar_type"]

class phantomTest(ScriptedLoadableModuleTest):
    # TODO: call unit tests here, and maybe do more tests using slicer
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_phantom1()

    def test_phantom1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")
        #
        # first, get some data
        #
        import urllib
        downloads = (
            ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
            )

        for url,name,loader in downloads:
            filePath = slicer.app.temporaryPath + '/' + name
            if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
                logging.info('Requesting download %s from %s...\n' % (name, url))
                urllib.urlretrieve(url, filePath)
            if loader:
                logging.info('Loading %s...' % (name,))
                loader(filePath)
        self.delayDisplay('Finished with download and loading')

        volumeNode = slicer.util.getNode(pattern="FA")
        logic = phantomLogic()
        self.assertIsNotNone( logic.hasImageData(volumeNode) )
        self.delayDisplay('Test passed!')
