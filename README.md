# Developer Getting Started

These instructions are tested on Ubuntu 18.04 and Python 2.7.15rc1.

## apt-get install

```
sudo apt-get install python-virtualenv python-dev python-pip python-matplotlib python-tk
```

## install virtualenv

use `--system-site-packages` to use the apt-get installed matplotlib:

```
virtualenv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt
```

To generate a new requirements.txt run `pip freeze --local > requirements.txt`
Remove pkg-resources==0.0.0 from requirements.txt

## run the unit tests

e.g.

```
(venv) jdimatteo@bb8:~/dev/PhantomAnalysis$ ./test.py
```

You can show intermediate results with some tests, e.g.

```
(venv) jdimatteo@bb8:~/dev/PhantomAnalysis$ export PHANTOM_ANALYSIS_SHOW_INTERMEDIATE_RESULTS=1
(venv) jdimatteo@bb8:~/dev/PhantomAnalysis$ ./test_voi.py
```


## running module in Slicer

These instructions were tested on 4.10.1, and can be installed from https://download.slicer.org/ .

1. Start slicer
2. Install OpenCV:
    1. Select View > Extension Manager > Install Extensions
    2. Search for "openCV"
    3. Install SlicerOpenCV 
    4. Restart Slicer
3. On linux/macOS install Scipy:
    1. Select View > Python Interactor
    2. In the Python shell enter
        ```
         >>> from pip._internal import main as pipmain
         >>> pipmain(['install', 'scipy'])
        ```
    3. Restart Slicer
4. Load the phantom_analysis library:
    1. Select Edit > Application Settings
    2. Select modules from left panel
    3. Next to "Additional module paths" click on Add
    4. Navigate to PhantomAnalysis/phantom_analysis and click Choose
    5. Click OK and restart Slicer
5. Load the extension:
    1. Module Dropdown > Developer Tools > Extension Wizard
    2. click "Select Extension"
    3. choose the `slicer_extension` directory
    4. In the next window that pops up, uncheck everything but phantom and click "Yes"
        * check "Add selected module to search paths"
6. Calculate the ADC or T1 and get the volumes of interest with the loaded extension
    1. open the "Phantom Analysis" module: Module Dropdown > Phantom > Phantom Analysis
    2. under parameters, for "DICOM Input", select the directory of DICOM files
    3. under parameter, for "Output DWI Volume", select "Create new Volume as..." and set the name to DWI, click OK
    3. under parameters, for "Output Scalar Volume", select "Create new Volume as..." and set the name to ADC or T1, click OK
    4. under parameters, for "Output VOI", select "Create new LabelMapVolume as..." and set the name to VOI, click OK
    5. Click the "Apply" button
    6. Hover over individual pixes to view the ADC/T1 in the data probe
    7. Change the display volume to see other outputs
    ![screenshot](/images/setup/ADC.png?raw=true)

## FAQ

### How do you uninstall a module or reload it from scratch?

1. shutdown slicer
2. delete (or backup/move) the config, e.g. `mv ~/.config/NA-MIC /tmp`
3. start slicer

### additional tools/links

* aeskulap is useful for simple DWI viewing: `sudo apt-get install aeskulap`
* DicomBrowser: install from https://wiki.xnat.org/xnat-tools/dicombrowser .deb download and then run `DicomBrowser`
* DICOM standard documentation: https://dicom.innolitics.com (e.g. try searching for "Image Position Patient")
