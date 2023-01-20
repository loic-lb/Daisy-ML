setImageType('BRIGHTFIELD_H_DAB');
setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "1.06636738, 1.14050912, 0.7931235", "Stain 2" : "DAB", "Values 2" : "0.41065507, 1.23342783, 1.35591711", "Background" : " 255 255 255"}');
selectAnnotations();
addShapeMeasurements("AREA", "LENGTH", "CIRCULARITY", "SOLIDITY", "MAX_DIAMETER", "MIN_DIAMETER")
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons": 0.325,  "region": "Circular tiles",  "tileSizeMicrons": 30.0,  "colorOD": false,  "colorStain1": false,  "colorStain2": true,  "colorStain3": false,  "colorRed": false,  "colorGreen": false,  "colorBlue": false,  "colorHue": false,  "colorSaturation": false,  "colorBrightness": false,  "doMean": true,  "doStdDev": true,  "doMinMax": true,  "doMedian": true,  "doHaralick": false,  "haralickDistance": 1,  "haralickBins": 32}');
