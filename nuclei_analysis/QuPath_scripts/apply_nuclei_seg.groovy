def entry = getProjectEntry()
print entry.getImageName()
def imgName = entry.getImageName().tokenize('.')[0]
print imgName

def gson = GsonTools.getInstance(true)

def json = new File("/home/l_le-bescond/Documents/Daisy-code-public/Results/nuclei/nuclei_annotations/nuclei_${imgName}.json").text


// Read the annotations
def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType()
def deserializedAnnotations = gson.fromJson(json, type)

// Set the annotations to have a different name (so we can identify them) & add to the current image
// deserializedAnnotations.eachWithIndex {annotation, i -> annotation.setName('New annotation ' + (i+1))}   # --- THIS WON"T WORK IN CURRENT VERSION
addObjects(deserializedAnnotations)