#Add necessary libraries
import os
import time
from PIL import Image, ImageDraw #Necessary libraries
import pandas as pd
import numpy as np

'''
Attention: Mask shoulbe be inside on main picture
'''

def Image_collection(FolderName): #File structure preparation
    tree = os.walk(FolderName) # List of files preparation
    tree = list(tree)

    CSVCollection = {}
    index = 0
    for path, dirs, files in tree:
        for j in files:
            index += 1
            CSVCollection[index] = [j, str(path) + '\\' + str(j)]
    return pd.DataFrame.from_dict(CSVCollection, orient='index', columns=['name', 'path'])

def Image_size (filename):
    image = Image.open(filename)
    return (str(image.size[0]) + ' x ' + str(image.size[1]))

def ImageMask_statistic(Image, Mask, position = [0,0]):
    '''
    :param Image: main image for correcture, type: PIL.Image
    :param Mask: mask of text for positioning (only the most bright pixels): type: ndarray
    :param position: start position for mask [X, Y], type = integer
    :return: mean bright of pixels inside mask [0] and outside mask [1]
    '''
    MaskHeight = len(Mask)
    MaskWidth = len(Mask[1])
    inmask, outofmask = 0, 0 #Накопленный черный и белый цвета
    pix = Image.load()

    for x in range(MaskWidth):
        for y in range(MaskHeight):
            red = pix[position[0] + x, position[1] + y][0]
            blue = pix[position[0] + x, position[1] + y][1]
            green = pix[position[0] + x, position[1] + y][2]
            S = (red + blue + green) // 3
            if Mask[y][x]:
                inmask += S
            else:
                outofmask += S
    inmask = inmask/np.sum(Mask)
    outofmask = outofmask / (np.size(Mask) - np.sum(Mask))

    return [inmask, outofmask]

def ImageMask_printmask(Image, Mask, position = [0,0], colour=[0,0,0]): #Procedure to show a Mask in the Image
    '''
    :param Image: main image for correcture, type: PIL.Image
    :param Mask: mask of text for positioning (only the most bright pixels): type: ndarray
    :param position: start position for mask [X, Y], type = integer
    :param colour: addition of colour to show mask (r, g, b), type = bite (0...255)
    '''
    MaskHeight = len(Mask)
    MaskWidth = len(Mask[1])
    pix = Image.load()
    draw = ImageDraw.Draw(Image)
    for x in range(MaskWidth):
        for y in range(MaskHeight):
            if Mask[y][x]:
                red = pix[position[0] + x, position[1] + y][0] + colour[0]
                green = pix[position[0] + x, position[1] + y][1] + colour[1]
                blue = pix[position[0] + x, position[1] + y][2] + colour[2]
                if red > 255: red = 255
                if green > 255: green = 255
                if blue > 255: blue = 255
                if red < 0: red = 0
                if green < 0: green = 0
                if blue < 0: blue = 0

                draw.point((position[0] + x, position[1] + y), (red, green, blue))

def ImageMask_positioning(Image, Mask, startposition = [0,0], radius = 2): #matching mask with the Image 
    '''
    :param Image: main image for correcture, type: PIL.Image
    :param Mask: mask of text for positioning: type: ndarray
    :param startposition: start position for mask [X, Y], type = integer
    :param radius: search radius to match mask in the image   px, type: integer
    :return: position [X, Y] with the best match mask in the image
    '''
    data = pd.DataFrame(0, index=range(1, 5), columns=['X', 'Y', 'inmask', 'outofmask'])
    index = 0
    for x in range(startposition[0]-radius, startposition[0]+radius+1):
        for y in range(startposition[1]-radius, startposition[1]+radius+1):
            res = ImageMask_statistic(Image, Mask, (x, y))
            data.loc[index] = {'X': x,
                               'Y': y,
                               'inmask': res[0],
                               'outofmask': res[1]}
            index += 1
    data = data.drop(0, axis=0)
    data['difference'] = 2*(data['inmask'] - data['outofmask']) / (data['inmask'] + data['outofmask'])
    if data['difference'].max() < 0.07:
        return [-1,-1]
    data = data[data['difference'] > 0]
    data = data.sort_values('inmask', ascending=False).reset_index().drop('index',axis=1)
    return [int(data.loc[0, 'X']), int(data.loc[0, 'Y'])]

def ImageMask_blurring(Image, Mask, position = [0,0], radius=10, exclmask=False): #blurring text in the image by mask
    '''
    :param Image: main image for correcture, type: PIL.Image
    :param Mask: mask of text for positioning, type: ndarray
    :param position: start position for mask [X, Y], type = integer
    :param radius: search radius to match mask in the image   px, type: integer
    :param exclmask: work only with pixels outside of mask (dont count current text pixels)
    '''
    MaskHeight = len(Mask)
    MaskWidth = len(Mask[1])
    pix = Image.load()
    draw = ImageDraw.Draw(Image)

    #исследование маски (смотрим толькос соседние точки)
    for xm in range(MaskWidth):
        for ym in range(MaskHeight):
            #rows selection
            minpos = ym-1
            maxpos = ym+2
            if minpos < 0: minpos = 0
            if maxpos > MaskHeight: maxpos = MaskHeight
            rows = np.array(range(minpos, maxpos), dtype=np.intp)
            # columns selection
            minpos = xm-1
            maxpos = xm+2
            if minpos < 0: minpos = 0
            if maxpos > MaskWidth: maxpos = MaskWidth
            columns = np.array(range(minpos, maxpos), dtype=np.intp)
            cropMask = Mask[rows[:, np.newaxis], columns]

            # If a mask pix is available near of image pixel 
            if cropMask.max() == 1:
                index = 0
                red, green, blue = 0, 0, 0
                xim = position[0] + xm
                yim = position[1] + ym
                #Count colours of near image pixels
                for dxim in range(xim-radius, xim+radius+1):
                    for dyim in range(yim-radius, yim+radius+1):
                        if exclmask:
                            xdm = dxim - position[0]
                            ydm = dyim - position[1]
                            if (xdm >= 0) and (xdm < MaskWidth-1) and (ydm >= 0) and (ydm < MaskHeight-1):
                                if Mask[ydm, xdm]: continue
                        red += pix[dxim, dyim][0]
                        green += pix[dxim, dyim][1]
                        blue += pix[dxim, dyim][2]
                        index += 1
                if index != 0:
                    red = red // index
                    green = green // index
                    blue = blue // index
                    draw.point((xim, yim), (red, green, blue))

def ApplicationLog (message, filename='-', path='-', details=[], start = False):
    global Logname
    dtl = ' '.join([str(item) for item in details])
    Logtext = ','.join([time.asctime(), filename, path, message, dtl])
    if start: #First record in the logfile
        with open(Logname, 'w') as Log:
            Log.write('Datetime, Image_Name, Image_Path, Progress, Details\n')
            Log.write(Logtext)
            Log.write('\n')
    else: #Regulag record to the logfile
        with open(Logname, 'a+') as Log:
            Log.write(Logtext)
            Log.write('\n')

#Settings
#StartFolder = input('Enter start folder name with path')
StartFolderName = 'D:\\eduar\\NAS_Disk\\Busineses\\Shell\\VideoCheck\\Inspection' #Main folder
#StartFolderName = 'C:\\Users\\eduar\\PycharmProjects\\Eddi_Learning\\VideocheckCleaner\\Files' #Test folder with limited qty of files
ImageExtensions = ['.jpg', '.JPG']
ImageSizes = ['768 x 576']
MasksDict = {'ShellVideocheck':['CheckMask_Videocheck.bmp', 'BlurMask_Videocheck.bmp', [276, 484]],
             'Olympus':['CheckMask_Olympus.bmp', 'BlurMask_Olympus.bmp', [572, 511]],
             '007': ['CheckMask_007.bmp', 'BlurMask_007.bmp', [517, 510]],
             '001': ['CheckMask_001.bmp', 'BlurMask_001.bmp', [515, 510]]}
Overwritecorrectfiles = True #Overwrite images corrected before
Logname = 'ProcessLog.txt'

#Start application
ApplicationLog('Start of application', start = True)

Filesdata = Image_collection(StartFolderName) #Selection of all files in the folder
ApplicationLog('Total files in selected folders',details=[Filesdata.shape[0]])

Filesdata['fileextension'] = Filesdata['name'].str[-4:] #Selection only images by extensions
Filesdata = Filesdata[Filesdata['fileextension'].isin(ImageExtensions)]
ApplicationLog('Total images',details=[Filesdata.shape[0]])

Filesdata['corrected'] = Filesdata['name'].str[-12:] #Selection corrected early images
if Overwritecorrectfiles:
    Filesdata = Filesdata[Filesdata['corrected'] != '_correct.JPG'] #Selecting only originals
    Filesdata = Filesdata.drop('corrected', axis=1)  #Deleting temp columns
else:
    Filesdata['dublicates'] = Filesdata['path'].str[:-4] #Deleting correcting early images from database
    Filesdata['dublicates'][Filesdata['corrected'] =='_correct.JPG'] = Filesdata['path'].str[:-12]
    Filesdata.drop_duplicates(subset=['dublicates'], keep=False, inplace=True)
    Filesdata = Filesdata.drop('corrected', axis=1)  # Удаление служебных колонок
    Filesdata = Filesdata.drop('dublicates', axis=1)  # Удаление служебных колонок

Filesdata['filesize'] = Filesdata['path'].apply(lambda x: Image_size(x)) #Selection endoscopic images by size -
Filesdata = Filesdata[Filesdata['filesize'].isin(ImageSizes)]
ApplicationLog('Total endoscope pictures',details=[Filesdata.shape[0]])

print('Total endoscope pictures',Filesdata.shape[0])
print('Extensions found - ', Filesdata['fileextension'].unique(),
      '\nSizes found - ', Filesdata['filesize'].unique())

# Reading of all Masks
MaskArray = pd.DataFrame([[0]*4]*len(MasksDict)) #columns=['name', 'checkmask', 'blurmask', 'startposition']
index = 0
for key, value in MasksDict.items():
    MaskArray[index]['name'] = key
    MaskArray[index]['checkmask'] = np.array(Image.open(value[0]))
    MaskArray[index]['blurmask'] = np.array(Image.open(value[1]))
    MaskArray[index]['startposition'] = value[2]
    index += 1

#crop/minimize list files for test
# Filesdata = Filesdata[Filesdata.index < 5]

# Image processing
ApplicationLog('Start processing images, in progress',details=[Filesdata.shape[0]])
indexpb = 0 #Progressbar counter

for row in Filesdata.values:
    indexpb += 1
    progress = indexpb*100//len(Filesdata)  # Progressbar index, from 0 to 100
    print('\rYou have finished %3d%%' % progress, end='', flush=True) #Simple progressbar

    ApplicationLog('Start of image processing', row[0], row[1])
    imagesave = False
    image = Image.open(row[1])
    for index in range(len(MaskArray)):
        ApplicationLog('Start of mask processing', row[0], row[1], [MaskArray[index]['name']])
        Xbest, Ybest = ImageMask_positioning(image, MaskArray[index]['checkmask'],
                                             startposition=MaskArray[index]['startposition'], radius=5)
        if Xbest == -1:
            ApplicationLog('Attension: Mask doesn\'t processed', row[0], row[1], [MaskArray[index]['name']])
            continue
        if (Xbest != MaskArray[index]['startposition'][0]) or (Ybest != MaskArray[index]['startposition'][1]):
            ApplicationLog('Attension: Mask position was corrected', row[0], row[1],
                           [MaskArray[index]['name'], ' - new position: ', Xbest, Ybest])
        ImageMask_blurring(image, MaskArray[index]['blurmask'], position=[Xbest, Ybest], radius=5, exclmask=True)
        imagesave = True
    if imagesave:
        filename = row[1][:-4] + '_correct.JPG'
        image.save(filename)
        ApplicationLog('Image processing finished', row[0], row[1])
    else:
        ApplicationLog('Image doesn\'t changed', row[0], row[1])

ApplicationLog('Application finished, progress 100%')
print()
print('You have finished 100%')