"""
Operations for diffraction image reduction

Includes:

"""
import os
from pathlib import Path
import numpy as np
from numpy.polynomial.chebyshev import chebval

import scipy
import scipy.io
from scipy.optimize import curve_fit, basinhopping
from scipy import integrate, signal
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad

import pandas as pd
import fabio
import re

import matplotlib.pyplot as plt

import pyFAI
from pyFAI.multi_geometry import MultiGeometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as AzInt

from .peakShapes import voigtFn, gaussFn
from .utils import folder_select

from xrdc import geometry
# To-Do: make type alisases (list[Path's])


def load_image(path: Path=Path('')) -> np.ndarray:
    """load_image loads images based on file extension
        
    :param path: path to single image file, defaults to Path('')
    :type path: Path, optional
    :return: image array data
    :rtype: np.ndarray
    """
    try:
        path = Path(path)
        # check if path exists (path.exists())
    except:
        print('could not verify path')
        im = np.array([0])

    if path.suffix in ['.tif', '.tiff']:
        # open tiff image
        im = fabio.open(path)
        # input image object into a numpy array
        imArray = im.data
    elif path.suffix in ['.raw']:
        # extract raw file
        im = open(path, 'rb')
        arr = np.fromstring(im.read(), dtype='int32')
        im.close()
        
        #raw requires prompting for dimensions, hard code for now
        arr.shape = (195, 487)
        imArray = np.array(arr)
        
    return imArray

def parse_calib(filename):
    """parse_calib Parses calibration from wxDiff
    
    :param filename: [description]
    :type filename: [type]
    :return: [description]
    :rtype: [type]
    """
    file=open(filename,'r')
    data = []
    with file as inputfile:
        for line in inputfile:
            data.append(line.strip().split('\n'))
    bcenter_x = float(data[6][0][10:])
    bcenter_y = float(data[7][0][10:])
    detector_dist = float(data[8][0][12:])
    detect_tilt_alpha = float(data[9][0][18:])
    detect_tilt_delta = float(data[10][0][18:])
    wavelength = float(data[11][0][11:])
    return detector_dist, detect_tilt_alpha, detect_tilt_delta, wavelength, bcenter_x, bcenter_y


center = (183, 245)
det = pyFAI.detectors.Pilatus100k()
sampleExpInfo = {'dist': 0.55, # meters
                'poni1': center[0] * det.pixel1,
                'poni2': center[1] * det.pixel2,
                'detector': det,
                'wavelength': 7.2932E-11, # meters?..
                'orientation': 'horizontal'
                }

def create_single() -> ():
    """create_single [summary]
    
    Creates single scan, look to see if normal azimuthal integrator 
    operates similarly to multigeometry for save_qchi, save_Itth
    """


def create_scan(imgList: list, specPath: Path, 
                expInfo: dict=sampleExpInfo) -> (MultiGeometry, list):
    """Create multigometry object containing images and coordinates 
    """
    spec = pd.read_csv(specPath)
    spec = spec.rename(columns=lambda x: x.strip())
    ais = []
    imgs = []

    for im in imgList:
        dat = load_image(im)    

        if expInfo['orientation'].lower() == "vertical":
            imgs.append(np.rot90(dat, k=1))
        elif expInfo['orientation'].lower() == 'horizontal':
            imgs.append(dat)
        
        scanNo = int(re.search(r'(\d{4})(\.raw)', str(im)).group(1))

        intInfo = {x: expInfo[x] for x in ['poni1', 'poni2', 'dist',
                                            'detector', 'wavelength']}

        ai = AzInt(**intInfo, 
                    rot2=np.pi/180*float(spec['TwoTheta'][scanNo]),
                    rot3=-np.pi/2)
        ais.append(ai)
    ##############################################################################
    ##############################################################################
    ##############################################################################
    mg = MultiGeometry(ais, unit="2th_deg", 
                        radial_range=(16, 28), azimuth_range=(-20, 20))
    ##############################################################################
    ##############################################################################
    ##############################################################################


    return mg, imgs

def generate_mask(img: list, xmin: int=0, xmax: int=-1, 
                    ymin: int=0, ymax: int=-1) -> np.ndarray:
    """Generates a mask the size of the image 
        where 1's are invaid, 0's are valid.  
    """
    mask = np.ones(np.shape(img))
    mask[ymin:ymax, xmin:xmax]
    return mask

def save_qchi(mg: MultiGeometry, imgs: list, mask: np.ndarray,
                spath: Path, template: str):
    # Create directory
    os.makedirs(spath, exist_ok=True)

    _, ax = plt.subplots(1,1)

    # Integrate image
    res2d = mg.integrate2d(imgs, 1000,360, lst_mask=mask)
    # gives 3 arrays.  First is intensities with shape (arg2, arg1)  
    # Second is 2th range, third is azimuthal range

    ax.pcolormesh(res2d[1], res2d[2], np.log(res2d[0]), cmap = 'viridis')
    plt.rc('text')
    plt.rc('font')

    ax.set_xlim(0,65)
    ax.set_xlabel('Radial Angle (2theta)')
    ax.set_ylabel('Azumuthal Angle (chi)')

    plt.savefig(spath + template + '_figures.png')

    scipy.io.savemat(spath + template + '_Qchi.mat', 
                     {'Q':res2d[1], 'chi':res2d[2], 'cake':res2d[0]})

def save_Itth(mg: MultiGeometry, imgs: list, mask: np.ndarray,
                spath: Path, template: str=''):
    # Create directory
    os.makedirs(spath, exist_ok=True)
    
    _, ax = plt.subplots(1,1)
    
    int1d = mg.integrate1d(imgs, 10000, lst_mask=mask)
    ax.plot(int1d[0], int1d[1])
    ax.set_ylabel('Intensity')
    ax.set_xlim(0,65)

    int1List = list(int1d)
    int1df = pd.DataFrame(data=int1List[1], index=int1List[0])
    int1df.to_csv(spath + template + '_1D.csv')

    plt.close()
    
def save_dict(data: dict, spath: Path, template: str=''):
    """Save data to spath+template, append if already exists
    """
    os.makedirs(spath, exist_ok=True)

    if os.path.exists(spath+template):
        df = pd.read_csv(spath+template+'_params.csv')
        dfnew = pd.DataFrame(data)
        df = df.append(dfnew)
        df.to_csv(spath+template+'_params.csv')
    else:
        # create new file
        pd.DataFrame(data).to_csv(spath+template+'_params.csv')

def summarize_params(csv_path: Path, 
                    search_str: str='', save_template: str=''):
    """Generate summary csv unique to Alan's plans. 

    pick out peaks with x-locations closest to provided tth locations
    append to area_summary.csvname   """
    csvList = folder_select(csv_path, search_str)
    result = None
    for f in csvList:
        df = pd.read_csv(f)
        df = df.rename(columns={df.columns[0]: 'param'})

        # whacky sorting thing
        df = df.set_index('param')
        df = df.sort_values(by='x0', axis=1)
        df.columns = [f'curve {j}' for j in range(len(df.columns))]
        df = df.rename_axis('param').reset_index()

        # Flatten and group.  Kind of a 'join on'
        melt = pd.melt(df, id_vars=df.columns[0], 
                        value_vars=df.columns[1:], value_name=f.stem)

        if result is None:
            result = melt
        else:
            result = result.merge(melt, on=('param', 'variable'))

    result.to_csv(Path(csv_path) / save_template)

def _get_params(d2):
    res = []
    keys = ['x0', 'y0', 'I', 'alpha', 'gamma']
    for k in keys:
        res.append(d2[k])
    return res

def paramdict_get_params(d):
    if d is None or len(d) == 0:
        return None
    res = []
    for _, d2 in d.items():
        res = res + _get_params(d2)
    return res

def paramdict_get_bounds(d):
    leftBnd = []
    rightBnd = []
    if d is None or len(d) == 0:
        return None
    res = []
    for _, d2 in d.items():
        leftBnd += d2['leftBnd']
        rightBnd += d2['rightBnd']
    return tuple([leftBnd, rightBnd])

def gen_curve_fits(x, y, params: dict,
                    peakShape: str='Voigt'):
    """
    Returns lists of y values for each fit curve given by params, with
    an additional final element that is the sum of the individual fits.
    """
    ncurves = len(params)
    if peakShape == 'Voigt':
        func = voigtFn
    elif peakShape == 'Gaussian':
        func = gaussFn
    else: 
        print('no peak shape chosen')

    popt = []
    curve_ys = []
    for j, (_, l) in zip(range(ncurves), params.items()):
        temp = _get_params(l)
        #temp = list(l.values())

        popt += temp
        curve_ys.append(func(x, *_get_params(l)))
        #curve_ys.append(func(x, *l.values()))

    curve_ys.append(func(x, *popt))
    return curve_ys

def plot_curve_fit(x, y, params: dict,
                    peakShape: str='Voigt'):
    """
    Plots y(x), individual fit curves, and sum of fit curves.

    Returns y values for each fit curve.
    """
    curve_ys = gen_curve_fits(x, y, params, peakShape)
    ncurves = len(params)
    for j, yj in enumerate(curve_ys[:-1]):
        plt.plot(x, yj, '.', alpha=0.5, 
                label='opt. curve {:.0f}'.format(j))

    plt.plot(x, y, marker='s', color='k', label='data')
    plt.plot(x, curve_ys[-1], color='r', label='combined data')
    
    plt.legend()
    return curve_ys

#def plot_curve_fit(x, y, params: dict,
#                    peakShape: str='Voigt'):
#    """
#    Plots y(x), individual fit curves, and sum of fit curves.
#
#    Returns y values for each fit curve.
#    """
#    ncurves = len(params)
#    if peakShape == 'Voigt':
#        func = voigtFn
#    elif peakShape == 'Gaussian':
#        func = gaussFn
#    else: 
#        print('no peak shape chosen')
#    plt.figure(figsize=(8,8))
#
#    popt = []
#    curve_ys = []
#    for j, (_, l) in zip(range(ncurves), params.items()):
#        temp = list(l.values())
#        popt += temp
#        plt.plot(x, func(x, *l.values()), '.', alpha=0.5, 
#                label='opt. curve {:.0f}'.format(j))
#        curve_ys.append(func(x, *l.values()))
#
#    curve_ys.append(func(x, *popt))
#    plt.plot(x, y, marker='s', color='k', label='data')
#    plt.plot(x, func(x, *popt), 
#            color='r', label='combined data')
#    
#    plt.legend()
#    return curve_ys

def save_curve_fit(x, y, params: dict, spath: Path, 
                    template: str='', 
                    peakShape: str='voigt'):
    """Plot curves vs data (x,y) for given curve parameters
    """

    plot_curve_fit(x, y, params)

    plt.savefig(spath + template + 'peakAt_' + 
                    '{:.3f}'.format(x[np.where(y==np.max(y))[0][0]]) + '.png')
    plt.close()        

def integrate_1d_mg(mg: MultiGeometry, imgs: list, 
                    mask: np.ndarray=None) -> np.ndarray:
    """Perform 1D integration on many images
    first axis is (2th or q), second is intensity
    """
    print(np.shape(imgs))
    int1d = mg.integrate1d(imgs, 10000, lst_mask=mask)
    
    return np.array(int1d)

def restrict_range(x, y, xRange: list=None, 
                    yRange: list=None) -> (np.ndarray, np.ndarray):
    """Return the subset of x, y that falls within xRange and yRange

    """

def bkgd_sub(x, y, downsample_int: int=50, 
            print_plot: bool=False) -> (np.ndarray, np.ndarray):
    """Subtract background with modified chebyshev polynomial
    """
    print("downsample:", downsample_int)
    def downsamp(size, x, y):
        """Downsample data based on size parameter.  Return downsampled data
        """
        yNew = y[::size] 
        xNew = x[::size] 
        
        return xNew, yNew
        
    def chebFunc(x, *params):
        """
        Modified chebyshev function with 1/x 
        """
        params = params[0]
        y = chebval(x, params[:4])
        E = params[4]
        y = y + E/x
        return y


    def objFunc(*params):
        """
        implement cost function for modified chevbyshev function
        varies cost to emphasize low Q region
        Scoping pulls trimmed x, y data from surrounding scope... this is bad

        return: cost of fit 
        """
        
        params = params[0]
        J = 0
        fit = chebval(downX, params[:4])
        E = params[4]
        fit = fit + E / downX

        for i in range(len(downY)):
            if downX[i] < 1:    # Treat low Q equally
                J = J + (downY[i] - fit[i])**4
            else:
                if downY[i] < fit[i]:
                    J = J + (downY[i] - fit[i])**4
                if downY[i] >= fit[i]:
                    J = J + (downY[i] - fit[i])**2
        return J
        
    # Create a sparse data set for fitting
    downX, downY = downsamp(downsample_int, x, y)
    
    # Remove points above median + 10% of range
    medianY = np.median(downY)
    rangeY = np.max(downY) - np.min(downY)
    downX = downX[downY <= (medianY + 0.1*rangeY)]
    downY = downY[downY <= (medianY + 0.1*rangeY)]
    
    # Re-append end values to try and fix boundary conditions
    downX = np.append(np.append(x[0:30:5], downX), x[-30:-1:5])
    downY = np.append(np.append(y[0:30:5], downY), y[-30:-1:5])

    x0 = [1,1,1,1,1]
    #appears to converge quickly, take 10 iterations rather than 100
    result = basinhopping(objFunc, x0, niter=100) 
    bkgd_sparse = chebFunc(x, result.x)
    # create function that interpolates sparse bkgd
    f = interp1d(x, bkgd_sparse, kind='cubic', bounds_error=False)

    # expressed background values
    bkgd1 = f(x)
    subDataY = y - bkgd1

    # Dump any nan values in all data
    finalDataY = subDataY[~np.isnan(subDataY)]
    finalDataX = x[~np.isnan(subDataY)]
    bkgd = bkgd1[~np.isnan(subDataY)]

    # Offset to make all values > 0 
    if np.min(subDataY) < 0:
        finalDataY = finalDataY + np.absolute(np.min(finalDataY))

    if print_plot:
        return finalDataX, finalDataY, bkgd1
    else: 
        return finalDataX, finalDataY

def bkgd_sub_GP(x, y, downsample_int: int=50): 
    pass

defaultFitOpts = {'peakShape': 'voigt', # ('voigt', 'gaussian')
                    'fitMode': 'fixed',
                    'numCurves': 4,
                    }


def peak_edge_cost(centers, x, buffer = 2., padding = 20):
    """
    Impose a penalty for Voigt profiles with center near the edge of a
    block.
    """
    start, end = x[0], x[-1]
    range = x[-1] - x[0]
    buffer = min(range / 4, buffer)
    centers = np.array(centers)
    # Linearly extend x so that we can evaluate cost for peaks with centers outside
    # the fitting range
    dx = x[1] - x[0]
    x = np.pad(np.array(x), (padding, padding))
    padramp = dx * np.ones(padding).cumsum()
    x[:padding] = -padramp[::-1] + x[padding]
    x[-padding:] = padramp + x[-padding - 1]
    offset_left = start - centers + buffer
    cmpleft = (offset_left) * np.heaviside(offset_left, 0)
    offset_right = centers - end + buffer
    cmpright = (offset_right) * np.heaviside(offset_right, 0)
    costs = np.max(np.vstack((cmpleft, cmpright)), axis = 0)
    return costs

def regularized_curvefit(func, x, y, reg_scale = 1e-3, reg_edge = 1, **kwargs):
    """
    Given a Voigt model function, add a regularization that penalizes
    negative offsets and then call curve_fit.
    """
    # TODO support shapes other than Voigt
    print(x)
    def model(xx, *params):
        #'x0', 'y0', 'I', 'alpha', 'gamma'
#        print(xx, 'ffooooo')
#        print(params)
        offsets = params[1::5]
        centers = params[::5]
        #return np.abs(func(xx, *params) - y) + reg_scale * (-np.min(list(offsets) + [0])) # penalize negative offsets
        # TODO This may be better in the case of data that isn't
        # properly background subtracted, but testing so far gives
        # similar results
        loss_scale = np.abs(func(xx, *params) - y).mean()
        # penalize negative offsets and peaks that are close to block edges
        return np.abs(func(xx, *params) - y) +\
            loss_scale * reg_scale * max(0, np.min([0, sum(offsets)]) - np.min(list(offsets))) +\
            reg_edge * peak_edge_cost(centers, x).sum()

    return curve_fit(model, x, np.zeros_like(y), **kwargs)

def fit_peak(x: np.ndarray=np.ones(5,), y: np.ndarray=np.ones(5,),
                peakShape: str=defaultFitOpts['peakShape'], 
                fitMode: str=defaultFitOpts['fitMode'],
                numCurves: int=defaultFitOpts['numCurves'],
                noise_estimate: np.ndarray = None,
                background: np.ndarray = None,
                stdratio_threshold: float = 2,
                fit_sigma: bool = False,
                **kwargs
            ) -> (dict, list):
    """Fit peak segment with specified peak shape and method
    return dictionary with peak information
    finalParams: all curve info
    FWHM: calculated FWHM's for each curve
    fit_sigma: if True, and if noise_estimate is provided, use given
    uncertainties to weight residuals in least squares optimization.

    
    """
    if len(x) == 0:
        return None, None 
    xRange = np.ptp(x)
    maxInd = np.where(y == np.max(y))[0][0]

    # Initialize guess and fitting limits for each function
    if peakShape == 'Voigt':
        func = voigtFn
        # x0 and I values to be replaced during loop
        guessTemp = [0, np.min(y), 0,
                        xRange / 5, xRange / 5 ]
        # x bounds to be replaced
        # [x0, y0, I, alpha, gamma]
        boundLowTemp = [x[0], -np.max(y)/4, 0., 0., 0.]
        boundUppTemp = [x[-1], np.inf, np.inf, xRange/1, xRange/1]

        boundLowerPart = [x[0]-0.05*xRange, -np.max(y)/4, 0, 0, 0]
        boundUpperPart = [x[-1]+0.05*xRange, np.inf, np.inf, 
                            xRange/1, xRange/1]
    elif peakShape == 'Gaussian':
        func = gaussFn     
        # x0 and I values to be replaced during loop
        guessTemp = [0, np.min(y), 0,
                        xRange / 10]
        # x bounds to be replaced
        # [x0, y0, I, sigma]
        boundLowTemp = [x[0]-0.05*xRange, -np.max(y)/4, 0., 0.]
        boundUppTemp = [x[-1]+0.05*xRange, np.inf, np.inf, xRange/4]

        boundLowerPart = [x[0], -np.max(y)/4, 0, 0]
        boundUpperPart = [x[-1], np.inf, np.inf, xRange/4]       

    # Initialize guess params and bounds for number of curves
    boundUpper = []
    boundLower = []
    guess = []
        
    # init fit array to compare residual 
    fit = np.ones_like(y) * np.min(y)
    
    # parameter array: [x0, y0, Intensity, alpha, gamma]
    # set up guesses
    curveCnt = 0
    resid = fit - y
    # To-do: figure out better way to zero shift
    errorCurr = np.mean(np.absolute(resid) / (y+1))
    print('Peak at {0}, start iteration with error = {1}'.format(x[maxInd], errorCurr))
    def keep_fitting(curveCnt):
        # fit more peaks as long as the residual RMSE is below-threshold
        # (or up to numCurves, if no noise estimate is given).
        # If residual increased, stop fitting
#        try:
#            if (resid**2).mean() >= (old_resid**2).mean():
#                return False
#        except NameError:
#            pass
        if curveCnt >= numCurves:
            return False
#        # TODO removing the below may cause failure in some cases
#        if curveCnt == 0:
#            return True
        if noise_estimate is not None:
            ratio = np.sqrt(((resid**2).mean() / (noise_estimate**2).mean() ))
            #ratio = np.sqrt(((resid / (np.abs(noise_estimate) + 1e-6))**2).mean())
#            print('ratio', ratio)
#            print('resid', resid)
#            print('noise', noise_estimate)
            print(ratio)
            if ratio < stdratio_threshold:
                return  False
        return True
    #while curveCnt < numCurves: # and errorCurr > 0.001: 
    while keep_fitting(curveCnt): # and errorCurr > 0.001: 
        # place peak at min residual
        xPosGuess = x[np.argmin(resid)]
        guessTemp[0] = xPosGuess
        guessTemp[2] = np.max(resid) - np.min(resid)
            
        # Deal with edge cases.. 
        xPosLow = xPosGuess - 0.01*xRange
        if xPosLow < x[0]: xPosLow = x[0]
        xPosHigh = xPosGuess + 0.01*xRange
        if xPosHigh > x[-1]: xPosHigh = x[-1]
            
        # Update temp bounds to be close to position guess
        boundLowTemp[0] = xPosLow
        boundUppTemp[0] = xPosHigh

        boundTemp = tuple([boundLowTemp, boundUppTemp])
        try: # Fit according to residual
            if noise_estimate is not None and fit_sigma:
                poptTemp, pcovTemp = regularized_curvefit(func, x, -resid, 
                    bounds=boundTemp, p0=guessTemp, sigma = noise_estimate, **kwargs)  
            else:
                poptTemp, pcovTemp = regularized_curvefit(func, x, -resid, 
                    bounds=boundTemp, p0=guessTemp, **kwargs)  
            #print('Fit to residual at {0}'.format(xPosGuess))
        except RuntimeError as e:
            print(e) 
            poptTemp = guessTemp

        # Check to see if error decreased
        guessHold = guess + list(poptTemp)
        fit = func(x, *guessHold)
        old_resid = resid
        resid = fit - y
        errorNew = np.mean(np.absolute(resid) / (y+1))

        # build guess real guess array, update fit
        guess = guessHold

        # concatenate lists for bounds for real fit
        boundLower += boundLowerPart 
        boundUpper += boundUpperPart

        # Combine bounds into tuple for input
        bounds = tuple([boundLower, boundUpper])
       
        # Calculate residual, increment, and print error information 
        errorCurr = errorNew 
        print('Peak at {0}, iteration {1}: error = {2}'.format(x[maxInd], 
                                                    curveCnt, errorCurr))
        curveCnt+=1

    ####################################### Now fit whole peak with acq curves 
    # Fit full curve, refining guesses
    try:
        # Curve fit function call using guess and bounds
        print('bounds', bounds)
        if noise_estimate is not None and fit_sigma:
            popt, pcov = regularized_curvefit(func, x, y, sigma = noise_estimate,
                                        bounds=bounds, p0=guess, **kwargs)
        else:
            popt, pcov = regularized_curvefit(func, x, y, 
                                        bounds=bounds, p0=guess, **kwargs)
    except RuntimeError as e:
        print(e) 
        popt = np.array(guess)
    except UnboundLocalError:# no peaks were fit
        return None, None 
       

    return generate_peakparams(x, y, popt, peakShape, curveCnt, func, bounds)

def generate_peakparams(x: np.ndarray, y: np.ndarray,
        popt, peakShape, curveCnt, func, bounds):
    # Calculate FWHM, area for each peak fit
    if peakShape == 'Voigt':
        FWHM = []
        names = ['x0', 'y0', 'I', 'alpha', 'gamma', 'leftBnd', 'rightBnd']
        c0 = 2.0056
        c1 = 1.0593
        for i in range(0, len(popt), 5): # grab fwhm for each peak
            fg = 2*popt[i+3]*np.sqrt(2*np.log(2))
            fl = 2*popt[i+4]
            phi = fl / fg
            FWHM.append(fg * (1-c0*c1 + np.sqrt(phi**2 + 2*c1*phi + (c0*c1)**2)))

    elif peakShape == 'Gaussian':
        FWHM = []
        names = ['x0', 'y0', 'I', 'alpha', 'gamma',
            'leftBnd', 'rightBnd']
        for i in range(0, len(popt), 4): # grab fwhm for each peak
            FWHM.append(2*popt[i+3]*np.sqrt(2*np.log(2)))

    ###########################################################################
    ### Organizing output
    ###########################################################################
    
    # Organize final parameters into dict
    curveParams = {}
    derivedParams = {}
    leftBnd = bounds[0]
    rightBnd = bounds[1]
    for j in range(curveCnt):   # Plot each individual curve
        L = int(0 + j * len(popt) / curveCnt)  # Sep popt array
        R = int((j+1) * len(popt) / curveCnt)

        area, err = quad(func, x[0], x[-1], tuple(popt[L:R]))

        tempd = {n:v for (n,v) in zip(names,list(popt[L:R]) + [leftBnd[L:R], rightBnd[L:R]])}
        curveParams[f'curve {j}'] = tempd
        # Init derived curve  sub-dict
        derivedParams[f'curve {j}'] = {}
        derivedParams[f'curve {j}']['FWHM'] = FWHM[j]
        derivedParams[f'curve {j}']['area'] = area
        derivedParams[f'curve {j}']['area_err'] = err
        derivedParams[f'curve {j}']['x0'] = curveParams[f'curve {j}']['x0']
    
    return curveParams, derivedParams


def refine_peaks(x: np.ndarray=np.ones(5,), y: np.ndarray=np.ones(5,),
                peakShape: str=defaultFitOpts['peakShape'], 
                fitMode: str=defaultFitOpts['fitMode'],
                numCurves: int=defaultFitOpts['numCurves'],
                noise_estimate: np.ndarray = None,
                background: np.ndarray = None,
                stdratio_threshold: float = 2,
                fit_sigma: bool = False,
                guess = None,
                bounds = None,
                **kwargs
            ) -> (dict, list):
    """
    """
    if guess is None or bounds is None:
        raise ValueError
    if peakShape == 'Voigt':
        func = voigtFn
        nparam = 5
    elif peakShape == 'Gaussian':
        raise NotImplementedError
        func = gaussFn
    else: 
        print('no peak shape chosen')
    curveCnt = len(guess) // nparam

    try:
        # Curve fit function call using guess and bounds
        if noise_estimate is not None and fit_sigma:
            popt, pcov = regularized_curvefit(func, x, y, sigma = noise_estimate,
                                        p0=guess, bounds = bounds, **kwargs)
        else:
            popt, pcov = regularized_curvefit(func, x, y, 
                                        p0=guess, bounds = bounds, **kwargs)
    except RuntimeError as e:
        print(e) 
        popt = np.array(guess)
        print('peak fitting failed')
    except TypeError as e:
        print(e)
        popt = np.array(guess)
        print('peak fitting failed, possibly due to excessive parameter count')
    except UnboundLocalError:# no peaks were fit
        return None, None 
    return generate_peakparams(x, y, popt, peakShape, curveCnt, func, bounds)

def bayesian_block_finder(x: np.ndarray=np.ones(5,), y: np.ndarray=np.ones(5,),
        norm = 1):
    """bayesian_block_finder performs Bayesian Block analysis on x, y data.

    see Jeffrey Scargle's papers at doi: 10.1088/0004-637X/764/2/167

    :param x: array of x-coordinates
    :type x: numpy.ndarray
    :param y: array of y-values with same length as x
    :type y: numpy.ndarray
    """
    # TODO check scargle paper to see the principled way of normalizing
    # (if any)
    y = norm * y / np.mean(y)
    print('norm', norm)
    data_mode = 3
    numPts = len(x)
    if len(x) != len(y):
        raise ValueError('x and y are not of equal length')

    tt = np.arange(numPts)
    nnVec = []

    sigmaGuess = np.std(y[ y <= np.median(y)])
    cellData = sigmaGuess * np.ones(len(x))

    ncp_prior = 0.5


    ## To implement: trimming/choosing of where to start/stop
    ## To implement: timing functions

    cp = []
    cntVec = []

    iterCnt = 0
    iterMax = 10

    while iterCnt <= iterMax:
        best = []
        last = []

        for r in range(numPts):
            # y-data background subtracted
            sumX1 = np.cumsum(y[r::-1]) 
            sumX0 = np.cumsum(cellData[r::-1]) # sigma guess
            fitVec = (np.power(sumX1[r::-1],2) / (4 * sumX0[r::-1]))

            paddedBest = np.insert(best,0,0)
            best.append(np.amax( paddedBest + fitVec - ncp_prior ))
            last.append(np.argmax( paddedBest + fitVec - ncp_prior ) )

            # print('Best = {0},  Last = {1}'.format(best[r], last[r]))

        # Find change points by peeling off last block iteratively
        index = last[numPts-1]

        while index > 0:
            cp = np.concatenate( ([index], cp) )
            index = last[index-1]

        # Iterate if desired, to implement later
        iterCnt += 1
        break


    numCP = len(cp)
    numBlocks = numCP + 1

    rateVec  = np.zeros(numBlocks)
    numVec   = np.zeros(numBlocks)

    cptUse = np.insert(cp, 0, 0)

    #print('cptUse start: {0}, end: {1}, len: {2}'.format(cptUse[0],
                                                    #cptUse[-1], len(cptUse)))
    #print('lenCP: {0}'.format(len(cp)))
    
    # what does this stuff do I don't know...
    print('numBlocks: {0}, dataPts/Block: {1}'.format(numBlocks, len(x)/numBlocks))
    for idBlock in range(numBlocks):
        # set ii1, ii2 as indexes.  Fix edge case at end of blocks
        ii1 = int(cptUse[idBlock])
        if idBlock < (numBlocks - 1):
            ii2 = int(cptUse[idBlock + 1] - 1)
        else:
            ii2 = int(numPts)
                
        subset = y[ii1:ii2]
        weight = cellData[ii1:ii2]
        if ii1 == ii2:
            subset = y[ii1]
            weight = cellData[ii1]
        rateVec[idBlock] = np.dot(weight, subset) / np.sum(weight)

        if np.sum(weight) == 0:
            raise ValueError('error, divide by zero at index: {0}'.format(idBlock))
            print('-------ii1: {0}, ii2: {1}'.format(ii1, ii2))
            

    # Simple hill climbing for merging blocks
    cpUse = np.concatenate(([1], cp, [len(y)]))
    cp = cpUse

    numCP = len(cpUse) - 1
    idLeftVec = np.zeros(numCP)
    idRightVec = np.zeros(numCP)

    for i in range(numCP):
        idLeftVec[i]  = cpUse[i]
        idRightVec[i] = cpUse[i+1]

    # Find maxima defining watersheds, scan for 
    # highest neighbor of each block

    idMax = np.zeros(numBlocks)
    idMax = np.zeros(numBlocks)
    for j in range(numBlocks):
        jL = (j-1)*(j>0) + 0*(j<=0) # prevent out of bounds
        jR = (j+1)*(j<(numBlocks-1)) + (numBlocks-1)*(j>=(numBlocks-1))

        rateL = rateVec[jL]
        rateC = rateVec[j]
        rateR = rateVec[jR]
        rateList = [rateL, rateC, rateR]

        jMax = np.argmax(rateList) #get direction [0, 1, 2]
        idMax[j] = j + jMax - 1 # convert direction to delta

    idMax[ idMax > numBlocks] = numBlocks
    idMax[ idMax < 0] = 0

    # Implement hill climbing (HOP algorithm)
    
    hopIndex = np.array(range(numBlocks)) # init: all blocks point to self
    hopIndex = hopIndex.astype(int)  # cast all as int
    ctr = 0
    # point each block to its max block
    while ctr <= len(x):
        newIndex = idMax[hopIndex]  # Point each to highest neighbor

        if np.array_equal(newIndex, hopIndex):
            break
        else:
            hopIndex = newIndex.astype(int)
            
        ctr += 1
        if ctr == len(x):
            print('Hill climbing did not converge...?')

    idMax = np.unique(hopIndex)
    numMax = len(idMax)

    # Convert to simple list of block boundaries
    boundaries = [0]
    for k in range(numMax):
        currVec = np.where(hopIndex == idMax[k])[0]

        rightDatum = idRightVec[currVec[-1]] - 1 # stupid leftover matlab index
        boundaries.append(rightDatum)
    
    return np.array(boundaries)
