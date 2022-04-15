'''

Code used to convert between various coordinate formats.
Has the following functions:

  Decimal2RA(deg):
    Converts Decimal Degrees Right Ascension to standard HH:MM:SS.ss format
    Takes float decimal degrees RA as input.
    Returns RA as an array [['HH'],['MM'],['SS.ss']]
    Use ":".join(Decimal2RA(deg)) to obtain 'HH:MM:SS.ss'

  Decimal2Dec(deg):
    Converts Decimal Degrees Declination to standard DD:MM:SS.ss format
    Takes float decimal degrees declination as input.
    Returns Declination as an array [['DD'],['MM'],['SS.ss']]
    Use ":".join(Decimal2Dec(deg)) to obtain 'DD:MM:SS.ss'

  RA2Decimal(deg):
    Converts Right Ascension from standard HH:MM:SS.ss to decimal degrees.
    Takes string 'HH:MM:SS.ss' or 'HH MM SS.ss' as input.
    Returns decimal degree RA as a float.

  Dec2Decimal(deg):
    Converts Declination from standard DD:MM:SS.ss to decimal degrees.
    Takes string 'DD:MM:SS.ss' or 'DD MM SS.ss' as input.
    Returns decimal degree Dec as a float.


'''
def Decimal2RA(deg):

    # Accepts decimal degrees Right Ascension and returns HH:MM:SS.SSSS.
    # Values are returned as strings.
    # Rounds seconds to N decimal places, defined by 'precision' variable.

    precision = 4

    hours = abs( deg/15. )
    minutes = (hours - int(hours))*60
    seconds = (minutes - int(minutes))*60

    if hours < 10.0:
        hours = "0" + str(int(hours))
    else:
        hours = str(int(hours))

    if minutes < 10.0:
        minutes =  "0" + str(int(minutes))
    else:
        minutes = str(int(minutes))
    if seconds < 10.0:
        seconds = "0" + str(round(seconds,precision))
    else:
        seconds = str(round(seconds,precision))
    while len(seconds) < (3+precision):
        seconds = str(seconds) + '0'

    return(hours,minutes,seconds)

def Decimal2Dec(deg):

    # Accepts decimal degrees Declination and returns DD:MM:SS.ssss
    # Values are returned as strings, preventing -00 and +00 from
    #   becoming 0.
    # Rounds arcseconds to N decimal places, defined by 'precision'
    #   variable.

    precision = 4

    if deg >= 0.:
        south = False
    else:
        south = True


    degrees = abs( deg )
    arcminutes = (degrees - int(degrees))*60
    arcseconds = (arcminutes - int(arcminutes))*60

    if south == True:
        if int(degrees) == 0:
            degrees = '-00'
        elif degrees < 10.0:
            degrees = "-0" + str(int(degrees))
        else:
            degrees = "-" + str(int(degrees))
    else:
        if int(degrees) == 0:
            degrees = '+00'
        elif degrees < 10.0:
            degrees = '+0' + str(int(degrees))
        else:
            degrees = '+' + str(int(degrees))

    if arcminutes < 10.0:
        arcminutes = '0' + str(int(arcminutes))
    else:
        arcminutes = str(int(arcminutes))

    if arcseconds < 10.0:
        arcseconds = '0' + str(round(arcseconds,precision))
    else:
        arcseconds = str(round(arcseconds,precision))
    while len(arcseconds) < (3+precision):
        arcseconds = str(arcseconds) + '0'


    return(degrees,arcminutes,arcseconds)

def RA2Decimal(RA):

    precision = 4

    if RA.find(":") and RA.count(":") == 2:
        hours   = float(RA.split(":")[0])
        minutes = float(RA.split(":")[1])
        seconds = float(RA.split(":")[2])
    else:
        hours   = float(RA.split()[0])
        minutes = float(RA.split()[1])
        seconds = float(RA.split()[2])

    decimal = hours*15. + minutes*15./60. + seconds*15./3600.

    return(str(round(decimal,precision)))

def Dec2Decimal(Dec):

    # Accepts Declination in DD:MM:SS.ssss or DD MM SS.ssss format.
    # Splits on ":" if two ":" characters found.
    # Splits on blank space otherwise.
    # Returns decimal Declination to N decimal, defined by precision
    #   variable.

    precision = 4

    if Dec.find(":") and Dec.count(":") == 2:
        south = True if "-" in Dec.split(":")[0] else False
        degrees    = float(Dec.split(":")[0])
        arcminutes = float(Dec.split(":")[1])
        arcseconds = float(Dec.split(":")[2])
    else:
        south = True if "-" in Dec.split()[0] else False
        degrees    = float(Dec.split()[0])
        arcminutes = float(Dec.split()[1])
        arcseconds = float(Dec.split()[2])

    if degrees < 0.0 or south:
        N = -1.
    else:
        N = +1.

    decimal = N*abs(degrees) + N*abs(arcminutes/60.) + N*abs(arcseconds/3600.)

    return(str(round(decimal,precision)))

#def Decimal2Galactic(RA,Dec):
    # Stuff here

#def Galactic2Decimal(Glat,Glon):
    # Stuff here
