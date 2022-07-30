# detects and saves motion

import Cmotion
import datetime

# use defaults (yes/no)
found = False
while not found:
    defaults = raw_input('Use defaults? (Y/N)\n')
    try:
        defaults = defaults.lower()
        if defaults == 'y':
            blurnum = 5
            accum = 0.03
            sens = 7
        found = True
    except AttributeError:
        pass

if defaults == 'n':
    # get blur level
    found = False
    while not found:
        blurnum = raw_input('What blur level do you want?  (Default is 1)\n')
        try:
            blurnum = int(blurnum)
            if blurnum > 0:
                found = True
        except ValueError:
            pass

    # get accumulator
    found = False
    while not found:
        accum = raw_input('How fast should the background reset? (Default is 0.1, lower is slower)\n')
        try:
            accum = float(accum)
            if blurnum > 0:
                found = True
        except ValueError:
            pass

    # get sensitivity
    found = False
    while not found:
        sens = raw_input('How much motion is required to trigger the program?  (Default is 2, lower is more sensitive)\n')
        try:
            sens = int(sens)
            if sens > 0:
                found = True
        except ValueError:
            pass

# check to see if setup mode is needed
found = False
while not found:
    setup = raw_input('Is this running in setup mode? (Y/N)\n')
    try:
        setup = setup.lower()
        if setup == 'y':
            setup = 1
        else:
            setup = 0
        found = True
    except AttributeError:
        pass
     

# get name if not setup mode
if setup == 0:
    found = False
    while not found:
        name = raw_input('Name the video file\n')
        try:
            name = str(name)
            if len(name) > 0:
                found = True
        except ValueError:
            pass
else:
    name = ''
runtime = 10800

print 'Started recording at',datetime.datetime.now().time()
Cmotion.run(blurnum,accum,sens,setup,name,runtime)
print 'Ended recording at',datetime.datetime.now().time()
