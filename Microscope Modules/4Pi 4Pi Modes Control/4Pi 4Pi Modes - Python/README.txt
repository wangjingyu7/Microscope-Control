1) Double click on run_prompt.bat to open a shell
2) Type `python make_configuration.py --help` and hit enter
3) A list of available options will be printed on the console
4) Choose your wanted options
5) Type `python make_configuration.py ` followed by the options and hit enter
   For example:
   run make_configuration.py --dm0-calibration C17W005p050_20190228_130929-0.990mm.h5 --dm1-calibration C17W123p456_20190228_130929-0.990mm.h5
6) Test the modes by running `python test.py`

*** WARNING ***
This LabVIEW code uses the obsolete CIUsb driver for the Boston Multi-DM. You
must power up the DM in the same order every time. The first DM that is
powered up will be assigned id 0. The other will be assigned id 1. The CIUsb
driver does not support serial number checking.
