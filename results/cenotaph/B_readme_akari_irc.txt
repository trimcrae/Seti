II/297           AKARI/IRC mid-IR all-sky Survey    (ISAS/JAXA, 2010)
================================================================================
The AKARI/IRC Mid-Infrared All-Sky Survey (Version 1)
    Ishihara D., Onaka T., Kataza H., Salama A., Alfageme C., Cassatella A.,
    Cox N., Garcia-Lario P., Stephenson C., Cohen M., Fujishiro N., 
    Fujiwara H., Hasegawa S., Ita Y., Kim W., Matsuhara H., Murakami H.,
    Muller T.G., Nakagawa T., Ohyama Y., Oyabu S., Pyo J., Sakon I.,
    Shibai H., Takita S., Tanab T., Uemizu K., Ueno M., Usui F., Wada T.,
    Watarai H., Yamamura I., Yamauchi C. 
   <Astron. Astrophys. 514, A1 (2010)>
   =2010A&A...514A...1I
================================================================================
ADC_Keywords: Infrared sources ; Surveys ; Photometry, infrared
Mission_Name: AKARI
Keywords: infrared: general - techniques: image processing - surveys

Description:
    The AKARI Infrared Astronomical Satellite observed the whole sky in
    the far infrared (50-180{mu}m) and the mid-infrared (9 and 18{mu}m)
    between May 2006 and August 2007 (Murakami et al. 2007PASJ...59S.369M)

    The AKARI/IRC Point Source Catalogue Version 1.0 provides positions
    and fluxes for 870,973 sources observed with the InfraRed Camera
    (IRC): 844,649 sources in the S9W filter, and 194,551 sources in the
    L18W filter; the "Note (1)" section below provides a summary of the
    IRC filter characteristics.

Recommendation:
    The users of the catalogue are requested to read the documents
    carefully before critical discussions of the data. Any questions and
    comments are appreciated at ISAS Helpdesk (iris_help@ir.isas.jaxa.jp)

    Please acknowledge the usage of the AKARI data (details at
    http://www.ir.isas.jaxa.jp/AKARI/Publications/guideline.html).

File Summary:
--------------------------------------------------------------------------------
 FileName  Lrecl  Records    Explanations
--------------------------------------------------------------------------------
ReadMe        80        .    This file
irc.dat      196   870973    AKARI/IRC All-Sky Survey Point Source Catalogue
                                  (Version 1.0)
--------------------------------------------------------------------------------

See also:
    II/298 : AKARI/FIS Bright Source Point Source Catalogue (ISAS/JAXA, 2010)
    http://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/ : AKARI home page
    http://darts.isas.jaxa.jp/astro/akari/cas.html : AKARI Archive Server

Byte-by-byte Description of file: irc.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label    Explanations
--------------------------------------------------------------------------------
   2- 10  I9    ---     objID    [200000001/200870973] AKARI source ID number.
  12- 25  A14   ---     objName  AKARI source name (HHMMSSs+DDMMSS) (2)
  27- 35  F9.5  deg     RAdeg    Right Ascension (J2000)
  37- 45  F9.5  deg     DEdeg    Declination (J2000)
  49- 53  F5.2  arcsec  errMaj   Major axis of position error ellipse
  57- 61  F5.2  arcsec  errMin   Minor axis of position error ellipse
  64- 69  F6.2  deg     errPA    Position angle of Major axis
  70- 80  E11.4 Jy      S09      ?=-999.9 Flux density in AKARI/S9W filter (1)
  81- 91  E11.4 Jy      S18      ?=-999.9 Flux density in AKARI/L18W filter (1)
  92-101  E10.3 Jy    e_S09      ?=-1000 Flux error in S9W
 102-111  E10.3 Jy    e_S18      ?=-1000 Flux error in L18W
     113  I1    ---   q_S09      [0,3] Flux quality flag for S9W (3)
     115  I1    ---   q_S18      [0,3] Flux quality flag for L18W (3)
 117-119  A3    ---     ---      [0]
     120  A1    ---     f09      [0-9a-f] Bit flags for S9W (4)
 122-124  A3    ---     ---      [0]
     125  A1    ---     f18      [0-9a-f] Bit flags for L18W (4)
 127-130  I4    ---     Ns09     Number of scans with source detection in S9W
 132-135  I4    ---     Ns18     Number of scans with source detection in L18W
 137-140  I4    ---     Np09     Number of scans with possible detection in S09W
 142-145  I4    ---     Np18     Number of scans with possible detection in L18W
 147-148  I2    ---     M09      [0,1]?=-1 1 is month confirmed and 0 is not.
                                    (inverted value of lower bit of f09)
 150-151  I2    ---     M18      [0,1]?=-1 1 is month confirmed and 0 is not.
                                    (inverted value of lower bit of f18)
 153-155  I3    ---     Nd09     ?=-1 Number of sources in 45" radius in S9W
 157-159  I3    ---     Nd18     ?=-1 Number of sources in 45" radius in L18W
 161-162  I2    ---     X09      [0,1]?=-1 Extended source flag (5)
 164-165  I2    ---     X18      [0,1]?=-1 Extended source flag (5)
 167-173  F7.2  arcsec  r09      ?=-999.9 Radius of source extent in S9W (6)
 175-181  F7.2  arcsec  r18      ?=-999.9 Radius of source extent in L18W (6)
 183-186  I4    ---     Ndet     Number of events used for position computation
 188-191  I4    ---     N09      Number of events used for S09 measurement
 193-196  I4    ---     N18      Number of events used for S18 measurement
--------------------------------------------------------------------------------
Note (1): the filter characteristics are:
     -------------------------------------------------
           Filter:    S9W      L18W
     -------------------------------------------------
       Center(um):     9         18
        Width(um):    4.10       9.97
        Range(um):  6.7-11.6  13.9-25.6
         Pixel("):  9.4x9.4   10.4x9.4
       Limit(mJy):     50       120     [at 5{sigma}]
     -------------------------------------------------

Note (2):
    The sources should be referred in the literatures by their full name
    (AKARI-IRC-V1) followed by the letter 'J' and the objName, e.g.
    AKARI-IRC-V1 J0123498-025805

Note (3): Four-level flux quality indicator:
    3 = high quality (source confirmed and flux is reliable)
    2 = source is confirmed but the flux is not reliable (see the flags)
    1 = the source is not confirmed
    0 = not observed

Note (4): Bit flags of data quality in hexadecimal:
      1 = not month confirmed.
         This means that the period between the first detection and
                the last detection is shorter than a month.
      2 = saturated (not used this version);
      4 = use events affected by the South Atlantic Anomaly (SAA)
          (not used in this version),
      8 = use edge events
      Combined values are represented by the sum, e.g. 9 = 8 + 1 =
      not month confirmed AND used edge events.

Note (5): the value 1 means that the source is possibly more extended than
     the point spread function (>15.6arcsec)

Note (6): average of major and minor axes of source extent = (a+b)/2
--------------------------------------------------------------------------------

History:
  * 21-Apr-2010: from the AKARI pages
    http://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/
================================================================================
(End)                                   Francois Ochsenbein [CDS]    21-Apr-2010
