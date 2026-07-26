II/298   AKARI/FIS All-Sky Survey Point Source Catalogues  (ISAS/JAXA, 2010)
================================================================================
AKARI/FIS All-Sky Survey Bright Source Catalogue Version 1.0
    Yamamura I., Makiuti S., Ikeda N., Fukuda Y., Oyabu S., Koga T., White G.J.
    <ISAS/JAXA (2010)>
    =2010yCat.2298....0Y
================================================================================
ADC_Keywords: Infrared sources ; Surveys ; Photometry, infrared
Mission_Name: AKARI

Description:
    The AKARI Infrared Astronomical Satellite observed the whole sky in
    the far infrared (50-180{mu}m) and the mid-infrared (9 and 18{mu}m)
    between May 2006 and August 2007 (Murakami et al. 2007PASJ...59S.369M)

    The AKARI/FIS All-Sky Survey Bright Source Catalog Version 1.0
    provides positions and fluxes for 427071 point sources in the 4
    far-infrared wavelengths centered at 65, 90, 140 and 160{mu}m (see
    filter characteristics in the "Note (1)" section below)

Recommendation:
    The users of the catalogue are requested to read the documents
    carefully before critical discussions of the data. Any questions and
    comments are appreciated at ISAS Helpdesk (iris_help@ir.isas.jaxa.jp)

Acknowledging AKARI data in publications:
    Please acknowledge the usage of the AKARI data (details at
    http://www.ir.isas.jaxa.jp/AKARI/Publications/guideline.html).

File Summary:
--------------------------------------------------------------------------------
 FileName  Lrecl  Records    Explanations
--------------------------------------------------------------------------------
ReadMe        80        .    This file
fis.dat      231   427071    AKARI/FIS All-Sky Survey Bright Source Catalogue
                                  (Version 1.0)
--------------------------------------------------------------------------------

See also:
    II/297 : AKARI/IRC All-Sky Survey Point Source Catalogue (ISAS/JAXA, 2010)
    http://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/ : AKARI home page
    http://darts.isas.jaxa.jp/astro/akari/cas.html : AKARI Archive Server

Byte-by-byte Description of file: fis.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label    Explanations
--------------------------------------------------------------------------------
       1  A1    ---     ---      [0]
   2- 10  I9    ---     objID    [3000001/3427071] Object ID
  12- 25  A14   ---     objName  AKARI source name (HHMMSSs+DDMMSS) (2)
  27- 35  F9.5  deg     RAdeg    Right Ascension (J2000)
  37- 45  F9.5  deg     DEdeg    Declination (J2000)
  47- 51  F5.2  arcsec  errMaj   [6.00] Major axis of position error ellipse
  53- 57  F5.2  arcsec  errMin   [6.00] Minor axis of position error ellipse
  59- 63  F5.1  deg     errPA    [0.0] position angle of error ellipse
  65- 74  E10.4 Jy      S65      ?=-999.9 Flux density in N60 (1)
  76- 85  E10.4 Jy      S90      ?=-999.9 Flux density in WIDE-S (1)
  87- 96  E10.4 Jy      S140     ?=-999.9 Flux density in WIDE-L (1)
  98-107  E10.4 Jy      S160     ?=-999.9 Flux density in N160 (1)
 109-117  E9.3  Jy    e_S65      ?=-99.9 uncertainty in N60
 119-127  E9.3  Jy    e_S90      ?=-99.9 uncertainty in WIDE-S
 129-137  E9.3  Jy    e_S140     ?=-99.9 uncertainty in WIDE-L
 139-147  E9.3  Jy    e_S160     ?=-99.9 uncertainty in N160
     149  I1    ---   q_S65      [0,3] quality flag for N60 (3)
     151  I1    ---   q_S90      [0,3] quality flag for WIDE-S (3)
     153  I1    ---   q_S140     [0,3] quality flag for WIDE-L (3)
     155  I1    ---   q_S160     [0,3] quality flag for N160 (3)
 157-159  A3    ---     ---      [0]
     160  A1    ---   f_S65      [0-9A-F] Bit flags for N60 (4)
 162-164  A3    ---     ---      [0]
     165  A1    ---   f_S90      [0-9A-F] Bit flags for WIDE-S (4)
 167-169  A3    ---     ---      [0]
     170  A1    ---   f_S140     [0-9A-F] Bit flags for WIDE-L (4)
 172-174  A3    ---     ---      [0]
     175  A1    ---   f_S160     [0-9A-F] Bit flags for N160 (4)
 177-180  I4    ---     Ns65     Number of scans with source detection in N60
 182-185  I4    ---     Ns90     Number of scans with source detection in WIDE-S
 187-190  I4    ---     Ns140    Number of scans with source detection in WIDE-L
 192-195  I4    ---     Ns160    Number of scans with source detection in N160
 197-200  I4    ---     Np65     Number of possible detections in N60
 202-205  I4    ---     Np90     Number of possible detections in WIDE-S
 207-210  I4    ---     Np140    Number of possible detections in WIDE-L
 212-215  I4    ---     Np160    Number of possible detections in N160
 217-218  I2    ---     M65      [0,1]?=-1 month confirmation flag in N60 (5)
 220-221  I2    ---     M90      [0,1]?=-1 month confirmation flag in WIDE-S (5)
 223-224  I2    ---     M140     [0,1]?=-1 month confirmation flag in WIDE-L (5)
 226-227  I2    ---     M160     [0,1]?=-1 month confirmation flag in N160 (5)
 229-231  I3    ---     Ndens    Number of sources within 5arcmin of source
--------------------------------------------------------------------------------

Note (1): the filter characteristics are:
     -----------------------------------------------------
         Filter:     N60    WIDE-S    WIDE-L     N160
     -----------------------------------------------------
     Center(um):     65       90       140       160
      Range(um):    50-80    60-110   110-180   140-180
       Pixel("):    26.8      26.8     44.2      44.2
     -----------------------------------------------------

Note (2):
    The sources should be referred in the literatures by their full name
    (AKARI-FIS-V1) followed by the letter 'J' and the objName, e.g.
    AKARI-FIS-V1 J0123498-025805

Note (3): Four-level flux quality indicator:
    3 = high quality (source confirmed and flux is reliable)
    2 = source is confirmed but the flux is not reliable (see the flags)
    1 = the source is not confirmed
    0 = not observed

Note (4): Bit flags of data quality in hexadecimal:
      1 = CDS mode used (Correlated Double Sampling used to observe
          bright sky regions to avoid saturation, e.g. in the inner
          Galactic plane)
      2 = flux too low
      4 = (not used)
      8 = possibly a 'side-lobe' detection
      Combined values are represented by the sum, e.g. 9 = 8 + 1 =
      a possible side-lobe detection in CDS mode.

Note (5): The value is 1 when the source is observed in scans separated by
     more than one month. This value is independent from hour confirmation
     and can be 1 even if the source is not confirmed (q_=1); similarly a
     value of 0 does not mean that the source is unreliable.
--------------------------------------------------------------------------------

History:
  * 21-Apr-2010: from the AKARI pages
    http://www.ir.isas.jaxa.jp/AKARI/Observation/PSC/Public/
================================================================================
(End)                                   Francois Ochsenbein [CDS]    30-Apr-2010
