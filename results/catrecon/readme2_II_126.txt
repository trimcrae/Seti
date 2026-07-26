II/126   IRAS Serendipitous Survey Catalog     (IPAC 1986)
================================================================================
IRAS Serendipitous Survey Catalog
     Kleinmann S.G., Cutri R.M., Young E.T., Low F.J., Gillett F.C.
    <Joint IRAS Science W.G. (1986)>
    =1986SSC...C......0K
================================================================================
ADC_Keywords: Photometry, infrared ; Infrared sources ; Surveys
Mission_Name: IRAS

Description:
   This is a catalog of 12, 25, 60 and 100 micron photometric
   observations of 43,866 point-like sources detected fortuitiously in
   the Infrared Astronomical Satellite Pointed Observation program.  The
   main objective was to take advantage of the longer-than-nominal
   integration time per source to extend the detection threshold relative
   to that of the Point Source Catalog (PSC); about three-fourths of the
   Serendipitous Survey Catalog (SSC) sources do not appear in the PSC.
   From 1813 Pointed Observation fields, the effective sky coverage is
   1108 square degrees. Relative to the PSC, the SSC is characterized by:
   enhanced sensitivity (by a factor of about 4) in all four wavelength
   bands; excellent reliability in uncrowded fields; uneven sky coverage
   and completeness; reduced positional accuracy; improved photometric
   accuracy; much greater depth in crowded fields at the expense of
   reliability and accuracy.  The SSC data processing, the catalog
   format, and an analysis are given in the Explanatory Supplement to the
   IRAS Serendipitous Survey Catalog.

File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl    Records    Explanations
--------------------------------------------------------------------------------
ReadMe          80          .    This file
ssc.doc         80        106    Information about interpretation of catalog
                                   entries, from the Explanatory Supplement
                                   to the IRAS Serendipitous Survey Catalog.
sources.dat    159      43886    Properties of the point sources identified
assoc.dat       56      29583    Information about positional associations
                                   with sources in other catalogs
meaning.txt     80        235    Details of association catalogues
headers.dat    150       1813    Information about the reference and confirming
                                  grids (observations) in which point sources
                                  were identified
overlap.dat     16       2149    Overlapping fields (Appendix A)
--------------------------------------------------------------------------------

See also:
    II/125 : IRAS catalogue of Point Sources (IRAS/PSC), Version 2.0 (IPAC 1986)
    II/156 : IRAS Faint Source Catalog, |b| > 10, Version 2.0 (Moshir+ 1989)
    VII/73 : IRAS Small Scale Structure Catalog (Helou+ 1985)

Byte-by-byte Description of file: headers.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 13  A13    ---     FNAME     Field Name (2) (1)
  14- 18  I5     ---     RGRID     Reference Grid No. (2), has the lower 60 um
                                     median noise.
  19- 21  I3     d       RDATE     Observation Date (JD 2445000+) of RGRID (2)
  22- 26  I5     ---     CGRID     Confirming Grid No. (2)
  27- 29  I3     d       CDATE     Observation Date (JD 2445000+) of CGRID (2)
      30  A1     ---     MACRO     Macro Type (2) (3)
  31- 33  I3     deg     GLON      Galactic Longitude (2)
  34- 36  I3     deg     GLAT      Galactic Latitude (2)
  37- 40  I4     arcsec  dRA       R.A. Difference between Grid Centers
  41- 44  I4     arcsec  dDE       Dec. Difference between Grid Centers
  45- 48  I4     deg     RGPA      Reference Grid Scan  Direction (E of N)
  49- 52  I4     deg     CGPA      Confirming Grid Scan Direction (E of N)
  53- 55  I3    0.01deg2 EFFAREA   Effective Area of Grid Overlap (2)
                                     covered by both the reference and
                                     confirming grids (in 0.01 square degrees).
  56- 57  I2     ---     RUNDF     No. of additional grid pairs with
                                     Overlap > 5% (2)
  81- 85  I5     mJy     RNOISE12  Median Noise of Ref. Grid (12um)
  86- 90  I5     mJy     RNOISE25  Median Noise of Ref. Grid (25um)
  91- 95  I5     mJy     RNOISE60  Median Noise of Ref. Grid (60um)
  96-100  I5     mJy     RNOISE100 Median Noise of Ref. Grid (100um)
 101-105  I5     mJy     CNOISE12  Median Noise of Conf.Grid (12um)
 106-110  I5     mJy     CNOISE25  Median Noise of Conf.Grid (25um)
 111-115  I5     mJy     CNOISE60  Median Noise of Conf.Grid (60um)
 116-120  I5     mJy     CNOISE100 Median Noise of Conf.Grid (100um)
 121-123  I3     ---     NSOURC12  Number of Confirmed Sources (12um)
 124-126  I3     ---     NSOURC25  Number of Confirmed Sources (25um)
 127-129  I3     ---     NSOURC60  Number of Confirmed Sources (60um)
 130-132  I3     ---     NSOURC100 Number of Confirmed Sources (100um)
 133-135  I3     ---     NCONF12   Number of Confused Confirmations (12um) (2)
 136-138  I3     ---     NCONF25   Number of Confused Confirmations (25um) (2)
 139-141  I3     ---     NCONF60   Number of Confused Confirmations (60um) (2)
 142-144  I3     ---     NCONF100  Number of Confused Confirmations (100um) (2)
 145-147  I3     ---     CIRRUS    Number of 100 um only Confirmed Sources (2)
 148-150  I3     ---     NMERGE    Number of Merged Sources, i.e. number of
                                     source records following the field header.
--------------------------------------------------------------------------------
Note (1):
    Fields are listed in order of increasing Right Ascension of the
    Reference Grid center. The IRAS/SSC field name is the position of
    the center of the reference grid, given in the form hhmmssSddmmss.
Note (2):
    This quantity is listed in the printed version of the catalog.
Note (3):
    Macro code is (Table II.A of "Explanations")
     ------------------------------------------------------
     Code   Name   Scans  Length Cross-step   SNR Gain
                          arcmin   arcmin    improvement
     ------------------------------------------------------
        A  DPS02B    6      96       0.3        4.8
        B  DPS05B    3     360       1.0        3.5
        C  DPS52B    6      96       0          4.8
        D  DPS55B    3     360       0          3.5
        E  DPS60B    4      60       0.8        4.0
        F  DPS60D    5      48       0.4        4.4
        G  DPS61C   12      48       0.2        6.9
        H  DPS61D   15      48       0.2        7.7
        I  DPS62D    9      96       0.4        6.0
        J  DPS63D    3      96       0.8        3.5
        K  DPS60C    5      48       0.4        4.4
        L  TPS52B    6      96       0          4.8
        M  DPS60M    5      48       0.4        4.4
     ------------------------------------------------------

Byte-by-byte Description of file: sources.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 11  A11   ---     IRAS      IRAS/SSC Source Name (1)
  12- 13  I2    h       RAh       Right Ascension 1950 (hours)
  14- 15  I2    min     RAm       Right Ascension 1950 (minutes)
  16- 18  I3    0.1s    RAds      Right Ascension 1950 (seconds)
      19  A1    ---     DE-       Declination 1950 (Sign)
  20- 21  I2    deg     DEd       Declination 1950 (degrees)
  22- 23  I2    arcmin  DEm       Declination 1950 (minutes)
  24- 25  I2    arcsec  DEs       Declination 1950 (seconds)
  27- 29  I3    ---     ANGLE     Position Angle of SSC Source Error Box
                                    expressed in degrees East of North.
  31- 39  E9.3  Jy      FLUX12    Averaged Non-color Corrected Flux Densities(2)
  40- 48  E9.3  Jy      FLUX25    Averaged Non-color Corrected Flux Densities(2)
  49- 57  E9.3  Jy      FLUX60    Averaged Non-color Corrected Flux Densities(2)
  58- 66  E9.3  Jy      FLUX100   Averaged Non-color Corrected Flux Densities(2)
      67  I1    ---     FQUAL12   Flux Density Quality (2) (5)
      68  I1    ---     FQUAL25   Flux Density Quality (2) (5)
      69  I1    ---     FQUAL60   Flux Density Quality (2) (5)
      70  I1    ---     FQUAL100  Flux Density Quality (2) (5)
  71- 75  I5    ---     RGRID     Reference Grid Number
  81- 83  I3    %       RELUNC12  ? Percent Relative FLUX12 Uncertainty (2)
  84- 86  I3    %       RELUNC25  ? Percent Relative FLUX12 Uncertainty (2)
  87- 89  I3    %       RELUNC60  ? Percent Relative FLUX12 Uncertainty (2)
  90- 92  I3    %       RELUNC100 ? Percent Relative FLUX12 Uncertainty (2)
  93- 96  I4    ---     TLSNR12   ? 10x Local Signal-to-Noise Ratio
  97-100  I4    ---     TLSNR25   ? 10x Local Signal-to-Noise Ratio
 101-104  I4    ---     TLSNR60   ? 10x Local Signal-to-Noise Ratio
 105-108  I4    ---     TLSNR100  ? 10x Local Signal-to-Noise Ratio
     109  A1    ---     CC12      Point Source Correlation Coefficient (2) (3)
     110  A1    ---     CC25      Point Source Correlation Coefficient (2) (3)
     111  A1    ---     CC60      Point Source Correlation Coefficient (2) (3)
     112  A1    ---     CC100     Point Source Correlation Coefficient (2) (3)
 113-114  I2    ---     TRFLUX12  ? 10x Fc/Fr (confirmed/reference) (6)
 115-116  I2    ---     TRFLUX25  ? 10x Fc/Fr (confirmed/reference) (6)
 117-118  I2    ---     TRFLUX60  ? 10x Fc/Fr (confirmed/reference) (6)
 119-120  I2    ---     TRFLUX100 ? 10x Fc/Fr (confirmed/reference) (6)
 121-124  I4    arcsec  dRA12     ? Right Ascension Delta (12um)
 125-128  I4    arcsec  dDE12     ? Declination Delta (12um)
 129-132  I4    arcsec  dRA25     ? Right Ascension Delta (25um)
 133-136  I4    arcsec  dDE25     ? Declination Delta (25um)
 137-140  I4    arcsec  dRA60     ? Right Ascension Delta (60um)
 141-144  I4    arcsec  dDE60     ? Declination Delta (60um)
 145-148  I4    arcsec  dRA100    ? Right Ascension Delta (100um)
 149-152  I4    arcsec  dDE100    ? Declination Delta (100um)
     153  I1    ---     PNEARC12  ? Number of Sources in Confusion Window (2)(4)
     154  I1    ---     PNEARC25  ? Number of Sources in Confusion Window (2)(4)
     155  I1    ---     PNEARC60  ? Number of Sources in Confusion Window (2)(4)
     156  I1    ---     PNEARC100 ? Number of Sources in Confusion Window (2)(4)
 157-158  I2    ---     NID       Number of Positional Associations (2)
     159  I1    ---     IDTYPE    Type of Object (2)
--------------------------------------------------------------------------------
Note (1):
    Sources are listed in order of increasing Right Ascension within each
    field.
Note (2):
    This quantity is listed in the printed version of the SSC.
Note (3): the Point Source Correlation Ceofficients are between 70-100%.
    These are encoded as alphabetic characters with A=100, B=99..Z=75-70
    (1 value per band).
    The quoted correlation coefficients come from the reference or
    confirming grids, whichever is higher, for high quality sources.
Note (4):
    In regions of high source density, the Pointed Observation
    source extraction process, as well as the Serendipitous
    Survey Confirmation and Band Merging processing, can result
    in degraded positions and incorrectly band merged sources.
    PNEARC is 1-(number of confirmed sources in the confusion
    and band merge window).
    Any value greater than zero is indicative of potential confusion
    in the processing and the resulting source information should be
    examined carefully, e.g. by inspection of the grids in question.
Note (5): Qualities:
    3=high-quality, 2=moderate quality,
    1=upper limit
Note (6): SSC sources can have flux density ratios
          0.5 < Fc/Fr < 2.0.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: assoc.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1-  2  I2     ---     CatNo    Catalog Number (1)
   3- 17  A15    ---     Source   Source ID
  18- 22  A5     ---     Type     Source Type/Spectral Class (2)
  23- 25  I3     arcsec  Radius   Radius Vector from SSC Position to Association
  26- 28  I3     deg     PosAngle Position Angle from SSC Position to
                                    Association (E of N)
  29- 32  I4     ---     Field1   Object Field #1 Dependent (3)
  33- 36  I4     ---     Field2   Object Field #2 Dependent (4)
  37- 40  I4     ---     Field3   Object Field #3 Dependent (5)
  41- 51  A11    ---     IRAS     ! SSC Name association (6)
  52- 56  I5     ---     RGRID    SSC Name association (6)
--------------------------------------------------------------------------------
Note (1):
    For associations with the IRAS/PSC <II/125>, this value is 41.
    For other associations, see details in file "meaning.txt".
Note (2):
    For associations with the IRAS/PSC <II/125> (i.e. CATNO=41),
    this field is left blank. For other associations, see details
    in "meaning.txt".
Note (3):
    For associations with the IRAS/PSC <II/125> (i.e. CATNO=41), this
    value is a flag indicating the bands in which the source was detected
    with medium or high quality; it is encoded as indicated in the
    PSC Supplement Table X.B.2 (or'ed number values of 1 (12um),
    2 (25um), 4(60um) and 8(100um)). For other associations, see details
    in "meaning.txt".
Note (4):
    For associations with the IRAS/PSC <II/125> (i.e. CATNO=41), this
    value is the PSC 2.0 Flux Density in the shortest (first) wavelength
    band in which it was detected. Flux Densities higher than 10 Jy are
    encoded 9999. For other associations, see details in "meaning.txt".
Note (5):
    For associations with the IRAS/PSC <II/125> (i.e. CATNO=41), this
    value is the PSC 2.0 Flux Density in the second wavelength band
    in which it was detected. Flux Densities higher than 10 Jy
    are encoded 9999. For other associations, see details in "meaning.txt".
Note (6):
    These fields are a repetition of bytes 1-11 and 71-75 of "sources" table.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: overlap.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1-  5  I5     ---     RGRID    Reference Grid No
   7- 11  I5     ---     GRID1    Overlapping grid no., >5% overlap
                                      with Reference Grid
  13- 16  I4     arcmin2 OVLP1    Overlapping area
--------------------------------------------------------------------------------

Reference(s):
   Kleinmann S.G., Cutri R.M., Young E.T., Low F.J., and
      Gillett F.C. 1986, Explanatory Supplement to the IRAS
      Serendipitous Survey Catalog (Pasadena: JPL)
   IRAS Catalogs and Atlases Explanatory Supplement, 1988, ed. Beichman C.,
      Neugebauer G., Habing H.J., Clegg P.E., and Chester T.J.
      (Washington, DC: GPO), NASA RP-1190, vol 1
   Young E.T., Neugebauer G., Kopan E.L., Benson R.D., Conrow T.P.,
      Rice W.L., and Gregorich D.T. 1985, A User's Guide to IRAS Pointed
      Observation Products, IPAC Preprint PRE-008N
================================================================================
(End)             Francois Ochsenbein [CDS], Seth Digel [SSDOO/ADC]  14-Aug-1997
