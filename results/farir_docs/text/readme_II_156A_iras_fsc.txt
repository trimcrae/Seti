II/156A     IRAS Faint Source Catalog, |b| > 10, Version 2.0 (Moshir+ 1989)
================================================================================
IRAS Faint Source Catalog, |b| > 10 Degrees, Version 2.0
    Moshir, M., Copan, G., Conrow, T., McCallon, H., Hacking, P., Gregorich, D.,
    Rohrbach, G., Melnyk, M., Rice, W., Fullmer, L., and Chester, T.J.
   <Infrared Processing and Analysis Center (1989)>
   =1990IRASF.C......0M
================================================================================
ADC_Keywords: Infrared sources ; Photometry, infrared ; Surveys
Mission_Name: IRAS

Description:
    The Faint Source Survey (FSS) is the definitive Infrared Astronomical
    Satellite data set for faint point sources. The FSS was produced by
    point-source filtering the individual detector data streams and then
    coadding those data streams using a trimmed-average algorithm. The
    resulting images, or plates, give the best estimate from the IRAS
    survey data of the point source flux density at every surveyed point
    of the sky. The Faint Source Catalog (FSC) is a compilation of the
    sources extracted from the FSS plates that have met reasonable
    reliability requirements. Averaged over the whole catalog, the FSC is
    at least 98.5% reliable at 12 and 25 microns, and ~94% at 60 microns.
    For comparison, the IRAS Point Source Catalog (PSC) is >99.997%
    reliable, but the sensitivity of the FSC exceeds that of the PSC by
    about a factor of 2.5. The FSC contains data for 173,044 point sources
    in unconfused regions with flux densities typically above 0.2 Jy at
    12, 25, and 60 microns, and above 1.0 Jy at 100 microns. The FSS
    plates are somewhat more sensitive but less reliable than the FSC;
    typically, only sources with SNR>5-6 in the plates are contained in
    the FSC. Sources with SNR>3 but which do not meet the reliability
    requirements of the FSC are catalogued in the Faint Source Reject File
    (FSR, Cat. II/275). The data products, the processing methods used to
    produce them, results of an analysis of these products, and cautionary
    notes are given in the Explanatory Supplement to the IRAS Faint Source
    Survey (see references in fsc.txt).

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl   Records    Explanations
--------------------------------------------------------------------------------
ReadMe         80         .    This file
fsc.txt        84       897    Original description (ADC CD-ROM Vol. 1, 1991)
SIMPLE.fih     80        60    FITS header
main.dat      227    173044    IRAS Faint Sources
main.fih       80       306         associated FITS header
assoc.dat      64    235935    Associations
assoc.fih      79        94         associated FITS header
meaning.txt    80       304    Details of association catalogues
tofits.sh      80        74    Unix shell to generate FITS (1)
--------------------------------------------------------------------------------
Note (1): This small script should run on any Unix platform.
    To generate the FITS version of the tables, execute
    ./tofits main > main.fits       (for the main table)
    ./tofits assoc  > assoc.fits    (for the association file)
--------------------------------------------------------------------------------

See also:
  II/125  : IRAS Point Source Catalog (IRAS-PSC)
  II/126  : IRAS Serendipitous Survey Catalog (IRAS-SSC, sources SHHMMm+DDMMA)
  II/274  : IRAS Point Source Reject Catalog (IRAS-PSR, sources RHHMMm+DDMM)
  II/275  : IRAS Faint Source Reject Catalog (sources IRAS ZHHMMm+DDMMA)
  VII/73  : IRAS Small Scale Structure Catalog (IRAS-SSSC, sources XHHMM+DDd)
  II/174  : IRAS 2Jy Redshift Survey Data File
  II/190  : IRAS Minor Planet Survey Data Base
  VII/91  : IRAS Asteroid and Comet Survey
  VII/109 : IRAS Observations of Large Optical Galaxies
  VII/113 : Catalogued Galaxies + QSOs observed in IRAS Survey

Byte-per-byte Description of file: main.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1- 12  A12    ---     IRAS     IRAS faint source name (starts by F)
  13- 14  I2     h       RAh      Hours RA, equinox 1950.0, epoch 1983.5
  15- 16  I2     min     RAm      Minutes RA, equinox 1950.0, epoch 1983.5
  17- 19  I3     0.1s    RAds     Deci-seconds RA
      20  A1     ---     DE-      Sign of DEC, equinox 1950.0, epoch 1983.5
  21- 22  I2     deg     DEd      Degrees Dec, equinox 1950.0, epoch 1983.5
  23- 24  I2     arcmin  DEm      Minutes Dec, equinox 1950.0, epoch 1983.5
  25- 26  I2     arcsec  DEs      Seconds Dec, equinox 1950.0, epoch 1983.5
  27- 29  I3     arcsec  Major    Uncertainty ellipse major axis
  30- 32  I3     arcsec  Minor    Uncertainty ellipse minor axis
  33- 35  I3     deg     PosAng   Uncertainty ellipse position angle
  36- 38  I3     ---   o_Fnu12   ? Number of times observed at 12um
  39- 41  I3     ---   o_Fnu25   ? Number of times observed at 25um
  42- 44  I3     ---   o_Fnu60   ? Number of times observed at 60um
  45- 47  I3     ---   o_Fnu100  ? Number of times observed at 100um
  48- 56  E9.3   Jy      Fnu12    Non-color corrected flux density at 12um
  57- 65  E9.3   Jy      Fnu25    Non-color corrected flux density at 25um
  66- 74  E9.3   Jy      Fnu60    Non-color corrected flux density at 60um
  75- 83  E9.3   Jy      Fnu100   Non-color corrected flux density at 100um
      84  I1     ---   q_Fnu12    Flux density quality at 12um
      85  I1     ---   q_Fnu25    Flux density quality at 25um
      86  I1     ---   q_Fnu60    Flux density quality at 60um
      87  I1     ---   q_Fnu100   Flux density quality at 100um
  88- 90  I3      %    e_Fnu12    Relative flux density uncertainty at 12um
  91- 93  I3      %    e_Fnu25    Relative flux density uncertainty at 25um
  94- 96  I3      %    e_Fnu60    Relative flux density uncertainty at 60um
  97- 99  I3      %    e_Fnu100   Relative flux density uncertainty at 100um
 100-101  I2      %      Rel      Percent minimum source reliability
 102-108  E7.1   ---     SNR12    ? Signal/Noise ratio at 12um
 109-115  E7.1   ---     SNR25    ? Signal/Noise ratio at 25um
 116-122  E7.1   ---     SNR60    ? Signal/Noise ratio at 60um
 123-129  E7.1   ---     SNR100   ? Signal/Noise ratio at 100um
 130-136  E7.1   ---  locSNR12    ? Local Signal/Noise ratio at 12um
 137-143  E7.1   ---  locSNR25    ? Local Signal/Noise ratio at 25um
 144-150  E7.1   ---  locSNR60    ? Local Signal/Noise ratio at 60um
 151-157  E7.1   ---  locSNR100   ? Local Signal/Noise ratio at 100um
 158-160  I3     pix     A12      ? Number of pixels above threshold at 12um
 161-163  I3     pix     A25      ? Number of pixels above threshold at 25um
 164-166  I3     pix     A60      ? Number of pixels above threshold at 60um
 167-169  I3     pix     A100     ? Number of pixels above threshold at 100um
 170-171  I2     ---     Ncat     Number of nearby catalog sources (6')
 172-173  I2     ---     Nx12     Number of nearby extractions at 12 um
 174-175  I2     ---     Nx25     Number of nearby extractions at 25 um
 176-177  I2     ---     Nx60     Number of nearby extractions at 60 um
 178-179  I2     ---     Nx100    Number of nearby extractions at 100 um
 180-181  I2     ---     Cir1     Number of nearby 100 um-only extractions
 182-183  I2     ---     Conf     Confusion flag, 1 per band, bit-encoded (1)
 184-188  F5.2   ---     NoisC12  ? Noise correction factor at 12um
 189-193  F5.2   ---     NoisC25  ? Noise correction factor at 25um
 194-198  F5.2   ---     NoisC60  ? Noise correction factor at 60um
 199-203  F5.2   ---     NoisC100 ? Noise correction factor at 100um
 204-205  I2     ---     nID      ? Number of positional associations
 206-207  I2     ---     Type     ? Type of associated object (G1)
 208-212  F5.3   ---     NoisR12  ? Ratio of 85% to 68% quantiles of flux
                                    distribution at 12um
 213-217  F5.3   ---     NoisR25  ? Ratio of 85% to 68% quantiles of flux
                                    distribution at 25um
 218-222  F5.3   ---     NoisR60  ? Ratio of 85% to 68% quantiles of flux
                                    distribution at 60um
 223-227  F5.3   ---     NoisR100 ? Ratio of 85% to 68% quantiles of flux
                                    distribution at 100um
--------------------------------------------------------------------------------
Note (1): corresponding to 12, 25, 60 and 100um from lowest to highest bit.
     Therefore the confusion flags are set in the bands as follows:
      1 = confusion in 12um band
      2 = confusion in 25um band
      4 = confusion in 60um band
      8 = confusion in 100um band
     Confusion in multiple bands are expressed by a sum of the values,
     e.g. 12 (8+4) means a confusion in 60 and 100{mu}m bands.
--------------------------------------------------------------------------------

Byte-per-byte Description of file: assoc.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1- 12  A12    ---     IRAS     IRAS source name
  13- 18  I6     ---     RecNo    Main data table record number for IRAS source
  19- 20  I2     ---     catID    Catalog number, details in file "meaning.txt"
  21- 35  A15    ---     Source   Source identification
  36- 40  A5     ---     Type     Source type or spectral class, or association
                                  catalog type (G1)
  41- 43  I3     arcsec  Dist     Distance of IRAS source to association
  44- 46  I3     deg     dPA      Position angle from IRAS source to association
  47- 49  I3     arcsec  dMaj     Distance from IRAS source to association
                                  along major axis of IRAS uncertainty ellipse
  50- 52  I3     arcsec  dMin     Distance from IRAS source to association
                                  along minor axis of IRAS uncertainty ellipse
  53- 56  I4     ---     Field1   ?=-999 Object data field 1, catalog dependent
  57- 60  I4     ---     Field2   ?=-999 Object data field 2, catalog dependent
  61- 64  I4     ---     Field3   ?=-999 Object data field 3, catalog dependent
--------------------------------------------------------------------------------

Global Notes:
Note (G1): Type ranges from 1 to 15 and states whether an association
    was found in:
      * a stellar catalogue (bit 0 = 1)      (*)
      * an extragalactic catalog (bit 1 = 2) (*)
      * catalogs with other types of objects (bit 2 = 4) or
      * in a catalog with mixed types (bit 3 = 8).
    For example, if associations were found to both an extragalactic
    catalog and a stellar catalog, Type is set to 3 (2+1).

    (*) The meaning of these two bits are different from what is stated
        in the original documentation "fsc.txt" file.
--------------------------------------------------------------------------------

History:
  * Original description from Susan Gessner (NASA/GSFC) in 1989
   (see file "fsc.txt")
  * 24-Feb-1993: standardized documentation at CDS
  * 30-Jan-2009: description homogenized with other IRAS catalogs at CDS.
================================================================================
(End)                                                                24-Feb-1993
