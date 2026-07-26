II/275              IRAS Faint Source Reject Catalog          (IPAC 1992)
================================================================================
IRAS Faint Source Reject Catalog, Version 2.0 (1992 July)
     Moshir M., Kopan G., Conrow T., Hacking P., Gregorich D., Rohrbach G.,
     Melnyk M., Rice W., Fullmer L., White J., Chester T.
    <Explanatory Supplement to the IRAS Faint Source Survey, Version 2, 
     JPL D-10015 8/92, Infrared Processing and Analysis Center (IPAC),
     California Institute of Technology (1982)>
    =2008yCat.2275....0M
================================================================================
ADC_Keywords: Photometry, infrared ; Infrared sources ; Surveys
Mission_Name: IRAS

Description:

    The Faint Source Reject Catalog contains 593,516 sources rejected for
    inclusion in the Faint Source Catalog (Cat. II/156) because they
    failed to meet one or more of the criteria established to ensure the
    reliability of the FSC. The REJECTED sources in the FSR are either in
    confused regions of the sky, or in areas with |b|<10-20{deg}, or were
    detected only in a single band with a signal-to-noise ratio of 3-6.
    The FSR also includes sources from areas of the sky covered by fewer
    than six detector passes, and sources contaminated by or caused by
    cometary debris trails.

    The files described here contain selected columns from the original
    Faint Source Reject IRAS catalogue; the full set is available as a
    ascii FITS table. In the descriptions below, the original names of
    the columns are added at the end of the explanations of each column.


File Summary:
--------------------------------------------------------------------------------
 FileName    Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe          80        .   This file
fsr.dat        426   593516   Faint Source Reject Catalog
assoc.dat      120   318691   Associations of the point sources in FSR.
fsr_long.fit  2880   296808   The full Faint Source Reject Catalog
fsr_asso.fit  2880    15942   The full association file
--------------------------------------------------------------------------------

See also:
    http://lambda.gsfc.nasa.gov/product/iras/i_products.cfm : IRAS products at
          the Legacy Archive for Microwave Background Data Analysis (LAMBDA)
    http://lambda.gsfc.nasa.gov/product/iras/faint_source_rej.cfm : IRAS
          Faint Sources Rejected catalog
    II/125 : IRAS Point Source Catalog (IRAS-PSC)
    II/156 : IRAS Faint Source Survey (IRAS-FSS, sources FHHMMm+DDMM)
    II/126 : IRAS Serendipitous Survey Catalog (IRAS-SSC, sources SHHMMm+DDMMA)
    VII/73 : IRAS Small Scale Structure Catalog (IRAS-SSSC, sources XHHMM+DDd)
    II/274 : IRAS Point Source Reject Catalog (IRAS-PSR, sources RHHMMm+DDMM)
   III/197 : IRAS Low Resolution Spectra (IRAS-LRS)
    VII/91 : IRAS Asteroid and Comet Survey
    II/190 : IRAS Minor Planet Survey data base
   III/170 : IRAS Point Source Identifications (MacConnell, 1993)

Byte-by-byte Description of file: fsr.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 12  A12   ---     IRAS      IRAS name (NAME) (1)
  14- 21  F8.4  deg     RAdeg     Right Ascension B1950.0, epoch 1983.5 (RA)
  23- 30  F8.4  deg     DEdeg     Declination B1950.0, epoch 1983.5 (DEC)
  33- 38  I6    ---     FSSkey    Data Base Pointer (KEYFSS)
  40- 44  I5   10mas    uncMaj    Uncertainty ellipse major axis (UNCMAJOR)
  46- 49  I4   10mas    uncMin    Uncertainty ellipse minor axis (UNCMINOR)
  51- 54  I4   0.1deg   uncPA     Uncertainty ellipse position angle (POSANG)
  56- 58  I3    ---     N12       ? Observations at 12{mu} (NOBS_12) (2)
  60- 62  I3    ---     N25       ? Observations at 25{mu} (NOBS_25) (2)
  64- 66  I3    ---     N60       ? Observations at 60{mu} (NOBS_60) (2)
  68- 70  I3    ---     N100      ? Observations at 100{mu} (NOBS_100) (2)
  72- 81  E10.4 Jy      Fnu12     Flux density at 12{mu} (FNU_12) (5)
  83- 92  E10.4 Jy      Fnu25     Flux density at 25{mu} (FNU_25) (5)
  94-103  E10.4 Jy      Fnu60     Flux density at 60{mu} (FNU_60) (5)
 105-114  E10.4 Jy      Fnu100    Flux density at 100{mu} (FNU_100) (5)
     116  I1    ---   q_Fnu12     [1,3] Quality of Fnu12 (FQUAL_12) (4)
     118  I1    ---   q_Fnu25     [1,3] Quality of Fnu25 (FQUAL_25) (4)
     120  I1    ---   q_Fnu60     [1,3] Quality of Fnu60 (FQUAL_60) (4)
     122  I1    ---   q_Fnu100    [1,3] Quality of Fnu100 (FQUAL_100) (4)
 124-126  I3   10-3   e_Fnu12     Relative uncertainty of Fnu12 (RELUNC_12)
 128-130  I3   10-3   e_Fnu25     Relative uncertainty of Fnu25 (RELUNC_25)
 132-134  I3   10-3   e_Fnu60     Relative uncertainty of Fnu60 (RELUNC_60)
 136-138  I3   10-3   e_Fnu100    Relative uncertainty of Fnu100 (RELUNC_100)
 140-141  I2    %       Rel12     ? Source reliability at 12{mu} (RELIAB_12) (2)
 143-144  I2    %       Rel25     ? Source reliability at 25{mu} (RELIAB_25) (2)
 146-147  I2    %       Rel60     ? Source reliability at 60{mu} (RELIAB_60) (2)
 149-150  I2    %       Rel100    ? Source reliability at 100um (RELIAB_100) (2)
 152-161  E10.4 ---     SNR12     ? Signal/Noise ratio at 12{mu} (SNR_12) (2)
 163-172  E10.4 ---     SNR25     ? Signal/Noise ratio at 25{mu} (SNR_25) (2)
 174-183  E10.4 ---     SNR60     ? Signal/Noise ratio at 60{mu} (SNR_60) (2)
 185-194  E10.4 ---     SNR100    ? Signal/Noise ratio at 100{mu} (SNR_100) (2)
 196-205  E10.4 ---  locSNR12     ? Local SNR at 12{mu} (LOCSNR_12) (2)
 207-216  E10.4 ---  locSNR25     ? Local SNR at 25{mu} (LOCSNR_25) (2)
 218-227  E10.4 ---  locSNR60     ? Local SNR at 60{mu} (LOCSNR_60) (2)
 229-238  E10.4 ---  locSNR100    ? Local SNR at 100{mu} (LOCSNR_100) (2)
 240-241  I2    ---     Ncat      Number of catalog sources within 6' (CATNBR)
 243-244  I2    ---     Nx12      Number of 12{mu} extractions within 6arcmin
                                    (EXTNBR1_12)
 246-247  I2    ---     Nx25      Number of 25{mu} extractions within 6arcmin
                                    (EXTNBR1_25)
 249-250  I2    ---     Nx60      Number of 60{mu} extractions within 6arcmin
                                    (EXTNBR1_60)
 252-253  I2    ---     Nx100     Number of 100{mu} extractions within 6arcmin
                                    (EXTNBR1_100)
 255-256  I2    ---     Cir1      Number of cold extractions within 30'
                                    (CIRRUS1) (3)
 258-259  I2    ---     Cir2      Number of cold extractions within 20'
                                    (CIRRUS2) (3)
 261-262  I2    ---     Cir3      Number of cold extractions within 6'
                                    (CIRRUS3) (3)
 264-273  E10.4 Jy      Flim12    ? 12{mu} flux density upper-limit (FNULIM_12)
 275-284  E10.4 Jy      Flim25    ? 25{mu} flux density upper-limit (FNULIM_25)
 286-295  E10.4 Jy      Flim60    ? 60{mu} flux density upper-limit (FNULIM_60)
 297-306  E10.4 Jy      Flim100   ? 100um  flux density upper-limit (FNULIM_100)
 308-312  F5.3  ---     cor12     12{mu} flux correction (NONLCOR_12) (6)
 314-318  F5.3  ---     cor25     25{mu} flux correction (NONLCOR_25)(6)
 320-324  F5.3  ---     cor60     60{mu} flux correction (NONLCOR_60)(6)
 326-330  F5.3  ---     cor100    100um flux correction (NONLCOR_100)(6)
 332-335  I4    ---     Plate     Plate source extracted from  (PLATE)
 337-339  I3    pix     A12       ? Area above threshold at 12{mu} (AREA_12) (2)
 341-343  I3    pix     A25       ? Area above threshold at 25{mu} (AREA_25) (2)
 345-347  I3    pix     A60       ? Area above threshold at 60{mu} (AREA_60) (2)
 349-351  I3    pix     A100      ? Area above threshold at 100um (AREA_100) (2)
 353-358  F6.3  ---     NoisC12   ? 12{mu} noise correc. factor (NOISCOR_12) (2)
 360-365  F6.3  ---     NoisC25   ? 25{mu} noise correc. factor (NOISCOR_25) (2)
 367-372  F6.3  ---     NoisC60   ? 60{mu} noise correc. factor (NOISCOR_60) (2)
 374-379  F6.3  ---     NoisC100  ? 100um noise correc. factor (NOISCOR_100) (2)
 381-386  F6.3  ---     NoisR12   ? Ratio of 87% to 68% quantiles of flux
                                    distribution at 12{um} (NOISRAT_12) (2)
 388-393  F6.3  ---     NoisR25   ? Ratio of 87% to 68% quantiles of flux
                                    distribution at 25{um} (NOISRAT_25) (2)
 395-400  F6.3  ---     NoisR60   ? Ratio of 87% to 68% quantiles of flux
                                    distribution at 60{um} (NOISRAT_60) (2)
 402-407  F6.3  ---     NoisR100  ? Ratio of 87% to 68% quantiles of flux
                                    distribution at 100{um} (NOISRAT_100) (2)
 409-410  I2    ---     Ndup      Number of source duplicates  (NDUPS)
     412  I1    ---     Nov       Number of overlapping plates  (NOVRLAP)
 414-415  I2    ---     nID       Number of positional associations  (NID)
 417-418  I2    ---     Type      ? Type of associated object (IDTYPE) (G1)
 420-426  I7    ---     IDrec     ? Pointer to association record (IDKEY)
--------------------------------------------------------------------------------
Note (1): IRAS name starts by F (Faint source catalog, see Cat. II/256)
          or Z (rejected)
Note (2): values set to 0 or blank signify no detection in that particular
     wavelength band; locSNR values sometimes have the value 0.0, distinct
     from blank,  when there were too few observations in the pixel
     containing the peak signal to estimate the noise there.
Note (3): The term 'cold' used in the three cirrus flags means 100 mu-only,
     or source having the color index
        log(F_{nu}_(60{mu})/F_{nu}_(100{mu})) <= -0.75.
Note (4): 3=high quality, 2=moderate quality, 1=upper limit
Note (5): Non-color corrected flux density
Note (6): Non-linear flux density correction
--------------------------------------------------------------------------------

Byte-by-byte Description of file: assoc.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label  Explanations
--------------------------------------------------------------------------------
   1- 12  A12   ---     IRAS   Name of source in FSR (NAME)
  14- 19  I6    ---     FSSkey Pointer of source in FSR (KEYFSS)
  21- 27  I7    ---     IDrec  Pointer of this group of associations (IDRECNO)
  29- 30  I2    ---     nID    Total number of positional associations (NID)
  32- 33  I2    ---   m_IDrec  Sequential number of this association (IDSEQNO)
  35- 36  I2    ---     catno  Catalog number (CATNO)
  38- 52  A15   ---     Source Source identification (SOURCE)
  54- 58  A5    ---     Class  Source type or spectral class (TYPE)
  60- 68  F9.5  deg     RAdeg  Association Right Ascension, 1950.0 (A_RA)
  70- 78  F9.5  deg     DEdeg  Association Declination, 1950.0 (A_DEC)
  80- 81  I2    ---     Type   Type of associated object (IDTYPE) (G1)
  83- 85  I3    arcsec  win    Catalog-dependent association window (WINDOW)
  87- 90  I4 0.1arcsec  Dist   Distance of IRAS source to association (RADIUS)
  92- 95  I4    0.1deg  dPA    Position angle IRAS source to association (POS)
  97-100  I4 0.1arcsec  dMaj   Distance from IRAS source to association along
                               major axis of IRAS uncertainty ellipse (DSTMAJOR)
 102-105  I4 0.1arcsec  dMin   Distance from IRAS source to association along
                               minor axis of IRAS uncertainty ellipse (DSTMINOR)
 107-110  I4    ---     Field1 ?=-999 Object field#1, catalog dependant (FIELD1)
 112-115  I4    ---     Field2 ?=-999 Object field#2, catalog dependant (FIELD2)
 117-120  I4    ---     Field3 ?=-999 Object field#3, catalog dependant (FIELD3)
--------------------------------------------------------------------------------

Global Notes:
Note (G1): Type ranges from 1 to 15 and states whether an association
     was found in:
      * a stellar catalogue (1=bit#0)
      * an extragalactic catalog (2=bit#1)
      * catalogs with other types of objects (4=bit#2)
      * in a catalog with mixed types (8=bit#3)
     For example, if associations were found to both an extragalactic
     catalog and a stellar catalog, Type has a value of 3 (1+2)

History:
    The file was created from the original FSR_LONG.DATA FITS file,
    extracting the columns (fields) mentioned in the
    the fields 1,2,17,18,21-52,79-99,196-205,230-238,256,267,268,319,325,326
    only were kept.

================================================================================
(End)                                   Francois Ochsenbein [CDS]    20-Apr-2008
