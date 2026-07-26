II/327              AKARI Far-Infrared Surveyor YSO catalog       (Toth+, 2014)
================================================================================
The AKARI Far-Infrared Surveyor young stellar object catalog.
    Toth L.V., Marton G., Zahorecz S., Balazs L.G., Ueno M., Tamura M.,
    Kawamura A., Kiss Z.T., Kitamura Y.
   <Publ. Astron. Soc. Jap., 66, 17 (2014)>
   =2014PASJ...66...17T
   =2014yCat.2327....0T
================================================================================
ADC_Keywords: Protostars ; Stars, pre-main sequence ; Stellar distribution ;
              Fundamental catalog
Keywords: catalogs - infrared: ISM - infrared: stars - ISM: bubbles -
          stars: formation

Abstract:
    We demonstrate the use of the AKARI all-sky survey photometric data in
    the study of galactic star formation. Our aim was to select young
    stellar objects (YSOs) in the AKARI Far-Infrared Surveyor (FIS) Bright
    Source Catalogue. We used AKARI/FIS and Wide-field Infrared Survey
    Explorer (WISE) data to derive mid- and far-infrared colors of YSOs.
    Classification schemes based on quadratic discriminant analysis (QDA)
    have been given for YSOs and the training catalog for QDA was the
    whole-sky selection of previously known YSOs (i.e., listed in the
    SIMBAD database). A new catalog of AKARI FIS YSO candidates including
    44001 sources has been prepared; the reliability of the classification
    is over 90%, as tested in comparison to known YSOs. As much as 76% of
    our YSO candidates are from previously uncatalogued types. The vast
    majority of these sources are Class I and II types according to the
    Lada classification. The distribution of AKARI FIS YSOs is well
    correlated with that of the galactic ISM; local over-densities were
    found on infrared loops and towards the cold clumps detected by
    Planck.

Description:
    AKARI FSC BSC data was combined with mid-IR photometric data from the
    WISE All-Sky Data Release. Classification was made with quadratic
    discriminant analysis based on the mid- and far-IR colours and
    brightness values. 44001 sources were classified as young stellar
    object candidate. For each candidate AKARI FIS ID, celestial position
    (RA2000, DEC2000), WISE magnitudes measured in passbands centered at
    3.4, 4.6, 12 and 22um (W1, W2, W3, W4), their corresponding errors
    (e_W1, e_W2, e_W3, e_W4), FIS flux densities measured in passbands
    with nominal wavelengths of 65, 90, 140 and 160um (F65, F90, F140,
    F160) and the corresponding flux qualities (F_QUAL), the probability
    of being YSO, SIMBAD main type and subtypes, the angular distance
    between the FIS source and the matching SIMBAD object and the SIMBAD
    ID of the matching object are listed.

File Summary:
--------------------------------------------------------------------------------
 FileName  Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe        80        .   This file
ysoc.dat     232    44001   Parameters of sources classified as YSO candidate
--------------------------------------------------------------------------------

See also:
    II/297 : AKARI/IRC mid-IR all-sky Survey (ISAS/JAXA, 2010)
    II/298 : AKARI/FIS All-Sky Survey Point Source Catalogues (ISAS/JAXA, 2010)
    II/311 : WISE All-Sky Data Release (Cutri+ 2012)

Byte-by-byte Description of file: ysoc.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label    Explanations
--------------------------------------------------------------------------------
   1- 14  A14   ---     AKARI    ID number from the AKARI FIS BSC (Cat. II/298)
                                   (objName, HHMMSSs+DDMMSS)
  16- 25  F10.6 deg     RAdeg    Right ascension (J2000.0)
  27- 36  F10.6 deg     DEdeg    Declination (J2000.0)
  38- 43  F6.3  mag     W1       WISE W1 magnitude
  45- 49  F5.3  mag   e_W1       Error of W1 magnitude
  51- 56  F6.3  mag     W2       WISE W2 magnitude
  58- 62  F5.3  mag   e_W2       Error of W2 magnitude
  64- 69  F6.3  mag     W3       WISE W3 magnitude
  71- 75  F5.3  mag   e_W3       Error of W3 magnitude
  77- 82  F6.3  mag     W4       WISE W4 magnitude
  84- 88  F5.3  mag   e_W4       Error of W4 magnitude
  90-100  F11.6 Jy      F65      AKARI N60 flux density (65um)
 102-112  F11.6 Jy      F90      AKARI WIDE-S flux density (90um)
 114-124  F11.6 Jy      F140     AKARI WIDE-L flux density (140um)
 126-137  F12.6 Jy      F160     AKARI N160 flux density (160um)
     139  I1    ---   q_F65      [1/3] 65um flux quality value (3=high quality)
     141  I1    ---   q_F90      [3] 90um flux quality value (3=high quality)
     143  I1    ---   q_F140     [3] 140um flux quality value (3=high quality)
     145  I1    ---   q_F160     [1/3] 160um flux quality value (3=high quality)
 147-154  F8.6  ---     Prob     [0.5/1] Probability of being a YSO candidate
 156-158  A3    ---     Type     Main Type of the source as listed in SIMBAD (1)
 160-189  A30   ---     Subtypes Subtypes of the source as listed in SIMBAD
 191-196  F6.3  arcsec  Dist     [0.1/30]?=0 Angular distance between the FIS 
                                   source and the matching SIMBAD object
 198-232  A35   ---     SIMBAD   SIMBAD ID of the matching SIMBAD object
--------------------------------------------------------------------------------
Note (1): The classification was done in late 2012, the paper was submitted in
  early 2013, while the preparation of the final form of the catalog was
  made in mid 2013. Meanwhile 4 of our YSO candidate sources were classified
  as galaxy in the SIMBAD which we did not remove. But in this catalog they
  are already listed as galaxy in the Main Type column.
--------------------------------------------------------------------------------

Acknowledgements:
    Gabor Marton , marton.gabor(at)csfk.mta.hu

================================================================================
(End)     Gabor Marton [Konkoly, Hungary], Patricia Vannier [CDS]    03-Jan-2014
