II/328              AllWISE Data Release                           (Cutri+ 2013)
================================================================================
AllWISE Data Release
     Cutri R.M., Wright E.L., Conrow T., Fowler J.W., Eisenhardt P.R.M.,
     Grillmair C., Kirkpatrick J.D., Masci F., McCallon H.L., Wheelock S.L., 
     Fajardo-Acosta S., Yan L., Benford D., Harbut M., Jarrett T., Lake S., 
     Leisawitz D., Ressler M.E., Stanford S.A., Tsai C.-W., Liu F., Helou G., 
     Mainzer A., Gettngs D., Gonzalez A., Hoffman D., Marsh K.A., Padgett D., 
     Skrutskie M.F., Beck R., Papin M., Wittman M. 
    <IPAC/Caltech (2013)>
    =2014yCat.2328....0C
================================================================================
ADC_Keywords: Infrared sources ; Photometry, infrared ; Surveys
Mission_Name: WISE

Description:
    The Wide-field Infrared Survey Explorer (WISE; see Wright et al.
    2010AJ....140.1868W) is a NASA Medium Class Explorer mission that
    conducted a digital imaging survey of the entire sky in the 3.4, 4.6,
    12 and 22um mid-infrared bandpasses (hereafter W1, W2, W3 and W4).
    The AllWISE program extends the work of the successful Wide-field Infrared
    Survey Explorer mission by combining data from the cryogenic and
    post-cryogenic survey phases to form the most comprehensive view of the
    mid-infrared sky currently available. AllWISE has produced a new Source
    Catalog and Image Atlas with enhanced sensitivity and accuracy compared
    with earlier WISE data releases. Advanced data processing for AllWISE
    exploits the two complete sky coverages to measure source motions for
    each Catalog source, and to compile a massive database of light curves
    for those objects.

Acknowledging WISE in publications:
    Please include the following in any published material that makes use
    of the WISE data products:

   "This publication makes use of data products from the Wide-field
    Infrared Survey Explorer, which is a joint project of the University of
    California, Los Angeles, and the Jet Propulsion Laboratory/California
    Institute of Technology, funded by the National Aeronautics and Space
    Administration."


File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl  Records    Explanations
--------------------------------------------------------------------------------
ReadMe         80        .   This file
allwise.sam    553     1000  *Sample of the AllWISE data release
                             (among 747,634,026 sources)
                             (updated version, 16-Feb-2021)
--------------------------------------------------------------------------------
Note on allwise.sam : the catalog described here is a subset of the full AllWISE
     catalog available from IRSA (http://irsa.ipac.caltech.edu/) with a
     selection of the columns of the full catalog.

See also:
   http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/ : Explanatory
          Supplement to the AllWISE Data Release Products
   II/311 : The WISE All-Sky data Release (Cutri+ 2012)
   II/246 : The 2MASS All-Sky Catalog of Point Sources (Cutri+ 2003)


Byte-by-byte Description of file: allwise.sam
--------------------------------------------------------------------------------
Bytes Format Units      Label       Explanations
--------------------------------------------------------------------------------
   1- 19 A19   ---      AllWISE     WISE All-Sky Release Catalog name,
                                    based on J2000 position,
                                    <WISEA JHHMMSS.ss+DDMMSS.s> (designation)
  21- 31 F11.7 deg      RAJ2000     Right ascension (J2000)
  33- 43 F11.7 deg      DEJ2000     Declination (J2000)
  45- 51 F7.4  arcsec   eeMaj       Semi-major axis of the error ellipse (1)
  53- 59 F7.4  arcsec   eeMin       Semi-minor axis of the error ellipse (1)
  61- 65 F5.1  deg      eePA        Position angle of the error ellipse (1)
  67- 72 F6.3  mag      W1mag       ? W1 magnitude (3.35um)
  74- 79 F6.3  mag      W2mag       ? W2 magnitude (4.6um)
  81- 86 F6.3  mag      W3mag       ? W3 magnitude (11.6um)
  88- 93 F6.3  mag      W4mag       ? W4 magnitude (22.1um)
  95-100 F6.3  mag      Jmag        ? 2MASS J magnitude (1.25um)
 102-107 F6.3  mag      Hmag        ? 2MASS H magnitude (1.65um)
 109-114 F6.3  mag      Kmag        ? 2MASS Ks magnitude (2.17um)
 116-120 F5.3  mag    e_W1mag       ? Mean error on W1 magnitude
 122-126 F5.3  mag    e_W2mag       ? Mean error on W2 magnitude
 128-132 F5.3  mag    e_W3mag       ? Mean error on W3 magnitude
 134-138 F5.3  mag    e_W4mag       ? Mean error on W4 magnitude
 140-144 F5.3  mag    e_Jmag        ? Mean error on J magnitude
 146-150 F5.3  mag    e_Hmag        ? Mean error on H magnitude
 152-156 F5.3  mag    e_Kmag        ? Mean error on Ks magnitude
 158-165 F8.3  pix      wx          x-pixel coordinate on the Atlas Image
 167-174 F8.3  pix      wy          y-pixel coordinate on the Atlas Image
 176-194 I19   ---      ID          Unique source ID
 196-202 F7.1  ---      snr1        ? Signal to noise ratio for W1 filter
 204-212 E9.3  ---      chi2W1      ? Reduced {chi}^2^ of the W1 profile fit
 214-220 F7.1  ---      snr2        ? Signal to noise ratio for W2 filter
 222-230 E9.3  ---      chi2W2      ? Reduced {chi}^2^ of the W2 profile fit
 232-238 F7.1  ---      snr3        ? Signal to noise ratio for W3 filter
 240-248 E9.3  ---      chi2W3      ? Reduced {chi}^2^ of the W3 profile fit
 250-257 F8.1  ---      snr4        ? Signal to noise ratio for W4 filter
 259-267 E9.3  ---      chi2W4      ? Reduced {chi}^2^ of the W4 profile fit
 269-277 E9.3  ---      chi2        All-band combined reduced {chi}^2^
 279-279 I1    ---      nb          [1/4] Number of PSF components used
                                     simultaneously in the profile-fitting
 281-281 I1    ---      na          [0/1] Active deblending flag (1 if a
                                    detection was split into multiple sources)
 283-287 F5.3  ---      sat1        ? Saturated pixel fraction in W1 (w1sat)
 289-293 F5.3  ---      sat2        ? Saturated pixel fraction in W2 (w2sat)
 295-299 F5.3  ---      sat3        ? Saturated pixel fraction in W3 (w3sat)
 301-305 F5.3  ---      sat4        ? Saturated pixel fraction in W4 (w4sat)
 307-310 A4    ---      ccf         [0DHOPdhop] Contamination and confusion 
                                     flag, one per band (cc_flags) (2)
 312-312 I1    ---      ex          [0-5] Extended source flag (ext_flg) (3)
 314-317 A4    ---      var         Variability flag, one per band (var_flg) (4)
 319-321 I3    ---      nW1         ? Frame detection count in W1 (w1nm) (5)
 323-325 I3    ---      mW1         ? Integer frame coverage in W1 (w1m) (6)
 327-329 I3    ---      nW2         ? Frame detection count in W2 (w2nm) (5)
 331-333 I3    ---      mW2         ? Integer frame coverage in W2 (w2m) (6)
 335-337 I3    ---      nW3         ? Frame detection count in W3 (w3nm) (5)
 339-341 I3    ---      mW3         ? Integer frame coverage in W3 (w3m) (6)
 343-345 I3    ---      nW4         ? Frame detection count in W4 (w4nm) (5)
 347-349 I3    ---      mW4         ? Integer frame coverage in W4 (w4m) (6)
 351-354 I4    ---      satnum      Minimum sample at which saturation occurs
                                     in each band
 356-366 F11.7 deg      RA_pm       ? Right ascension at epoch MJD=55400.0
                                     (2010.5589) from the profile-fitting
                                     measurement model that includes motion.
 368-378 F11.7 deg      DE_pm       ? Declination at epoch MJD=55400.0
                                     (2010.5589) from the profile-fitting
                                     measurement model that includes motion.
 380-386 F7.4  arcsec e_RA_pm       ? Error on RA_pm
 388-394 F7.4  arcsec e_DE_pm       ? Error on DE_pm
 396-403 F8.4  arcsec   cosig_pm    ? co-sigma of the equatorial position
                                     uncertainties
 405-411 I7    mas/yr   pmRA        ? Apparent motion in RA (9)
 413-419 I7    mas/yr e_pmRA        ? Mean error on pmRA (9)
 421-427 I7    mas/yr   pmDE        ? Apparent motion in DE (9)
 429-435 I7    mas/yr e_pmDE        ? Mean error on pmDE (9)
 437-445 E9.3  ---      chi2W1_pm   ? Reduced {chi}^2^ of the W1 profile fit
                                     including motion estimation
 447-455 E9.3  ---      chi2W2_pm   ? Reduced {chi}^2^ of the W2 profile fit
                                     including motion estimation
 457-465 E9.3  ---      chi2W3_pm   ? Reduced {chi}^2^ of the W3 profile fit
                                     including motion estimation
 467-475 E9.3  ---      chi2W4_pm   ? Reduced {chi}^2^ of the W4 profile fit
                                     including motion estimation
 477-485 E9.3  ---      chi2pm      ? Combined reduced {chi}^2^ profile fit
                                     including motion estimation
 487-491 A5    ---      qpm         ? Motion estimation quality (pmcode) (7)
 493-496 A4    ---      qph         [ABCUXZ] Photometric quality flag (8)
 498-499 I2    ---      fdet        [0/15] Bit-encoded integer indicating bands
                                     in which a source has a w?snr>2 detection
 501-504 I4    ---      fmoon       Scattered moonlight contamination flag
 506-512 F7.3  ---      covW1       ? Mean pixel coverage in W1
 514-520 F7.3  ---      covW2       ? Mean pixel coverage in W2
 522-528 F7.3  ---      covW3       ? Mean pixel coverage in W3
 530-536 F7.3  ---      covW4       ? Mean pixel coverage in W4
 538-547 I10   ---      2Mkey       ? 2MASS PSC association (not identification)
 549-553 F5.3  arcsec   d2M         ? Distance separating the positions of the
                                     WISE source and associated 2MASS PSC source
--------------------------------------------------------------------------------

Note (1): the parameters of the error ellipse are computed from the
     1-{sigma} in RA and Dec (sigra, sigdec) and the co-{sigma} sigradec
     with the formulae:
     {Delta} = (sigra^2^-sigdec^2^)^2^ + 4*sigradec^2^
     eeMaj^2^ = 0.5*(sigra^2^+sigdec^2^+sqrt({Delta}))
     eeMin^2^ = 0.5*(sigra^2^+sigdec^2^-sqrt({Delta}))
     tan(eePA) = (eeMaj^2^-sigdec^2^)/(sigradec*|sigradec|)
               = (sigradec*|sigradec|)/(eeMaj^2^-sigra^2^)

     Conversely, the sigra/sigdec are given by:
     sigra^2^  = eeMaj^2^sin^2^(eePA) + eeMin^2^cos^2^(eePA)
     sigdec^2^ = eeMaj^2^cos^2^(eePA) + eeMin^2^sin^2^(eePA)

Note (2): One character per band (W1/W2/W3/W4) that indicates that the
     photometry and/or position measurements of a source may be
     contaminated or biased due to proximity to an image artifact:
 D,d =  Diffraction spike. Source may be a spurious detection of or
     contaminated by a diffraction spike from a nearby bright star on
     the same image
 P,p = Persistence. Source may be a spurious detection of or contaminated
       by a latent image left by a bright star
 H,h = Halo. Source photometry may be a spurious detection of or
       contaminated by the scattered light halo surrounding a nearby
       bright source
 O,o = (letter "o") Optical ghost. Source may be a spurious detection
     of or contaminated by an optical ghost image caused by a nearby bright
     source
 0  = (number zero) Source is unaffected by known artifacts.

Note (3): Extended source flag.
 0 = The source shape is consistent with a point-source and the source is not
     associated with or superimposed on a 2MASS XSC source
 1 - The profile-fit photometry goodness-of-fit, w?rchi2, is >3.0 in one or
     more bands.
 2 = The source falls within the extrapolated isophotal footprint of a
     2MASS XSC source.
 3 = The profile-fit photometry goodness-of-fit, w?rchi2, is >3.0 in one or
     more bands, and The source falls within the extrapolated isophotal
     footprint of a 2MASS XSC source.
 4 = The source position falls within 5" of a 2MASS XSC source.
> 5 = The profile-fit photometry goodness-of-fit, w?rchi2, is >3.0 in one or
>     more bands, and the source position falls within 5" of a 2MASS XSC source.

Note (4): The variability flag is a 4-character string, one character per
     band, containing a value related to the probability that the source
     flux measured on the individual WISE exposures is variable.
  * value "n" indicates insufficient or inadequate data to make a
    determination  (<6 exposures)
  * values 0 thru 9 indicate increasing probabilities of variation;
    - values 0-5 are most likely not variable,
    - values 6-7 are likely variables (but susceptible of false-positive
      variability)
    - values >7 have the highest probability of being true variables

Note (5): Integer frame detection count. This column gives the number of
     individual 7.7s exposures on which this source was detected with SNR>3
     in the W1 profile-fit measurement. This number can be zero for sources
     that are well-detected on the coadded Atlas Image, but too faint for
     detection on the single exposures.

Note (6): number of individual 7.7s W1 exposures on which a profile-fit
     measurement of this source was possible. This number can differ between
     the four bands because band-dependent criteria are used to select
     individual frames for inclusion in the coadded Atlas Images. This column
     is null if there is no frame coverage in this band at the position of
     this source.

Note (7): five character string that encodes information about factors that
     impact the accuracy of the motion estimation. These include the original
     blend count, whether a blend-swap took place, and the distance in
     hundredths of an arcsecond between the non-motion position and the
     motion mean-observation-epoch position. The format is NQDDD
     where N is the original blend count, Q is either "Y" or "N" for "yes" or
     "no" a blend-swap occurred (i.e., the original primary component was not
     the final primary component), and DDD is the radial distance between the
     non-motion and motion at mean-observation epoch positions in units of
     0.01 arcsec, clipped at 999 (almost 10 arcsec).

Note (8): Four character flag, one character per band [W1/W2/W3/W4], that
     provides a shorthand summary of the quality of the profile-fit photometry
     measurement in each band, as derived from the measurement signal-to-noise
     ratio:
     A = Source is detected in this band with a flux signal-to-noise ratio
         w?snr>10.
     B = Source is detected in this band with a flux signal-to-noise ratio
         3<w?snr<10.
     C = Source is detected in this band with a flux signal-to-noise ratio
         2<w?snr<3.
     U = Upper limit on magnitude. Source measurement has w?snr<2. The
         profile-fit magnitude w?mpro is a 95% confidence upper limit.
     X = A profile-fit measurement was not possible at this location in this
         band. The value of w?mpro and w?sigmpro will be "null" in this band.
     Z = A profile-fit source flux measurement was made at this location, but
         the flux uncertainty could not be measured. The value of w?sigmpro
         will be "null" in this band. The value of w?mpro will be "null" if
         the measured flux, w?flux, is negative, but will not be "null" if
         the flux is positive. If a non-null magnitude is present, it
         corresponds to the true flux, and not the 95% confidence upper limit.
         This occurs for a small number of sources found in a narrow range of
         ecliptic longitude which were covered by a large number of saturated
         pixels from 3-Band Cryo single-exposures.

Note (9): CAUTION - Note concerning "proper" motions:
         according to AllWISE documentation, the motions measured are not 
         proper motions because they are affected by parallax motions.
         More information is given in:
    http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_6.html\
    #How_To_Interpret
--------------------------------------------------------------------------------

History:
 From http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec2_1a.html

 * 05-Feb-2014: First version
 * 16-Feb-2021: Updated version 
    + Correction of column eePA: the position angle was wrongly computed as 
       being north-of-east instead of east-of-north.
      The correct position angle from the previous version is thus 90-eePA 
       (then transformed to match the range [0-180[ instead of ]-90, 90]).
    + Change the nW1, mW1, nW2, mW2, nW3, mW3, nW4, mW4 datatypes from one byte 
       to two bytes to cover the full range of possible values
       (affects less than 1% of all rows).

================================================================================
(End)             Laurent Cambresy, Patricia Vannier [CDS]           16-Feb-2021
