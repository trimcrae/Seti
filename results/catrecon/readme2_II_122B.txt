II/122B             UBV Photoelectric Catalog: data 1953-1985  (Mermilliod 1987)
================================================================================
UBV Photoelectric Photometry Catalogue (1986): I. The Original data
     Mermilliod J.C.
    <Astron. Astrophys. Suppl. Ser. 71, 413 (1987)>
    =1987A&AS...71..413M
================================================================================
ADC_Keywords: Photometry, UBV

Description:
    The catalogue collects all UBV photoelectric photometry data published
    since 1953 to the end of 1985 in the Johnson and Morgan system. In
    addition, data obtained in closely related systems using UBV filters
    have also been collected. All the data have been merged, yielding a
    final catalogue of 136719(*) entries concerning 87267 stars and
    components. The data have been taken from 1413 (UBV) and 127 (other)
    references found in 73 different periodicals. Great care has been
    taken to detect and correct all kinds of errors.

    (*) Two entries have been deleted in version 'B' of this catalogue

    The analysis of the data is published in 1987A&AS...71..119M

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl    Records    Explanations
--------------------------------------------------------------------------------
ReadMe         80          .    This file
ubv.dat        48     109293    Measurements in Johnson UBV system
ubva.dat       48      27425   *Measurements in closely related UBV filters
merged.dat     48     136717    Merged data from ubv and ubva
refs.dat       80       4986    References ordered by Number
refj.dat       80       5215    References ordered by Journal
dubious.dat    79       3700    Stars showing significant differences in UBV
                                   but not known to be variable
codes.txt      80       2462    Description of the LID numbering system
adc.txt       122        654    Document of Version 'A' of Catalogue
adc.tex        79        548    Document of Version 'A' in LaTeX
adc.sty        78         68    LaTeX definitions needed to process adc.tex
--------------------------------------------------------------------------------
Note on ubva.dat: see Note (1) of file "refs"
--------------------------------------------------------------------------------

See also:
    II/193 : UBV Photoelectric Catalog: Data 1986-1992 (Mermilliod 1994)

Byte-by-byte Description of file: ubv.dat ubva.dat merged.dat
--------------------------------------------------------------------------------
 Bytes  Format   Unit   Label   Explanation
--------------------------------------------------------------------------------
   2- 11  I10    ---    LID     Code number (see details in file "codes.txt")
      12  A1     ---    Rem     [1-8DQSTU] Remark on duplicity or
                                      identification of components (1)
      13  A1     ---    Var     [VD] Remark on variability
  15- 20  F6.3   mag    Vmag    [-1.5/21.5]? V magnitude
  22- 27  F6.3   mag    B-V     ? B-V colour index
  29- 34  F6.3   mag    U-B     ? U-B colour index
      37  A1     ---  n_UBV     [S/*] Flag on the number of measurements:
                                  S = Standard star
                                  / = Number is unknown
                                  * = a minimum number of measurements is given
  38- 40  I3     ---  o_UBV      ? Number of measurements
  44- 48  I5     ---  r_UBV      Reference key (>2000 for non-Johnson filters,
                                  see file refs)
--------------------------------------------------------------------------------
Note (1): the components are indicated by an extra digit;
     "D" stands for double
     "Q" for obvious misidentifications (two incompatible measurements
         of the same star in the same paper)
     "S" to "U" designates supplementary stars (example in Praesepe,
         KW 250 and KW 250s both exist)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: dubious.dat
--------------------------------------------------------------------------------
   Bytes Format  Units  Label    Explanations
--------------------------------------------------------------------------------
   2- 11  I10    ---    LID      Code number (see details in file "codes.txt")
      12  A1     ---    Rem      [1-8DQSTU] Duplicity as in file "ubv"
  14- 19  F6.3   mag    Vmag     ? V magnitude
  21- 26  F6.3   mag    B-V      ? B-V color
  28- 33  F6.3   mag    U-B      ? U-B color
      35  A1     ---  n_UBV      [S/*] Flag on the number of measurements,
                                       as in file "ubv"
  36- 38  I3     ---  o_UBV      ? Number of measurements
  41- 45  I5     ---  r_UBV      ? Reference key (>2000 for non-Johnson filters,
                                   see file refs)
      47  A1     ---  f_UBV      [*] '*' indicates a known error
  49- 80  A32    ---    Comment  Comment on inconsistency
--------------------------------------------------------------------------------

Byte-by-byte Description of file: refs.dat
--------------------------------------------------------------------------------
   Bytes Format  Units   Label    Explanations
--------------------------------------------------------------------------------
   1-  5  I5     ---     Ref      [1/73005] Reference number,
                                        repeated for a continuation record (1)
   9- 80  A72    ---     Text     Full reference
--------------------------------------------------------------------------------
Note (1): the first two digits of the reference number is related to
    the used UBV system:
    02xxx   BV from Cape UcBV system (Nicolet 1975; Catalogue II/27)
    14xxx   UBV(E) data from Eggen
    37xxx   UBViyz, Jennens and Helfer (1975)
    08xxx   UBV(RI), Johnson standards (Johnson and Morgan 1953) (*)
    68XXX   UBV(RI), Kunkel and Rydgren (1979) (*)
    72XXX   UBV(RI), Moffet and Barnes (1979) (*)
    73XXX   UBV(RI), Neckel and Chini (1980) (*)
    (*)     Various UBVRI systems, see Lanz 1986; Catalogue II/116
--------------------------------------------------------------------------------

Historical Notes:

  * Sept.1987: version 'A' by Anne C. Raugh (see "adc.txt",
               or "adc.tex")
    This version is included on the "Selected Astronomical Catalogs,
    Vol.1" CD-ROM, directory /photom/ubv

  * 11-Jul-1995: version 'B' (Francois Ochsenbein, CDS)
    The format was transformed to be fully compatible with catalogue
    II/193 (1986-1992 data): "STD" flag moved to column 37.
    The discrepant values were checked, leading to the following
    corrections for LID stars (the order is provided within
    parentheses when several measurements exist for a single star)
    (corrections already mentioned in version 'A' are preceded by (A))
    ------------------------------------------------------------------------
         Corrections in "ubv"  file
    ------------------------------------------------------------------------
        -006007270     shifted columns
    (A) -004611571 (2) deleted (was duplicated)
        -002306322    'I' in column "o_UBV" transformed to '1'
    (A)  003003360     could not be located in the reference #100, and was
                       deleted; it contained the values:
                       .01  + .42  - .01      2 1     100
    (A)  100000886 (1) Reference is 0, not corrected
         100025290     shifted columns
         100047627D    shifted columns
         100156201 (3) shifted columns
         100208786 (2) shifted columns
         100285968     shifted columns
    (A)  100306247 (2) B-V=0.966  (instead of 9.966)
         224670098 (1) B-V=0.0310 (the dot was missing)
         251380319 (2) B-V=0.05   (the dot was missing)
    (A)  5122800921(2) Ref=390    (instead of 3900)
    (A)  5122800921(3) Ref=401    (instead of 4010)
    (A)  5122800921(4) Ref=411    (instead of 4110)
    (A)  620102305     o_UBV=3    (instead of blank)
         642002248     shifted columns
    (A)  920330012     B-V=-0.29  (instead of 18.12)
         926752107     B-V=-0.22  (instead of -1.05 which is U-B color)
    ------------------------------------------------------------------------
         Corrections in "ubva"  file
    ------------------------------------------------------------------------
    (A)  100159560 (6) Ref=8015   (instead of 80150) and shifted columns
         600247163     '+' removed in B-V column
    ------------------------------------------------------------------------
         Corrections in "dubious"  file
    ------------------------------------------------------------------------
         318050165     Deleted, because U-B was not measured in ref. 49
    ------------------------------------------------------------------------
         Corrections in "refs"  file
    ------------------------------------------------------------------------
    (A)    100         Page=221   (instead of 223)
    (A)   1464         Page=254   (instead of 1272)
    ------------------------------------------------------------------------

  * 06-Nov-2006: ReadMe revisited

References:
  Jennens, P.A., and Helfer, H.L., 1975MNRAS.172..667J
  Johnson, H.L., and Morgan, W.W., 1953ApJ...117..313J
  Kunkel, W.E., and Rydgren, A.E., 1979AJ.....84..633K
  Lanz, T., 1986A&AS...65..195L
  Moffett, T.J., and Barnes, T.G., 1979AJ.....84..627M
  Neckel, Th., and Chini, R., 1980A&AS...39..411N
  Nicolet, B., 1975A&AS...22..239N (Cat. II/27)
================================================================================
(End)                   Francois Ochsenbein [CDS]  11-Jul-1995, rev. 06-Nov-2006
