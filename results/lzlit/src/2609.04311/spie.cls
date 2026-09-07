\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{spie}[2015/08/14  v3.4 SPIE Proceedings class]
\LoadClassWithOptions{article}  
\DeclareOption{a4paper}{   \AtEndOfClass{       \typeout{a4paper used}
       }
   }

\DeclareOption{10pt}{\PassOptionsToClass{10pt}{article}}   
\DeclareOption{11pt}{\PassOptionsToClass{11pt}{article}}
\DeclareOption{12pt}{\PassOptionsToClass{12pt}{article}}

\ProcessOptions\relax      
\if0\@ptsize\typeout{ten-point font}\fi
\if1\@ptsize\typeout{eleven-point font}\fi
\if2\@ptsize\typeout{twelve-point font}\fi

\RequirePackage{authblk}
\RequirePackage{ifthen}
\RequirePackage[superscript]{cite}[2003/11/04] 
\topmargin 0.0in                \oddsidemargin -0.125in           \evensidemargin -0.125in          \textheight 8.74in                \textwidth 6.75in                 \headheight 0in \headsep 0in      \pagestyle{empty}                 \setlength{\parskip}{1ex plus 1ex minus 0.3ex} \date{}                           
\setlength{\abovecaptionskip}{0ex}
\setlength{\floatsep}{0.9ex plus 0.3ex minus 0.6ex} 
\setlength{\textfloatsep}{4ex plus 3ex minus 1.5ex}
\renewcommand{\textfraction}{0.10} 
\renewcommand{\floatpagefraction}{0.60} 
\renewcommand{\topfraction}{0.90} 
\renewcommand{\bottomfraction}{0.90}
\setcounter{totalnumber}{3} 
\setcounter{topnumber}{2} 
\setcounter{bottomnumber}{2} 

\def\ample{\large}   
\def\setamplefont{
\if0\@ptsize      \def\ample{\@setsize\large{12pt}\xipt\@xipt}
\fi

\if1\@ptsize      \def\ample{\@setsize\large{12pt}\xiipt\@xiipt}
\fi

\if2\@ptsize      \renewcommand{\ample}{\fontsize{13.2pt}{15.8pt}\selectfont\bfseries} 
  \ifcmr \font\ample = cmbx17 at 13.2truept  \fi
\fi  }


\def\titlefont{\normalfont\LARGE\bfseries}  \def\title#1{\gdef\@title{\titlefont #1}} 

\def\settitlefont{
\typeout{\rmdefault -font_in_spiecls}
\def\cmr{cmr}  \newif\ifcmr
\def\ptm{ptm}
\newif\ifptm
\ifthenelse{\equal{\rmdefault}{\cmr}}{\cmrtrue}{\cmrfalse}
\ifthenelse{\equal{\rmdefault}{\ptm}}{\ptmtrue}{\ptmfalse}
\if0\@ptsize      \renewcommand{\titlefont}{\fontsize{16pt}{19.2pt}\selectfont\bfseries}
  \ifcmr  \font\titlefont = cmbx12 at 16truept  \fi
\fi
\if2\@ptsize      \renewcommand{\titlefont}{\fontsize{19.2pt}{23pt}\selectfont\bfseries}
  \ifcmr  \font\titlefont = cmbx12 at 19.2truept  \fi
\fi  }

\def\authorinfo#1{\gdef\@authorinfo{#1}}
\authorinfo{}     
\renewcommand\@makefntext[1]{\leftskip=1.4em\hskip-0em\@makefnmark#1}


\def\maketitle{\par 
\settitlefont
\setamplefont
 \begingroup
   \def\thefootnote{\fnsymbol{footnote}}   \def\@makefnmark{\hbox
       to\z@{$\m@th^{\@thefnmark}$\hss}}   \if@twocolumn
     \twocolumn[\@maketitle]     \else \newpage
     \global\@topnum\z@
     \@maketitle \fi     
     \@thanks
 \endgroup
 \let\maketitle\relax
 \let\@maketitle\relax
 \gdef\@thanks{}\gdef\@author{}\gdef\@title{}\let\thanks\relax

\renewcommand{\footnotesize}{\small}  \renewcommand{\thefootnote}{\fnsymbol{footnote}}
\ifx\@authorinfo\empty \else\footnotetext[0]{\@authorinfo}\fi      
\renewcommand{\thefootnote}{\fnsymbol{footnote}}  

\renewcommand\@makefntext{\leftskip=0em\hskip1.4em\@makefnmark}
}




\def\and{\bigskip\\}    
 
\def\@maketitle{\newpage
 \null
\if0\@ptsize\vspace{-10mm}\else\vspace{-12mm}\fi
 \begin{center}  \ifcmr 
    \if0\@ptsize {\typeout{10pt}\setlength{\baselineskip}{19.2pt} \@title \par}  \fi
    \if1\@ptsize {\typeout{11pt}\setlength{\baselineskip}{20.4pt} \@title \par} \fi
    \if2\@ptsize {\typeout{12pt}\setlength{\baselineskip}{23pt} \@title \par} \fi
  \else 
    {\@title \par} 
  \fi 
  \vskip 3.5mm 
  {\large           \@author \par}     \vskip 1.5ex	
  {\large \@date} \end{center} \par
}


\def\sectfont{\centering\ample\bf}
\def\subsectfont{\raggedright\ample\bf}
\def\subsubsectfont{\raggedright\normalsize\bf}
\def\append{0}

\def\section{\@startsection{section}{1}{\z@}
   {-2.5ex plus -1ex minus -0.5ex}{0.2ex plus 0.5ex minus 0ex}{\sectfont}} 
\def\subsection{\@startsection{subsection}{2}{\z@}
   {-1.5ex plus -1ex minus -0.5ex}{0.1ex plus 0.1ex minus 0.1ex}{\subsectfont}}
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}
   {-1ex plus -1ex minus -0.5ex}{0.1ex plus 0.1ex}{\subsubsectfont}}

\def\@sect#1#2#3#4#5#6[#7]#8{\ifnum #2>\c@secnumdepth
     \let\@svsec\@empty
     \let\@svsecp\@empty
     \let\@svsub\@empty
     \else
     \refstepcounter{#1}
     \edef\@svsec{\csname the#1\endcsname\hskip 0.5em plus 0.3em}
     \edef\@svsecp{\csname the#1\endcsname.\hskip 0.3em plus 0.3em}
     \edef\@svsub{\csname the#1\endcsname\hskip 0.5em plus 0.3em}\fi
     \@tempskipa #5\relax
      \ifdim \@tempskipa>\z@
       \begingroup #6\relax
          \ifnum #2=1
             \ifnum \append=1 {\interlinepenalty \@M 
	        APPENDIX \@svsecp\uppercase{#8}\par}
	     \else {\interlinepenalty \@M \@svsecp\uppercase{#8}\par}
	     \fi
	  \else\ifnum #2=2
             \noindent{\interlinepenalty \@M \@svsub #8\par}
	  \else
             \noindent{\interlinepenalty \@M \@svsub #8\par}
	       \fi
	  \fi
       \endgroup
       \csname #1mark\endcsname{#7}\addcontentsline
         {toc}{#1}{\ifnum #2>\c@secnumdepth \else
                      \protect\numberline{\csname the#1\endcsname}\fi
                    #7}\else
        \def\@svsechd{#6\hskip #3\relax                     \@svsec #8\csname #1mark\endcsname
                      {#7}\addcontentsline
                           {toc}{#1}{\ifnum #2>\c@secnumdepth \else
                             \protect\numberline{\csname the#1\endcsname}\fi
                       #7}}\fi
     \@xsect{#5}}

\def\abstract{\section*{ABSTRACT}}
\def\endabstract{}

\def\keywords#1{
\par\vspace{0.5ex}{\noindent\normalsize\bf Keywords:} #1
\vspace{0.5ex}   }

\def\acknowledgments{\section*{ACKNOWLEDGMENTS}}
\def\endacknowledgments{}
\def\acknowledgements{\section*{ACKNOWLEDGMENTS}}
\def\endacknowledgements{}

\def\thebibliography#1{\section*{REFERENCES\@mkboth
 {REFERENCES}{REFERENCES}}\list
 {[{\hskip .02em}\arabic{enumi}{\hskip .05em}]}
 {\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
 \advance\leftmargin\labelsep \usecounter{enumi}}
 \def\newblock{\hskip .11em plus .33em minus .07em}
   \parskip -0.7ex plus 0.5ex minus 0ex   
\if0\@ptsize\else\small\fi       \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist

\def\@begintheorem#1#2{
   \par\noindent\bgroup{\sc #1\ #2. }\it\ignorespaces}
\def\@opargbegintheorem#1#2#3{
   \par\bgroup{\sc #1\ #2\ (#3). }\it\ignorespaces}
\def\@endtheorem{\egroup}
\def\proof{\par{\it Proof}. \ignorespaces}
\def\endproof{{\ \vbox{\hrule\hbox{   \vrule height1.3ex\hskip0.8ex\vrule}\hrule
  }}\par}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}

\long\def\@makecaption#1#2{  \vskip\abovecaptionskip
  \sbox\@tempboxa{{\footnotesize #1.\ }{\footnotesize #2}}  \ifdim \wd\@tempboxa >\hsize       {\footnotesize #1.\ }{\footnotesize #2 \par}
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}  \fi
  \vskip\belowcaptionskip}

\def\appendix{\def\append{1}
  \par
  \setcounter{section}{0}
  \setcounter{subsection}{0}
  \setcounter{subsubsection}{0}
  \def\thesection{\Alph{section}}  
  \def\thesubsection{\Alph{section}.\arabic{subsection}}
\def\thesubsubsection{
\Alph{section}.\arabic{subsection}.\arabic{subsubsection}}
}

