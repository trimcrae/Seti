%
% arXiv preprint adaptation of the IOP Publishing journal article class.
%
% Based on iopjournal.cls, Copyright 2025 IOP Publishing Ltd.
% The original file is distributed under the LaTeX Project Public License,
% version 1.3c or later. This modified file has a distinct name so that it is
% not confused with the maintained IOP Publishing class.
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iopjournal-arxiv}
  [2026/08/10 arXiv preprint adaptation of the IOP journal article template]

% Pass unknown options to the standard article class.
\newif\if@anonymous
\@anonymousfalse
\DeclareOption{anonymous}{\@anonymoustrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[a4paper]{article}

% Required packages.
\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage[colorlinks=true,allcolors=blue]{hyperref}

% Page layout. This retains the compact, single-column IOP-style text area,
% without journal branding, provisional volume data, or editorial dates.
\setlength\voffset{-25mm}
\setlength\headsep{10mm}
\setlength\footskip{12mm}
\setlength\marginparwidth{25mm}
\setlength\marginparsep{2mm}
\setlength\textwidth{153mm}
\setlength\textheight{255mm}
\setlength\parindent{15pt}
\pagestyle{plain}
\setlength{\oddsidemargin}{\dimexpr(\paperwidth-\textwidth)/2-1in\relax}
\setlength{\evensidemargin}{\oddsidemargin}

% Article type. The original journal name, Crossmark block, received date,
% and revised date are intentionally omitted from the preprint version.
\newcommand{\articletype}[1]{%
  \vspace*{-4mm}%
  \noindent{\scriptsize\sffamily\bfseries\MakeUppercase{#1}}\par
  \vspace{5mm}%
}

% Front matter.
\renewcommand{\title}[1]{%
  {\exhyphenpenalty=10000\hyphenpenalty=10000
   \fontsize{18}{21}\selectfont\noindent\raggedright
   \textsf{#1}\par}%
  \suppressfloats[t]%
}

\renewcommand{\author}[1]{%
  {\vspace{5mm}%
   \fontsize{10}{12}\selectfont\raggedright
   \if@anonymous Author list removed for anonymity\else #1\fi\par
   \vspace{3mm}}%
}

\newcommand{\affil}[1]{%
  {\fontsize{8}{10}\selectfont\raggedright
   \if@anonymous\phantom{#1}\else #1\fi\par}%
}

\newcommand{\email}[1]{%
  \vspace*{12pt}%
  {\fontsize{8}{10}\selectfont\raggedright
   \textbf{E-mail:} \if@anonymous\phantom{#1}\else #1\fi\par}%
  \vspace{3mm}%
}

\newcommand{\keywords}[1]{%
  {\fontsize{8}{10}\selectfont\raggedright
   \textbf{Keywords:} #1\par}%
}

\renewenvironment{abstract}{%
  \vspace{16pt plus3pt minus3pt}%
  {\color{gray}\hrule}%
  \vspace{4pt}%
  \noindent\fontsize{11}{12}\selectfont\textbf{Abstract}\par
  \normalfont\ignorespaces
}{%
  \par\vspace{3mm}{\color{gray}\hrule}%
}

% Section titles.
\renewcommand\section{\@startsection{section}{1}{\z@}%
  {-3.25ex\@plus -1ex\@minus -.2ex}%
  {1sp}%
  {\reset@font\normalsize\bfseries\raggedright}}

\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-3.25ex\@plus -1ex\@minus -.2ex}%
  {1sp}%
  {\reset@font\normalsize\itshape\raggedright}}

\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex\@minus -.2ex}%
  {-1em\@plus .2em}%
  {\reset@font\normalsize\itshape}}

% Back-matter helpers. Acknowledgments and identifying statements are removed
% automatically when the anonymous option is selected.
\newcommand{\ack}[1]{%
  \section*{Acknowledgments}%
  \if@anonymous Removed for anonymity\else #1\fi}

\newcommand{\funding}[1]{%
  \section*{Funding}%
  \if@anonymous Removed for anonymity\else #1\fi}

\newcommand{\data}[1]{%
  \section*{Data availability}%
  \if@anonymous Removed for anonymity\else #1\fi}

\newcommand{\roles}[1]{%
  \section*{Author contributions}%
  \if@anonymous Removed for anonymity\else #1\fi}

\newcommand{\suppdata}[1]{%
  \section*{Supplementary data}%
  #1}

% Figure and table captions.
\renewcommand{\@makecaption}[2]{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\fontsize{8}{10}\selectfont\textbf{#1.} #2}%
  \ifdim\wd\@tempboxa>\hsize
    \raggedright\fontsize{8}{10}\selectfont\textbf{#1.} #2\par
  \else
    \global\@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip
}

\let\ioparxiv@oldtabular\tabular
\renewcommand{\tabular}{\fontsize{8}{10}\selectfont\ioparxiv@oldtabular}

% Optional ORCID icon. Include an `orcid.pdf` file only if this command is used.
\newcommand{\orcid}[1]{%
  \href{https://orcid.org/#1}{\includegraphics[width=8pt]{orcid}}}

\endinput
