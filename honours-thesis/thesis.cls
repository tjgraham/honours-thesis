% ---------------------------------------------------------------------------- %
% Honours Thesis                                                               %
% Document Class                                                               %
% ---------------------------------------------------------------------------- %

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2020/09/10 Honours Thesis Class]

\DeclareOption{draft}{\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{oneside}{\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{twoside}{\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax

\LoadClass[a4paper, 10pt]{memoir}



\RequirePackage{graphicx}
\RequirePackage{xcolor}

\definecolor{color1}{HTML}{051E42}
\definecolor{color2}{HTML}{354E71}
\definecolor{color3}{HTML}{420B10}
\definecolor{color4}{HTML}{841F27}



% ---------------------------------------------------------------------------- %
% Typography                                                                   %
% ---------------------------------------------------------------------------- %

% Page Layout
\settypeblocksize{580pt}{360pt}{*}

\if@twoside
    \setlrmargins{*}{*}{1.5}
\else
    \setlrmargins{*}{*}{1}
\fi

\setulmargins{*}{*}{1.5}

\checkandfixthelayout



% Fonts
\RequirePackage[T1]{fontenc}
\RequirePackage[tbtags]{kpfonts}

\RequirePackage{microtype}

\linespread{1.025}


% ---------------------------------------------------------------------------- %
% Abstract                                                                     %
% ---------------------------------------------------------------------------- %

\abstractintoc
\renewcommand{\abstractnamefont}{
    \Large
    \bfseries
    \sffamily
    \color{color1}
}


% ---------------------------------------------------------------------------- %
% Sections                                                                     %
% ---------------------------------------------------------------------------- %

% Chapter Style
\colorlet{chp-title-color}{color1}
\colorlet{chp-num-color}{color1}
\makechapterstyle{thesis}{%
    \renewcommand*{\chapnumfont}{%
        \Huge%
        \sffamily%
        \bfseries%
        \color{chp-num-color}%
    }%
    \renewcommand*{\chaptitlefont}{%
        \Huge%
        \sffamily%
        \bfseries%
        \color{chp-title-color}%
    }%
    \renewcommand*{\printchaptername}{}%
    \renewcommand*{\printchapternum}{
        \chapnumfont%
        \thechapter%
    }%
    \renewcommand*{\afterchapternum}{}%
    \renewcommand*{\printchaptertitle}[1]{%
        \hfill%
        \chaptitlefont%
        ##1%
    }%
    \renewcommand{\afterchaptertitle}{%
        \color{black}%
        \vskip%
        \onelineskip%
        \hrule%
        \vskip%
        \afterchapskip%
    }%
}

\chapterstyle{thesis}



% Section Style
\colorlet{sec-color}{color2}
\setsecheadstyle{%
    \Large%
    \sffamily%
    \bfseries%
    \raggedright%
    \color{sec-color}%
}

% Subsection Style
\colorlet{subsec-color}{color2}
\setsubsecheadstyle{%
    \large%
    \sffamily%
    \bfseries%
    \raggedright%
    \color{subsec-color}%
}

% Subsubsection Style
\colorlet{subsubsec-color}{color2}
\setsubsubsecheadstyle{%
    \normalsize%
    \sffamily%
    \bfseries%
    \raggedright%
    \color{subsubsec-color}%
}

% Paragraph Style
\colorlet{par-color}{black}
\setparaheadstyle{%
    \normalsize%
    \sffamily%
    \bfseries%
    \raggedright%
    \color{par-color}%
}

% Subparagraph Style
\colorlet{subpar-color}{black}
\setsubparaheadstyle{%
    \normalsize%
    \sffamily%
    \bfseries%
    \raggedright%
    \color{subpar-color}%
}



% ---------------------------------------------------------------------------- %
% Headers & Footers                                                            %
% ---------------------------------------------------------------------------- %

\nouppercaseheads

% Chapter Page Style
\makepagestyle{chapter}
    \makeevenhead{chapter}{}{}{}
    \makeoddhead{chapter}{}{}{}
    \makeevenfoot{chapter}{\thepage}{}{}
    \makeoddfoot{chapter}{}{}{\thepage}

% Plain Page Style
\makepagestyle{plain}
    \makeevenhead{plain}{}{}{}
    \makeoddhead{plain}{}{}{}
    \makeevenfoot{plain}{\thepage}{}{}
    \makeoddfoot{plain}{}{}{\thepage}

% Main Page Style
\makepagestyle{thesis}
    \makeheadrule{thesis}{\ifonlyfloats{0pt}{\textwidth}}{\normalrulethickness}
    \makeevenhead{thesis}{\ifonlyfloats{}{\sffamily\leftmark}}{}{}
    \makeoddhead{thesis}{}{}{\ifonlyfloats{}{\sffamily\rightmark}}
    \makeevenfoot{thesis}{\ifonlyfloats{}{\thepage}}{}{}
    \makeoddfoot{thesis}{}{}{\ifonlyfloats{}{\thepage}}
\makepsmarks{thesis}{%
    \createmark{chapter}{left}{shownumber}{}{. }
    \createmark{chapter}{right}{shownumber}{}{. }
}
    

\pagestyle{thesis}





% ---------------------------------------------------------------------------- %
% Mathematics                                                                  %
% ---------------------------------------------------------------------------- %

\RequirePackage[tbtags]{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}

\RequirePackage{etoolbox}
\RequirePackage{mathtools}
\RequirePackage{nicefrac}
\RequirePackage{thmtools}

% Main Theorem Style
\newtheoremstyle{thesis}%
    {\topsep}% Space Above
    {\topsep}% Space Below
    {\normalfont}% Body Font
    {}% Indent Amount
    {\bfseries\color{color4}}% Theorem Head Font
    {.}% Punctuation
    {.5em}% Space After Theorem Head
    {}% Theorem Head Spec

% Theorems
\theoremstyle{thesis}
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{algorithm}[theorem]{Algorithm}

% Proof Equation & Case Numbering
\newcounter{equationstore}
\AtBeginEnvironment{proof}{
    \setcounter{equationstore}{\value{equation}}
    \setcounter{equation}{0}
    \renewcommand{\theequation}{\roman{equation}}
}
\AtEndEnvironment{proof}{
    \setcounter{equation}{\value{equationstore}}
}

% ---------------------------------------------------------------------------- %
% Floats & Captions                                                            %
% ---------------------------------------------------------------------------- %

% Caption Style
\captiondelim{. }
\captionnamefont{%
    \small%
    \sffamily%
    \color{color4}%
}
\captiontitlefont{%
    \small%
    \sffamily%
    \mathversion{sf}%
}
\captionstyle[\centering]{}

% Sub-Caption Style
\subcaptionsize{\small}
\subcaptionlabelfont{\sffamily}
\subcaptionfont{\sffamily}

% Default Table Font 
\setfloatadjustment{table}{
    \small
    \sffamily
    \mathversion{sf}
}

% Default Figure Font
\setfloatadjustment{figure}{
    \small
    \sffamily
    \mathversion{sf}
}

% Allow Sub-Floats within Figure Environment
\newsubfloat{figure}



% ---------------------------------------------------------------------------- %
% Lists                                                                        %
% ---------------------------------------------------------------------------- %

\firmlists
\RequirePackage{enumitem}


% ---------------------------------------------------------------------------- %
% Tables                                                                       %
% ---------------------------------------------------------------------------- %

\RequirePackage{multirow}
\RequirePackage{makecell}

